"""Utilities for parsing LLM responses into structured Python objects.

Handles common LLM output quirks such as markdown code fences, trailing
commas, JavaScript-style comments, and partial JSON embedded in prose.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from nuwa.core.exceptions import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?([\s\S]*?)```",
    re.DOTALL,
)

_SINGLE_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_quirks(text: str) -> str:
    """Remove common LLM output artefacts that break ``json.loads``."""
    # Strip single-line JS comments (// …)
    text = _SINGLE_LINE_COMMENT_RE.sub("", text)
    # Strip trailing commas before } or ]
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _extract_json_substring(text: str) -> str:
    """Return the outermost JSON object or array found in *text*.

    Tries, in order:
    1. Markdown code-fence extraction.
    2. First ``{…}`` or ``[…]`` span (greedy brace/bracket matching).
    """
    # 1. Fenced code block
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # 2. First balanced { … } or [ … ]
    # Determine order by which delimiter appears first so that a raw JSON
    # array (starting with '[') is not incorrectly parsed as a dict.
    pairs: list[tuple[str, str]] = [("{", "}"), ("[", "]")]
    brace_pos = text.find("{")
    bracket_pos = text.find("[")
    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        pairs = [("[", "]"), ("{", "}")]
    for open_ch, close_ch in pairs:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    # Nothing found – return the full text and let the caller handle the error
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_json_response(text: str) -> dict | list:
    """Extract and parse JSON from an LLM text response.

    Supports:
    - Raw JSON strings
    - JSON wrapped in markdown ```json … ``` fences
    - Responses that contain prose *around* a JSON payload
    - Trailing commas, single-line ``//`` comments

    Parameters
    ----------
    text:
        The raw LLM output.

    Returns
    -------
    dict | list
        The parsed JSON value (object or array).

    Raises
    ------
    nuwa.core.exceptions.LLMError
        If no valid JSON can be extracted.
    """
    if not text or not text.strip():
        raise LLMError("LLM returned an empty response; expected JSON.")

    candidate = _extract_json_substring(text)
    candidate = _strip_quirks(candidate)

    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Last-ditch: try stripping the whole text after quirk removal
        fallback = _strip_quirks(text.strip())
        try:
            result = json.loads(fallback)
        except json.JSONDecodeError:
            logger.debug(
                "JSON parse failed. Original text (first 500 chars): %s",
                text[:500],
            )
            raise LLMError(
                f"Failed to parse JSON from LLM response: {exc}"
            ) from exc

    if not isinstance(result, (dict, list)):
        raise LLMError(
            f"Expected a JSON object or array, got {type(result).__name__}."
        )
    return result


def parse_structured(text: str, schema: type[T]) -> T:
    """Parse an LLM response and validate it against a Pydantic model.

    Parameters
    ----------
    text:
        Raw LLM output containing JSON.
    schema:
        A Pydantic ``BaseModel`` subclass to validate against.

    Returns
    -------
    T
        A validated instance of *schema*.

    Raises
    ------
    nuwa.core.exceptions.LLMError
        If parsing or validation fails.
    """
    data = parse_json_response(text)

    # If the schema expects an object but we got a list, try unwrapping
    if isinstance(data, list):
        raise LLMError(
            f"Expected a JSON object for {schema.__name__}, "
            f"but received an array with {len(data)} items."
        )

    try:
        return schema.model_validate(data)
    except PydanticValidationError as exc:
        logger.debug("Pydantic validation errors: %s", exc.errors())
        raise LLMError(
            f"LLM response failed {schema.__name__} validation: {exc}"
        ) from exc
