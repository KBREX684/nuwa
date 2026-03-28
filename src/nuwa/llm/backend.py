"""LiteLLM-based backend implementing the ModelBackend protocol.

Provides a unified async interface for LLM completions (plain text and
structured / JSON-validated) with automatic retries, exponential back-off
with jitter for rate-limit errors, circuit breaker pattern, and
token-usage logging.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from nuwa.core.defaults import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    LLM_BACKOFF_MULTIPLIER,
    LLM_INITIAL_BACKOFF_S,
    LLM_MAX_RETRIES,
)
from nuwa.core.exceptions import LLMError
from nuwa.llm.response_parser import parse_structured

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retry configuration
_MAX_RETRIES: int = LLM_MAX_RETRIES
_INITIAL_BACKOFF_S: float = LLM_INITIAL_BACKOFF_S
_BACKOFF_MULTIPLIER: float = LLM_BACKOFF_MULTIPLIER

# Circuit breaker thresholds
_CB_FAILURE_THRESHOLD: int = 5
_CB_RECOVERY_TIMEOUT_S: float = 60.0

# LiteLLM exceptions that warrant a retry
_RETRYABLE_ERROR_NAMES = {
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
}
_UNSUPPORTED_RESPONSE_FORMAT_ERRORS = {
    "BadRequestError",
    "NotFoundError",
}


class LiteLLMBackend:
    """Async LLM backend powered by LiteLLM.

    Parameters
    ----------
    model:
        Model identifier in LiteLLM format, e.g. ``"openai/gpt-4o"``.
    api_key:
        Optional API key override.  When *None*, LiteLLM falls back to
        environment variables.
    base_url:
        Optional custom base URL (e.g. for Azure or self-hosted endpoints).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the completion response.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Cumulative token counters for the lifetime of this backend instance
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

        # Circuit breaker state
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0  # monotonic timestamp

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send *messages* to the LLM and return the assistant's text reply.

        Parameters
        ----------
        messages:
            A list of chat-message dicts with ``role`` and ``content`` keys.
        **kwargs:
            Extra keyword arguments forwarded to ``litellm.acompletion``
            (e.g. ``stop``, ``top_p``).

        Returns
        -------
        str
            The text content of the first choice in the completion.

        Raises
        ------
        nuwa.core.exceptions.LLMError
            If the call fails after all retry attempts.
        """
        response = await self._call_with_retry(messages, **kwargs)
        self._log_usage(response)

        try:
            content: str = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise LLMError("LLM response did not contain a valid message.") from exc

        if content is None:
            raise LLMError("LLM returned a null content field.")
        return content

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: type[T],
        **kwargs: Any,
    ) -> T:
        """Complete and validate the response against a Pydantic schema.

        The method first attempts to use the provider-native
        ``response_format`` parameter (JSON-mode / structured outputs).  If
        the provider does not support it, it falls back to extracting JSON
        from the plain-text response and validating with Pydantic.

        Parameters
        ----------
        messages:
            Chat messages.
        response_schema:
            A Pydantic ``BaseModel`` subclass describing the expected output.
        **kwargs:
            Extra arguments forwarded to ``litellm.acompletion``.

        Returns
        -------
        T
            A validated instance of *response_schema*.
        """
        # Try native structured output first
        try:
            response = await self._call_with_retry(
                messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            self._log_usage(response)
            raw_text = response.choices[0].message.content or ""
            return parse_structured(raw_text, response_schema)
        except Exception as exc:
            if isinstance(exc, LLMError):
                # Structured parse failed on native JSON mode output; fall back
                logger.debug(
                    "Native JSON mode returned unparseable output; retrying "
                    "with plain-text extraction."
                )
            elif type(exc).__name__ in _UNSUPPORTED_RESPONSE_FORMAT_ERRORS:
                # Provider does not support response_format — fall back
                logger.debug(
                    "Provider does not support response_format; "
                    "falling back to plain-text JSON extraction."
                )
            else:
                raise

        # Fallback: plain completion + parse
        raw_text = await self.complete(messages, **kwargs)
        return parse_structured(raw_text, response_schema)

    # --------------------------------------------------------------------- #
    #  Internals                                                             #
    # --------------------------------------------------------------------- #

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Call ``litellm.acompletion`` with exponential-backoff retries and circuit breaker.

        Only rate-limit and transient service errors are retried.  Other
        exceptions propagate immediately.  A circuit breaker prevents
        hammering a failing API.
        """
        # Circuit breaker: check if open
        if self._cb_failures >= _CB_FAILURE_THRESHOLD:
            if time.monotonic() < self._cb_open_until:
                raise LLMError(
                    f"Circuit breaker open: {_CB_FAILURE_THRESHOLD} consecutive "
                    f"failures. Retrying after {_CB_RECOVERY_TIMEOUT_S:.0f}s."
                )
            # Half-open: allow one attempt
            logger.info("Circuit breaker half-open: attempting recovery call.")

        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                call_kwargs = dict(kwargs)
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=call_kwargs.pop("temperature", self.temperature),
                    max_tokens=call_kwargs.pop("max_tokens", self.max_tokens),
                    **call_kwargs,
                )
                # Success: reset circuit breaker
                self._cb_failures = 0
                return response
            except Exception as exc:
                if type(exc).__name__ in _RETRYABLE_ERROR_NAMES:
                    last_exc = exc
                    self._cb_failures += 1
                    if self._cb_failures >= _CB_FAILURE_THRESHOLD:
                        self._cb_open_until = time.monotonic() + _CB_RECOVERY_TIMEOUT_S
                        logger.error(
                            "Circuit breaker tripped after %d failures. Blocking calls for %.0fs.",
                            self._cb_failures,
                            _CB_RECOVERY_TIMEOUT_S,
                        )
                    if attempt < _MAX_RETRIES:
                        jitter = random.uniform(0, backoff * 0.5)
                        delay = backoff + jitter
                        logger.warning(
                            "LLM call attempt %d/%d failed with %s: %s - "
                            "retrying in %.1fs (jitter +%.1fs)",
                            attempt,
                            _MAX_RETRIES,
                            type(exc).__name__,
                            exc,
                            backoff,
                            jitter,
                        )
                        await asyncio.sleep(delay)
                        backoff *= _BACKOFF_MULTIPLIER
                    continue
                raise LLMError(f"LLM call failed with non-retryable error: {exc}") from exc

        raise LLMError(f"LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_exc}")

    def _log_usage(self, response: Any) -> None:
        """Extract and log token-usage information from a completion response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        logger.info(
            "Token usage — prompt: %d, completion: %d, total: %d "
            "(cumulative prompt: %d, cumulative completion: %d)",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            self._total_prompt_tokens,
            self._total_completion_tokens,
        )
