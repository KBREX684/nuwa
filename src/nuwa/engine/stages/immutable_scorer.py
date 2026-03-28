"""Immutable scorer -- tamper-proof, rule-based metrics.

Inspired by autoresearch's approach of keeping evaluation logic separate from
LLM-generated code so it cannot be gamed by prompt optimisation.  These
deterministic checks complement the subjective LLM-based scoring in
:mod:`~nuwa.engine.stages.evaluation`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from nuwa.core.types import AgentResponse, EvalSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (may be overridden via constructor)
# ---------------------------------------------------------------------------
_DEFAULT_MIN_LENGTH = 1
_DEFAULT_MAX_LENGTH = 10_000
_DEFAULT_MAX_LATENCY_MS = 30_000.0


class ImmutableScorer:
    """Deterministic, rule-based scorer for objective metrics.

    Unlike the LLM judge these checks are fully deterministic and cannot be
    influenced by prompt mutations.  The class is intentionally kept *outside*
    the agent's optimisation loop.

    Parameters
    ----------
    min_length:
        Minimum acceptable character count for agent output.
    max_length:
        Maximum acceptable character count for agent output.
    max_latency_ms:
        Upper bound on acceptable response latency in milliseconds.
    required_keywords:
        Optional list of keywords that must appear in the output.
    expected_format:
        One of ``"json"``, ``"regex:<pattern>"``, or ``None`` (no format
        check).
    """

    def __init__(
        self,
        *,
        min_length: int = _DEFAULT_MIN_LENGTH,
        max_length: int = _DEFAULT_MAX_LENGTH,
        max_latency_ms: float = _DEFAULT_MAX_LATENCY_MS,
        required_keywords: list[str] | None = None,
        expected_format: str | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.max_latency_ms = max_latency_ms
        self.required_keywords = required_keywords or []
        self.expected_format = expected_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        sample: EvalSample,
        response: AgentResponse,
    ) -> dict[str, float]:
        """Return a dict of ``metric_name -> score`` (each in ``[0, 1]``).

        Metrics
        -------
        format_compliance
            1.0 if the output matches the expected format, else 0.0.
        length_compliance
            1.0 if within bounds; degrades linearly when outside.
        keyword_presence
            Fraction of required keywords found in the output.
        response_time
            1.0 when latency <= threshold; linearly degrades to 0.0 at 2x.
        """
        output = response.output_text

        return {
            "format_compliance": self._check_format(output),
            "length_compliance": self._check_length(output),
            "keyword_presence": self._check_keywords(output),
            "response_time": self._check_latency(response.latency_ms),
        }

    def aggregate(self, metrics: dict[str, float]) -> float:
        """Return the unweighted mean of all metric scores."""
        if not metrics:
            return 0.0
        return sum(metrics.values()) / len(metrics)

    # ------------------------------------------------------------------
    # Individual metric implementations
    # ------------------------------------------------------------------

    def _check_format(self, output: str) -> float:
        """Check whether *output* matches the expected format."""
        if self.expected_format is None:
            return 1.0

        if self.expected_format == "json":
            try:
                json.loads(output)
                return 1.0
            except (json.JSONDecodeError, TypeError):
                return 0.0

        if self.expected_format.startswith("regex:"):
            pattern = self.expected_format[len("regex:"):]
            try:
                return 1.0 if re.search(pattern, output) else 0.0
            except re.error as exc:
                logger.warning("Invalid regex in expected_format: %s", exc)
                return 1.0  # Don't penalise for bad config

        # Unknown format specification -- default to pass
        logger.warning("Unknown expected_format %r; skipping check.", self.expected_format)
        return 1.0

    def _check_length(self, output: str) -> float:
        """Score length compliance with linear degradation outside bounds."""
        length = len(output)

        if self.min_length <= length <= self.max_length:
            return 1.0

        if length < self.min_length:
            return max(0.0, length / self.min_length) if self.min_length > 0 else 0.0

        # length > max_length -- degrade linearly, reaching 0 at 2x max
        overshoot = length - self.max_length
        margin = self.max_length if self.max_length > 0 else 1
        return max(0.0, 1.0 - overshoot / margin)

    def _check_keywords(self, output: str) -> float:
        """Fraction of required keywords present (case-insensitive)."""
        if not self.required_keywords:
            return 1.0

        lower_output = output.lower()
        found = sum(1 for kw in self.required_keywords if kw.lower() in lower_output)
        return found / len(self.required_keywords)

    def _check_latency(self, latency_ms: float) -> float:
        """1.0 when within budget; degrades to 0.0 at 2x the budget."""
        if latency_ms <= self.max_latency_ms:
            return 1.0

        overshoot = latency_ms - self.max_latency_ms
        margin = self.max_latency_ms if self.max_latency_ms > 0 else 1
        return max(0.0, 1.0 - overshoot / margin)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ImmutableScorer(min_length={self.min_length}, "
            f"max_length={self.max_length}, "
            f"max_latency_ms={self.max_latency_ms}, "
            f"keywords={self.required_keywords!r}, "
            f"format={self.expected_format!r})"
        )
