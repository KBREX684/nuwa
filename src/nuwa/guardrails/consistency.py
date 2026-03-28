"""Consistency guardrail for the Nuwa AI Trainer framework.

Detects unstable training behaviour by measuring the coefficient of
variation (CV) of validation scores across recent rounds and flagging
wild oscillations that suggest the mutation strategy is not converging.
"""

from __future__ import annotations

import logging
import math

from nuwa.core.types import GuardrailVerdict, RoundResult
from nuwa.guardrails.base import BaseGuardrail

logger = logging.getLogger(__name__)


class ConsistencyGuardrail(BaseGuardrail):
    """Flags instability when validation scores oscillate across rounds.

    Two complementary checks are performed:

    1. **Intra-round variance** -- if the latest round's individual
       validation result scores have a coefficient of variation (CV)
       exceeding ``1 - threshold`` the round is considered internally
       noisy.
    2. **Inter-round oscillation** -- the CV of validation *mean scores*
       across the most recent ``min_rounds`` rounds is computed.  If it
       exceeds ``1 - threshold`` the training trajectory is flagged as
       unstable.

    Both checks must have enough data to be meaningful; when insufficient
    history exists the guardrail passes silently.

    Args:
        threshold: Stability threshold in ``[0, 1]``.  Higher values demand
            more consistency.  The maximum acceptable CV is derived as
            ``1 - threshold`` (e.g. threshold=0.8 allows CV up to 0.2).
        min_rounds: Minimum number of recent rounds (with validation
            scores) required before the inter-round oscillation check
            activates.
        guardrail_name: Optional human-readable name override.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        min_rounds: int = 2,
        guardrail_name: str | None = None,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name or "ConsistencyGuardrail")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if min_rounds < 1:
            raise ValueError(f"min_rounds must be >= 1, got {min_rounds}")
        self._threshold = threshold
        self._min_rounds = min_rounds
        self._max_cv = 1.0 - threshold

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _cv(values: list[float]) -> float:
        """Compute the coefficient of variation for *values*.

        Returns 0.0 when the mean is zero or only a single value is provided,
        avoiding division-by-zero errors.
        """
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        if mean == 0.0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return math.sqrt(variance) / abs(mean)

    def _check_intra_round(self, rr: RoundResult) -> str | None:
        """Return a failure reason if the latest round's results are too noisy."""
        if rr.val_scores is None or not rr.val_scores.results:
            return None

        scores = [r.score for r in rr.val_scores.results]
        if len(scores) < 2:
            return None

        cv = self._cv(scores)
        if cv > self._max_cv:
            return (
                f"Intra-round instability in round {rr.round_num}: "
                f"validation result CV={cv:.4f} exceeds max allowed "
                f"CV={self._max_cv:.4f} (threshold={self._threshold:.2f})."
            )
        return None

    def _check_inter_round(self, history: list[RoundResult]) -> str | None:
        """Return a failure reason if recent round means oscillate too much."""
        # Collect the most recent validation mean scores.
        means: list[float] = []
        for rr in reversed(history):
            if rr.val_scores is not None:
                means.append(rr.val_scores.mean_score)
            if len(means) >= self._min_rounds:
                break

        if len(means) < self._min_rounds:
            return None

        # Restore chronological order (not strictly necessary for CV, but
        # useful for logging clarity).
        means.reverse()

        cv = self._cv(means)
        if cv > self._max_cv:
            scores_str = ", ".join(f"{m:.4f}" for m in means)
            return (
                f"Inter-round oscillation detected across last "
                f"{len(means)} rounds: val mean scores=[{scores_str}], "
                f"CV={cv:.4f} exceeds max allowed CV={self._max_cv:.4f} "
                f"(threshold={self._threshold:.2f})."
            )
        return None

    # -- protocol ------------------------------------------------------------

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Evaluate training consistency based on recent validation scores.

        Args:
            history: All round results accumulated so far.

        Returns:
            A :class:`GuardrailVerdict` with ``passed=False`` when score
            instability is detected.
        """
        if not history:
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason="No history available; nothing to check.",
            )

        latest = history[-1]

        # --- Intra-round check -----------------------------------------------
        intra_reason = self._check_intra_round(latest)
        if intra_reason is not None:
            logger.warning("%s: %s", self.name, intra_reason)
            return GuardrailVerdict(
                passed=False,
                should_stop=False,
                guardrail_name=self.name,
                reason=intra_reason,
            )

        # --- Inter-round check -----------------------------------------------
        inter_reason = self._check_inter_round(history)
        if inter_reason is not None:
            logger.warning("%s: %s", self.name, inter_reason)
            return GuardrailVerdict(
                passed=False,
                should_stop=False,
                guardrail_name=self.name,
                reason=inter_reason,
            )

        logger.debug(
            "%s: round %d consistency OK (threshold=%.2f).",
            self.name,
            latest.round_num,
            self._threshold,
        )
        return GuardrailVerdict(
            passed=True,
            guardrail_name=self.name,
            reason=(
                f"Round {latest.round_num}: consistency checks passed "
                f"(threshold={self._threshold:.2f})."
            ),
        )
