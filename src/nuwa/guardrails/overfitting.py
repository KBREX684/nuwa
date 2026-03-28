"""Overfitting guardrail for the Nuwa AI Trainer framework.

Detects when the gap between training and validation scores grows too large,
indicating that mutations are over-specialising to the training set.
"""

from __future__ import annotations

import logging

from nuwa.core.types import GuardrailVerdict, RoundResult
from nuwa.guardrails.base import BaseGuardrail

logger = logging.getLogger(__name__)


class OverfittingGuardrail(BaseGuardrail):
    """Flags rounds where the train/val score gap indicates overfitting.

    Two independent triggers exist:

    1. **Absolute threshold** -- if ``train_score - val_score > threshold``
       for the most recent round, overfitting is flagged immediately.
    2. **Trend detection** -- if the gap has *widened* for
       ``trend_rounds`` consecutive rounds the guardrail triggers, even when
       the absolute gap is still below *threshold*.  This catches slow
       drift before it becomes severe.

    Args:
        threshold: Maximum acceptable absolute gap between the training and
            validation mean scores.
        trend_rounds: Number of consecutive rounds of widening gap required
            to trigger the trend detector.
        guardrail_name: Optional human-readable name override.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        trend_rounds: int = 2,
        guardrail_name: str | None = None,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name or "OverfittingGuardrail")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if trend_rounds < 1:
            raise ValueError(f"trend_rounds must be >= 1, got {trend_rounds}")
        self._threshold = threshold
        self._trend_rounds = trend_rounds

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _gap(rr: RoundResult) -> float | None:
        """Return ``train_mean - val_mean``, or *None* when val is missing."""
        if rr.val_scores is None:
            return None
        return rr.train_scores.mean_score - rr.val_scores.mean_score

    def _detect_trend(self, history: list[RoundResult]) -> bool:
        """Return *True* when the gap has widened for *trend_rounds* consecutive rounds."""
        # Collect the tail of valid gaps.
        gaps: list[float] = []
        for rr in reversed(history):
            g = self._gap(rr)
            if g is None:
                break  # cannot assess trend through a missing-val gap
            gaps.append(g)
            if len(gaps) > self._trend_rounds:
                break  # we have enough

        # Reverse so they are in chronological order.
        gaps.reverse()

        if len(gaps) <= self._trend_rounds:
            # Not enough history to evaluate a trend.
            return False

        # Check that each successive gap is strictly larger than the previous.
        widening_count = 0
        for i in range(1, len(gaps)):
            if gaps[i] > gaps[i - 1]:
                widening_count += 1
            else:
                widening_count = 0  # reset streak
            if widening_count >= self._trend_rounds:
                return True

        return False

    # -- protocol ------------------------------------------------------------

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Evaluate overfitting risk based on the training history.

        Args:
            history: All round results accumulated so far.

        Returns:
            A :class:`GuardrailVerdict` with ``passed=False`` when
            overfitting is detected.
        """
        if not history:
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason="No history available; nothing to check.",
            )

        latest = history[-1]
        gap = self._gap(latest)

        if gap is None:
            logger.debug(
                "%s: round %d has no validation scores; skipping.",
                self.name,
                latest.round_num,
            )
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason=(
                    f"Round {latest.round_num} has no validation scores; "
                    "overfitting check skipped."
                ),
            )

        # --- Trigger 1: absolute threshold -----------------------------------
        if gap > self._threshold:
            reason = (
                f"Overfitting detected in round {latest.round_num}: "
                f"train/val gap {gap:.4f} exceeds threshold {self._threshold:.4f} "
                f"(train={latest.train_scores.mean_score:.4f}, "
                f"val={latest.val_scores.mean_score:.4f})."  # type: ignore[union-attr]
            )
            logger.warning("%s: %s", self.name, reason)
            return GuardrailVerdict(
                passed=False,
                should_stop=False,
                guardrail_name=self.name,
                reason=reason,
            )

        # --- Trigger 2: trend detection --------------------------------------
        if self._detect_trend(history):
            reason = (
                f"Overfitting trend detected: the train/val gap has widened for "
                f"{self._trend_rounds} consecutive rounds (current gap={gap:.4f}). "
                f"Consider reverting or reducing mutation aggressiveness."
            )
            logger.warning("%s: %s", self.name, reason)
            return GuardrailVerdict(
                passed=False,
                should_stop=False,
                guardrail_name=self.name,
                reason=reason,
            )

        logger.debug(
            "%s: round %d OK — gap=%.4f (threshold=%.4f).",
            self.name,
            latest.round_num,
            gap,
            self._threshold,
        )
        return GuardrailVerdict(
            passed=True,
            guardrail_name=self.name,
            reason=(
                f"Round {latest.round_num}: train/val gap {gap:.4f} "
                f"within threshold {self._threshold:.4f}."
            ),
        )
