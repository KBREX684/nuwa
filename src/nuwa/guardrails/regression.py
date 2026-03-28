"""Regression guardrail for the Nuwa AI Trainer framework.

Detects when a mutation causes the validation score to drop below the
historical best by more than a configurable tolerance, recommending a
rollback to the best-known configuration.
"""

from __future__ import annotations

import logging

from nuwa.core.types import GuardrailVerdict, RoundResult
from nuwa.guardrails.base import BaseGuardrail

logger = logging.getLogger(__name__)


class RegressionGuardrail(BaseGuardrail):
    """Flags rounds where the validation score regresses beyond tolerance.

    The guardrail tracks the best validation mean score observed across all
    rounds in *history*.  If the most recent round's validation score drops
    below ``best - tolerance``, the verdict recommends rolling back to the
    configuration that produced the best score.

    Args:
        tolerance: Maximum acceptable drop from the historical best
            validation score before a regression is flagged.
        guardrail_name: Optional human-readable name override.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        guardrail_name: str | None = None,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name or "RegressionGuardrail")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self._tolerance = tolerance

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _best_val_score(history: list[RoundResult]) -> tuple[float, int]:
        """Return ``(best_score, best_round)`` across all rounds with validation scores."""
        best_score = -1.0
        best_round = -1
        for rr in history:
            if rr.val_scores is not None and rr.val_scores.mean_score > best_score:
                best_score = rr.val_scores.mean_score
                best_round = rr.round_num
        return best_score, best_round

    # -- protocol ------------------------------------------------------------

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Evaluate whether the latest round's validation score has regressed.

        Args:
            history: All round results accumulated so far.

        Returns:
            A :class:`GuardrailVerdict` with ``passed=False`` and a rollback
            recommendation when a significant regression is detected.
        """
        if not history:
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason="No history available; nothing to check.",
            )

        latest = history[-1]

        if latest.val_scores is None:
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
                    "regression check skipped."
                ),
            )

        current_score = latest.val_scores.mean_score
        best_score, best_round = self._best_val_score(history)

        if best_score < 0:
            # No valid validation scores at all (should not happen if latest has val).
            return GuardrailVerdict(
                passed=True,
                guardrail_name=self.name,
                reason="No historical validation scores to compare against.",
            )

        drop = best_score - current_score

        if drop > self._tolerance:
            reason = (
                f"Regression detected in round {latest.round_num}: "
                f"val_score={current_score:.4f} dropped {drop:.4f} below "
                f"best={best_score:.4f} (round {best_round}), "
                f"exceeding tolerance {self._tolerance:.4f}. "
                f"Recommend rollback to round {best_round} configuration."
            )
            logger.warning("%s: %s", self.name, reason)
            return GuardrailVerdict(
                passed=False,
                should_stop=False,
                guardrail_name=self.name,
                reason=reason,
            )

        logger.debug(
            "%s: round %d OK — val=%.4f, best=%.4f, drop=%.4f (tol=%.4f).",
            self.name,
            latest.round_num,
            current_score,
            best_score,
            drop,
            self._tolerance,
        )
        return GuardrailVerdict(
            passed=True,
            guardrail_name=self.name,
            reason=(
                f"Round {latest.round_num}: val_score={current_score:.4f}, "
                f"best={best_score:.4f}, drop={drop:.4f} within "
                f"tolerance {self._tolerance:.4f}."
            ),
        )
