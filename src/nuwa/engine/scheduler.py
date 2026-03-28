"""Training scheduler -- convergence detection and round budgeting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nuwa.core.types import LoopContext, TrainingConfig

if TYPE_CHECKING:
    from nuwa.engine.objectives.pareto import ParetoTracker

logger = logging.getLogger(__name__)

_CONVERGENCE_WINDOW = 3
_MIN_IMPROVEMENT = 0.01
_DEFAULT_SCORE_TARGET = 0.95
_PARETO_STALL_WINDOW = 3


class TrainingScheduler:
    """Decides when to stop training and how many samples each round gets."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------

    def should_stop(self, context: LoopContext) -> tuple[bool, str]:
        """Return ``(True, reason)`` if training should halt.

        Checks, in order:
        1. Max rounds reached.
        2. Score target reached.
        3. Convergence -- improvement below threshold for N consecutive rounds.
        """
        # 1. Max rounds
        if context.round_num >= self._config.max_rounds:
            return True, f"max_rounds ({self._config.max_rounds}) reached"

        # 2. Score target
        if context.best_val_score >= _DEFAULT_SCORE_TARGET:
            return True, (
                f"score target reached (best_val={context.best_val_score:.3f} "
                f">= {_DEFAULT_SCORE_TARGET})"
            )

        # 3. Convergence detection: look at the last N val scores.
        if len(context.history) >= _CONVERGENCE_WINDOW:
            recent = context.history[-_CONVERGENCE_WINDOW:]
            val_scores = [
                rr.val_scores.mean_score
                for rr in recent
                if rr.val_scores is not None
            ]
            if len(val_scores) >= _CONVERGENCE_WINDOW:
                max_delta = max(val_scores) - min(val_scores)
                if max_delta < _MIN_IMPROVEMENT:
                    return True, (
                        f"converged: val score improvement < {_MIN_IMPROVEMENT} "
                        f"over last {_CONVERGENCE_WINDOW} rounds "
                        f"(range={max_delta:.4f})"
                    )

        return False, ""

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def get_round_budget(self, round_num: int) -> dict:
        """Return the sample budget for a given round.

        Later rounds receive slightly more samples to stress-test the agent
        on harder / more diverse inputs.
        """
        base = self._config.samples_per_round
        # Ramp: +10 % per round after the first, capped at 2x base.
        multiplier = 1.0 + 0.1 * max(0, round_num - 1)
        multiplier = min(multiplier, 2.0)
        effective = max(1, int(base * multiplier))

        return {
            "samples_per_round": effective,
            "round_num": round_num,
            "multiplier": round(multiplier, 2),
        }

    # ------------------------------------------------------------------
    # Multi-objective stopping
    # ------------------------------------------------------------------

    def should_stop_multi_objective(
        self,
        context: LoopContext,
        pareto_tracker: ParetoTracker,
    ) -> tuple[bool, str]:
        """Return ``(True, reason)`` if training should halt under multi-objective mode.

        Checks the same single-objective criteria first, then adds a
        Pareto-frontier stall check: if the frontier has not changed for
        ``_PARETO_STALL_WINDOW`` consecutive rounds, the optimisation is
        considered converged.

        Parameters
        ----------
        context:
            Current loop context.
        pareto_tracker:
            The :class:`ParetoTracker` tracking the Pareto frontier.

        Returns
        -------
        tuple[bool, str]
            ``(should_stop, reason)`` -- the reason string is non-empty only
            when ``should_stop`` is True.
        """
        # Delegate to single-objective checks first (max_rounds, score target,
        # single-objective convergence).
        stop, reason = self.should_stop(context)
        if stop:
            return stop, reason

        # Pareto frontier stall detection.
        frontier = pareto_tracker.frontier
        if not frontier:
            return False, ""

        history_len = len(context.history)
        if history_len < _PARETO_STALL_WINDOW:
            return False, ""

        # Determine the most recent round that contributed a frontier point.
        latest_frontier_round = max(p.round_num for p in frontier)
        rounds_since_change = context.round_num - latest_frontier_round

        if rounds_since_change >= _PARETO_STALL_WINDOW:
            return True, (
                f"pareto_converged: Pareto frontier unchanged for "
                f"{rounds_since_change} rounds (stall window={_PARETO_STALL_WINDOW}). "
                f"Frontier size={len(frontier)}."
            )

        return False, ""
