"""Pareto frontier tracking for multi-objective optimisation.

Maintains the set of Pareto-optimal configurations discovered across
training rounds, enabling the trainer to explore trade-offs between
competing objectives (e.g. accuracy vs. latency vs. safety).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nuwa.engine.objectives.types import ObjectiveSet

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    round_num: int
    config: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0


class ParetoTracker:
    """Tracks the Pareto frontier across training rounds.

    A configuration is *Pareto-optimal* if no other configuration dominates it
    -- i.e. no other configuration is strictly better on **all** objectives
    simultaneously.

    This allows the trainer to maintain multiple good solutions representing
    different trade-offs between objectives.
    """

    def __init__(self, objectives: ObjectiveSet) -> None:
        self._objectives = objectives
        self._all_points: list[ParetoPoint] = []
        self._frontier: list[ParetoPoint] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        round_num: int,
        config: dict[str, Any],
        scores: dict[str, float],
    ) -> bool:
        """Add a new point.  Returns ``True`` if it lies on the Pareto frontier."""
        weights = self._objectives.weights_dict()
        total_weight = sum(weights.values()) or 1.0
        weighted = sum(
            scores.get(n, 0.0) * weights.get(n, 1.0) for n in self._objectives.names()
        ) / total_weight

        point = ParetoPoint(
            round_num=round_num,
            config=dict(config),
            scores=dict(scores),
            weighted_score=weighted,
        )
        self._all_points.append(point)

        # Check whether the new point is dominated by any existing frontier
        # member.  If so it is not Pareto-optimal.
        if any(self.dominates(fp.scores, point.scores) for fp in self._frontier):
            return False

        # Remove any frontier members now dominated by the new point.
        self._frontier = [
            fp for fp in self._frontier if not self.dominates(point.scores, fp.scores)
        ]
        self._frontier.append(point)

        logger.debug(
            "Pareto frontier updated (size=%d) after round %d",
            len(self._frontier),
            round_num,
        )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def frontier(self) -> list[ParetoPoint]:
        """Return all Pareto-optimal points."""
        return list(self._frontier)

    def best_by_objective(self, objective_name: str) -> ParetoPoint | None:
        """Return the frontier point with the best score for *objective_name*."""
        if not self._frontier:
            return None

        obj = self._objectives.get(objective_name)
        if obj is None:
            return None

        reverse = obj.direction == "maximize"
        return max(
            self._frontier,
            key=lambda p: p.scores.get(objective_name, 0.0) if reverse else -p.scores.get(objective_name, 0.0),
        )

    def best_weighted(self) -> ParetoPoint | None:
        """Return the frontier point with the highest weighted aggregate score."""
        if not self._frontier:
            return None
        return max(self._frontier, key=lambda p: p.weighted_score)

    # ------------------------------------------------------------------
    # Dominance
    # ------------------------------------------------------------------

    def dominates(self, a: dict[str, float], b: dict[str, float]) -> bool:
        """Check if scores *a* Pareto-dominates scores *b*.

        *a* dominates *b* iff *a* is at least as good on every objective and
        strictly better on at least one.
        """
        at_least_as_good = True
        strictly_better = False

        for obj in self._objectives.objectives:
            va = a.get(obj.name, 0.0)
            vb = b.get(obj.name, 0.0)

            if obj.direction == "maximize":
                if va < vb:
                    at_least_as_good = False
                    break
                if va > vb:
                    strictly_better = True
            else:  # minimize
                if va > vb:
                    at_least_as_good = False
                    break
                if va < vb:
                    strictly_better = True

        return at_least_as_good and strictly_better

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_improvement_suggestions(self) -> list[str]:
        """Analyse the frontier and suggest which objectives have room for improvement."""
        if not self._frontier:
            return ["No data yet -- run at least one round."]

        suggestions: list[str] = []
        for obj in self._objectives.objectives:
            name = obj.name
            values = [p.scores.get(name, 0.0) for p in self._frontier]
            best = max(values) if obj.direction == "maximize" else min(values)

            target = obj.target
            if target is not None:
                if obj.direction == "maximize" and best < target:
                    gap = target - best
                    suggestions.append(
                        f"'{name}' best={best:.3f} is below target={target:.3f} (gap={gap:.3f}). "
                        f"Focus on improving {name}."
                    )
                elif obj.direction == "minimize" and best > target:
                    gap = best - target
                    suggestions.append(
                        f"'{name}' best={best:.3f} is above target={target:.3f} (gap={gap:.3f}). "
                        f"Focus on reducing {name}."
                    )

            # Check spread: large spread means the frontier has significant
            # trade-offs on this objective.
            spread = max(values) - min(values)
            if spread > 0.2 and len(self._frontier) > 1:
                suggestions.append(
                    f"'{name}' has high variance on the frontier (spread={spread:.3f}). "
                    f"Consider whether the trade-off is acceptable."
                )

        if not suggestions:
            suggestions.append("All objectives appear well-covered on the current frontier.")

        return suggestions

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging / display."""
        if not self._frontier:
            return {"frontier_size": 0, "points": [], "suggestions": []}

        best = self.best_weighted()
        objective_bests: dict[str, dict[str, float | int]] = {}
        for obj in self._objectives.objectives:
            best_point = self.best_by_objective(obj.name)
            if best_point is None:
                objective_bests[obj.name] = {
                    "best_value": 0.0,
                    "best_round": -1,
                }
            else:
                objective_bests[obj.name] = {
                    "best_value": best_point.scores.get(obj.name, 0.0),
                    "best_round": best_point.round_num,
                }

        return {
            "frontier_size": len(self._frontier),
            "best_weighted_score": best.weighted_score if best else 0.0,
            "best_weighted_round": best.round_num if best else -1,
            "objective_bests": objective_bests,
            "suggestions": self.get_improvement_suggestions(),
        }
