"""Multi-objective data types for the Nuwa AI Trainer.

Defines objectives, objective sets, per-sample multi-objective scores, and
aggregate score cards that summarise performance across multiple axes
(e.g. accuracy, helpfulness, safety) in a single training round.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Objective(BaseModel):
    """A single training objective."""

    name: str = Field(
        ..., description="Objective identifier, e.g. 'accuracy', 'latency', 'safety'."
    )
    weight: float = Field(
        default=1.0, ge=0.0, description="Relative importance for weighted aggregation."
    )
    direction: str = Field(default="maximize", description="'maximize' or 'minimize'.")
    target: float | None = Field(
        default=None, description="Optional target value (e.g. accuracy >= 0.9)."
    )
    description: str = Field(default="", description="Human-readable description of the objective.")


class ObjectiveSet(BaseModel):
    """A collection of objectives for multi-objective optimisation."""

    objectives: list[Objective] = Field(default_factory=list)

    def get(self, name: str) -> Objective | None:
        """Return the objective with the given *name*, or ``None``."""
        for obj in self.objectives:
            if obj.name == name:
                return obj
        return None

    def names(self) -> list[str]:
        """Return an ordered list of objective names."""
        return [obj.name for obj in self.objectives]

    def weights_dict(self) -> dict[str, float]:
        """Return a mapping of objective name to weight."""
        return {obj.name: obj.weight for obj in self.objectives}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> ObjectiveSet:
        """Default objectives: accuracy + helpfulness + safety."""
        return cls(
            objectives=[
                Objective(
                    name="accuracy", weight=1.0, direction="maximize", description="回答准确性"
                ),
                Objective(
                    name="helpfulness", weight=0.8, direction="maximize", description="回答有用性"
                ),
                Objective(
                    name="safety", weight=1.0, direction="maximize", description="回答安全性"
                ),
            ]
        )

    @classmethod
    def from_list(cls, names: list[str]) -> ObjectiveSet:
        """Quick creation from a list of objective names.

        All objectives default to ``weight=1.0`` and ``direction='maximize'``.
        """
        return cls(objectives=[Objective(name=n) for n in names])


class MultiObjectiveScore(BaseModel):
    """Scores for a single sample across multiple objectives."""

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of objective_name -> score (0-1).",
    )
    weighted_aggregate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted combination of per-objective scores.",
    )

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the score for *name*, falling back to *default*."""
        return self.scores.get(name, default)


class MultiObjectiveScoreCard(BaseModel):
    """Aggregate multi-objective scores for a round."""

    sample_scores: list[MultiObjectiveScore] = Field(default_factory=list)
    objective_means: dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective average across all samples.",
    )
    objective_pass_rates: dict[str, float] = Field(
        default_factory=dict,
        description="Per-objective pass rate (fraction of samples scoring >= 0.7).",
    )
    weighted_mean: float = Field(
        default=0.0,
        description="Overall weighted aggregate score across all samples.",
    )

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def from_scored_results(
        cls,
        results: list[MultiObjectiveScore],
        objectives: ObjectiveSet,
    ) -> MultiObjectiveScoreCard:
        """Build a score card from a list of :class:`MultiObjectiveScore` items.

        Parameters
        ----------
        results:
            Per-sample multi-objective scores.
        objectives:
            The :class:`ObjectiveSet` defining names and weights.
        """
        if not results:
            return cls()

        pass_threshold = 0.7
        obj_names = objectives.names()
        weights = objectives.weights_dict()

        # Per-objective means and pass rates.
        objective_means: dict[str, float] = {}
        objective_pass_rates: dict[str, float] = {}
        for name in obj_names:
            values = [r.scores.get(name, 0.0) for r in results]
            objective_means[name] = sum(values) / len(values)
            objective_pass_rates[name] = sum(1 for v in values if v >= pass_threshold) / len(values)

        # Overall weighted mean.
        total_weight = sum(weights.get(n, 1.0) for n in obj_names) or 1.0
        weighted_mean = (
            sum(objective_means.get(n, 0.0) * weights.get(n, 1.0) for n in obj_names) / total_weight
        )

        return cls(
            sample_scores=results,
            objective_means=objective_means,
            objective_pass_rates=objective_pass_rates,
            weighted_mean=weighted_mean,
        )
