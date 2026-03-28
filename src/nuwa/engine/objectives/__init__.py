"""Multi-objective optimisation support for the Nuwa AI Trainer.

Public API
----------
.. autoclass:: Objective
.. autoclass:: ObjectiveSet
.. autoclass:: MultiObjectiveScore
.. autoclass:: MultiObjectiveScoreCard
.. autoclass:: ParetoPoint
.. autoclass:: ParetoTracker
.. autoclass:: MultiObjectiveScorer
"""

from __future__ import annotations

from nuwa.engine.objectives.pareto import ParetoPoint, ParetoTracker
from nuwa.engine.objectives.scorer import MultiObjectiveScorer
from nuwa.engine.objectives.types import (
    MultiObjectiveScore,
    MultiObjectiveScoreCard,
    Objective,
    ObjectiveSet,
)

__all__ = [
    "Objective",
    "ObjectiveSet",
    "MultiObjectiveScore",
    "MultiObjectiveScoreCard",
    "ParetoPoint",
    "ParetoTracker",
    "MultiObjectiveScorer",
]
