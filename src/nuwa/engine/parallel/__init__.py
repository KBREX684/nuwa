"""Parallel / distributed evaluation infrastructure for Nuwa AI Trainer.

Provides high-throughput parallel execution and multi-judge ensemble
evaluation as drop-in replacements for the default sequential stages.
"""

from __future__ import annotations

from nuwa.engine.parallel.evaluator import (
    EnsembleStrategy,
    JudgeConfig,
    ParallelEvaluator,
)
from nuwa.engine.parallel.executor import ParallelExecutor
from nuwa.engine.parallel.stage import (
    ParallelEvaluationStage,
    ParallelExecutionStage,
    ParallelValidationStage,
)

__all__ = [
    "EnsembleStrategy",
    "JudgeConfig",
    "ParallelEvaluator",
    "ParallelExecutor",
    "ParallelEvaluationStage",
    "ParallelExecutionStage",
    "ParallelValidationStage",
]
