"""Distributed training coordinator (MVP)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from nuwa.core.types import TrainingResult


class DistributedTrainingCoordinator:
    """Run multiple training workers concurrently and pick the best result."""

    async def run_workers(
        self,
        worker_runs: Sequence[Callable[[], Awaitable[TrainingResult]]],
    ) -> tuple[TrainingResult, list[TrainingResult], int]:
        if not worker_runs:
            raise ValueError("worker_runs must not be empty")

        results = list(await asyncio.gather(*(runner() for runner in worker_runs)))
        best_index = max(range(len(results)), key=lambda idx: results[idx].best_val_score)
        winner = results[best_index]
        winner = winner.model_copy(
            update={
                "stop_reason": (
                    f"distributed workers={len(results)} winner={best_index + 1}; "
                    f"{winner.stop_reason}"
                )
            }
        )
        return winner, results, best_index
