"""Unit tests for distributed training coordinator."""

from __future__ import annotations

import pytest

from nuwa.core.types import TrainingResult
from nuwa.engine.distributed.coordinator import DistributedTrainingCoordinator


def _result(score: float, reason: str) -> TrainingResult:
    return TrainingResult(
        rounds=[],
        best_round=1,
        best_val_score=score,
        final_config={"score": score},
        stop_reason=reason,
        total_duration_s=1.0,
    )


@pytest.mark.asyncio
async def test_run_workers_picks_highest_score() -> None:
    coordinator = DistributedTrainingCoordinator()

    async def _worker(score: float) -> TrainingResult:
        return _result(score, "done")

    winner, all_results, winner_idx = await coordinator.run_workers(
        [
            lambda: _worker(0.52),
            lambda: _worker(0.81),
            lambda: _worker(0.67),
        ]
    )

    assert winner_idx == 1
    assert len(all_results) == 3
    assert winner.best_val_score == pytest.approx(0.81)
    assert "distributed workers=3 winner=2" in winner.stop_reason


@pytest.mark.asyncio
async def test_run_workers_requires_non_empty_list() -> None:
    coordinator = DistributedTrainingCoordinator()
    with pytest.raises(ValueError, match="must not be empty"):
        await coordinator.run_workers([])
