"""Unit tests for TrainingLoop resume behavior."""

from __future__ import annotations

from typing import Any

import pytest

from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    Reflection,
    RoundResult,
    ScoreCard,
    ScoredResult,
    TrainingConfig,
)
from nuwa.engine.loop import TrainingLoop


class _DummyTarget:
    def __init__(self) -> None:
        self._config: dict[str, Any] = {"tone": "base"}
        self.applied: list[dict[str, Any]] = []

    async def invoke(self, input_text: str, config: dict[str, Any] | None = None):
        _ = input_text
        _ = config
        raise RuntimeError("invoke should not be called in resume no-op test")

    def get_current_config(self) -> dict[str, Any]:
        return dict(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self.applied.append(dict(config))


class _DummyBackend:
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        _ = messages
        _ = kwargs
        raise RuntimeError("complete should not be called in resume no-op test")

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type,
        **kwargs: Any,
    ) -> Any:
        _ = messages
        _ = response_schema
        _ = kwargs
        raise RuntimeError("complete_structured should not be called in resume no-op test")


@pytest.mark.asyncio
async def test_resume_no_op_uses_existing_history_and_best_config() -> None:
    target = _DummyTarget()
    backend = _DummyBackend()

    round_1 = RoundResult(
        round_num=1,
        train_scores=ScoreCard(
            results=[
                ScoredResult(
                    sample=EvalSample(
                        input_text="hello",
                        expected_behavior="say hello",
                        difficulty="easy",
                    ),
                    response=AgentResponse(output_text="hello", latency_ms=1.0),
                    score=0.8,
                    reasoning="good",
                )
            ],
            failure_analysis="ok",
        ),
        val_scores=ScoreCard(
            results=[
                ScoredResult(
                    sample=EvalSample(
                        input_text="world",
                        expected_behavior="say world",
                        difficulty="easy",
                    ),
                    response=AgentResponse(output_text="world", latency_ms=1.0),
                    score=0.75,
                    reasoning="good",
                )
            ],
            failure_analysis="ok",
        ),
        reflection=Reflection(
            round_num=1,
            diagnosis="baseline",
            failure_patterns=[],
            proposed_changes=[],
            priority="low",
        ),
        mutation=None,
        applied=False,
    )

    loop = TrainingLoop(
        config=TrainingConfig(training_direction="resume test", max_rounds=2, samples_per_round=2),
        backend=backend,
        target=target,
        guardrails=[],
        start_round=3,
        initial_history=[round_1],
        initial_best_config={"tone": "resumed"},
        initial_best_val_score=0.75,
    )

    result = await loop.run()

    assert "resume no-op" in result.stop_reason
    assert result.best_round == 1
    assert result.best_val_score == pytest.approx(0.75)
    assert result.final_config == {"tone": "resumed"}
    assert target.applied[-1] == {"tone": "resumed"}
