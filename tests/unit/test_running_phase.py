"""Unit tests for running-phase callback contract."""

from __future__ import annotations

from types import SimpleNamespace

from nuwa.conversation.phases.running import RunningPhase
from nuwa.conversation.renderer import NuwaRenderer
from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    Reflection,
    RoundResult,
    ScoreCard,
    ScoredResult,
)


def _make_round_result(round_num: int = 1) -> RoundResult:
    sample = EvalSample(
        input_text="hello",
        expected_behavior="respond politely",
        difficulty="easy",
    )
    response = AgentResponse(output_text="hi", latency_ms=12.0)
    scored = ScoredResult(
        sample=sample,
        response=response,
        score=0.8,
        reasoning="Looks good.",
    )
    card = ScoreCard(results=[scored], failure_analysis="")
    reflection = Reflection(
        round_num=round_num,
        diagnosis="stable",
        failure_patterns=[],
        proposed_changes=[],
        priority="low",
    )
    return RoundResult(
        round_num=round_num,
        train_scores=card,
        val_scores=card,
        reflection=reflection,
        mutation=None,
        applied=False,
    )


def test_running_phase_callback_matches_training_loop_contract() -> None:
    renderer = NuwaRenderer()
    phase = RunningPhase(renderer)
    callback = phase.build_callback()

    context = SimpleNamespace(config=SimpleNamespace(max_rounds=3))
    round_result = _make_round_result(round_num=1)

    # TrainingLoop callback contract is cb(round_result, context).
    callback(round_result, context)

    assert phase._max_rounds == 3
    assert phase._scores == [round_result.train_scores.mean_score]
