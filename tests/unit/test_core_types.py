"""Unit tests for nuwa.core.types — Pydantic data models."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    GuardrailVerdict,
    LoopContext,
    Mutation,
    Reflection,
    RoundResult,
    ScoreCard,
    ScoredResult,
    TrainingConfig,
    TrainingResult,
)

# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig(training_direction="test")
        assert cfg.max_rounds == 10
        assert cfg.samples_per_round == 20
        assert cfg.train_val_split == 0.7
        assert cfg.overfitting_threshold == 0.15
        assert cfg.consistency_runs == 3
        assert cfg.consistency_threshold == 0.8
        assert cfg.regression_tolerance == 0.05

    def test_custom_values(self):
        cfg = TrainingConfig(
            training_direction="improve accuracy",
            max_rounds=5,
            samples_per_round=10,
            train_val_split=0.8,
        )
        assert cfg.training_direction == "improve accuracy"
        assert cfg.max_rounds == 5
        assert cfg.samples_per_round == 10
        assert cfg.train_val_split == 0.8

    def test_max_rounds_minimum(self):
        with pytest.raises(Exception):
            TrainingConfig(training_direction="t", max_rounds=0)

    def test_train_val_split_bounds(self):
        with pytest.raises(Exception):
            TrainingConfig(training_direction="t", train_val_split=0.0)
        with pytest.raises(Exception):
            TrainingConfig(training_direction="t", train_val_split=1.0)

    def test_consistency_threshold_bounds(self):
        with pytest.raises(Exception):
            TrainingConfig(training_direction="t", consistency_threshold=1.5)
        with pytest.raises(Exception):
            TrainingConfig(training_direction="t", consistency_threshold=-0.1)


# ---------------------------------------------------------------------------
# EvalSample
# ---------------------------------------------------------------------------


class TestEvalSample:
    def test_auto_id(self):
        s = EvalSample(
            input_text="hello",
            expected_behavior="respond politely",
            difficulty="medium",
        )
        # UUID parse should not raise
        uuid.UUID(s.id)

    def test_custom_id(self):
        s = EvalSample(
            id="custom-1",
            input_text="hi",
            expected_behavior="wave",
            difficulty="easy",
        )
        assert s.id == "custom-1"

    def test_difficulty_values(self):
        for d in ("easy", "medium", "hard"):
            s = EvalSample(input_text="x", expected_behavior="y", difficulty=d)
            assert s.difficulty == d

    def test_invalid_difficulty(self):
        with pytest.raises(Exception):
            EvalSample(input_text="x", expected_behavior="y", difficulty="extreme")

    def test_default_tags(self):
        s = EvalSample(input_text="x", expected_behavior="y", difficulty="easy")
        assert s.tags == []

    def test_custom_tags(self):
        s = EvalSample(
            input_text="x",
            expected_behavior="y",
            difficulty="easy",
            tags=["math", "reasoning"],
        )
        assert s.tags == ["math", "reasoning"]


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponse:
    def test_basic(self):
        r = AgentResponse(output_text="Hello!", latency_ms=42.5)
        assert r.output_text == "Hello!"
        assert r.latency_ms == 42.5
        assert r.raw_metadata == {}

    def test_negative_latency_rejected(self):
        with pytest.raises(Exception):
            AgentResponse(output_text="x", latency_ms=-1.0)

    def test_with_metadata(self):
        r = AgentResponse(
            output_text="ok",
            latency_ms=10.0,
            raw_metadata={"tokens": 50},
        )
        assert r.raw_metadata["tokens"] == 50


# ---------------------------------------------------------------------------
# ScoredResult & ScoreCard
# ---------------------------------------------------------------------------


def _sample() -> EvalSample:
    return EvalSample(input_text="q", expected_behavior="a", difficulty="easy")


def _response() -> AgentResponse:
    return AgentResponse(output_text="ans", latency_ms=10.0)


class TestScoredResult:
    def test_score_bounds(self):
        ScoredResult(sample=_sample(), response=_response(), score=0.0, reasoning="ok")
        ScoredResult(sample=_sample(), response=_response(), score=1.0, reasoning="ok")
        with pytest.raises(Exception):
            ScoredResult(sample=_sample(), response=_response(), score=1.5, reasoning="bad")
        with pytest.raises(Exception):
            ScoredResult(sample=_sample(), response=_response(), score=-0.1, reasoning="bad")


class TestScoreCard:
    def test_empty_card(self):
        card = ScoreCard()
        assert card.mean_score == 0.0
        assert card.pass_rate == 0.0

    def test_single_result(self):
        r = ScoredResult(sample=_sample(), response=_response(), score=0.8, reasoning="good")
        card = ScoreCard(results=[r])
        assert card.mean_score == pytest.approx(0.8)
        assert card.pass_rate == pytest.approx(1.0)

    def test_mixed_scores(self):
        results = [
            ScoredResult(sample=_sample(), response=_response(), score=0.9, reasoning=""),
            ScoredResult(sample=_sample(), response=_response(), score=0.5, reasoning=""),
            ScoredResult(sample=_sample(), response=_response(), score=0.7, reasoning=""),
        ]
        card = ScoreCard(results=results)
        assert card.mean_score == pytest.approx(0.7)
        assert card.pass_rate == pytest.approx(2 / 3)

    def test_all_fail(self):
        r = ScoredResult(sample=_sample(), response=_response(), score=0.3, reasoning="")
        card = ScoreCard(results=[r])
        assert card.pass_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class TestReflection:
    def test_basic(self):
        ref = Reflection(
            round_num=1,
            diagnosis="needs work",
            priority="high",
        )
        assert ref.round_num == 1
        assert ref.failure_patterns == []
        assert ref.proposed_changes == []

    def test_invalid_priority(self):
        with pytest.raises(Exception):
            Reflection(round_num=1, diagnosis="x", priority="urgent")


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestMutation:
    def test_basic(self):
        mut = Mutation(
            description="tweak prompt",
            original_config={"k": "v1"},
            proposed_config={"k": "v2"},
            reasoning="should improve",
        )
        assert mut.original_config == {"k": "v1"}
        assert mut.proposed_config == {"k": "v2"}

    def test_defaults(self):
        mut = Mutation(description="x", reasoning="y")
        assert mut.original_config == {}
        assert mut.proposed_config == {}


# ---------------------------------------------------------------------------
# RoundResult
# ---------------------------------------------------------------------------


class TestRoundResult:
    def _make_score_card(self, scores: list[float]) -> ScoreCard:
        results = [
            ScoredResult(sample=_sample(), response=_response(), score=s, reasoning="")
            for s in scores
        ]
        return ScoreCard(results=results)

    def test_basic(self):
        rr = RoundResult(
            round_num=1,
            train_scores=self._make_score_card([0.8]),
            val_scores=self._make_score_card([0.7]),
            reflection=Reflection(round_num=1, diagnosis="ok", priority="low"),
            applied=True,
        )
        assert rr.round_num == 1
        assert rr.applied is True
        assert rr.error is None
        assert isinstance(rr.timestamp, datetime)

    def test_with_error(self):
        rr = RoundResult(
            round_num=2,
            train_scores=self._make_score_card([0.5]),
            reflection=Reflection(round_num=2, diagnosis="skipped", priority="medium"),
            error="LLM timeout",
        )
        assert rr.error == "LLM timeout"
        assert rr.val_scores is None


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResult:
    def test_basic(self):
        result = TrainingResult(
            best_round=3,
            best_val_score=0.92,
            final_config={"prompt": "optimized"},
            stop_reason="converged",
            total_duration_s=120.5,
        )
        assert result.rounds == []
        assert result.best_round == 3
        assert result.best_val_score == pytest.approx(0.92)
        assert result.pareto_frontier is None
        assert result.sandbox_session_id is None


# ---------------------------------------------------------------------------
# GuardrailVerdict
# ---------------------------------------------------------------------------


class TestGuardrailVerdict:
    def test_pass(self):
        v = GuardrailVerdict(passed=True, guardrail_name="test")
        assert v.passed is True
        assert v.should_stop is False
        assert v.reason == ""

    def test_fail_with_stop(self):
        v = GuardrailVerdict(
            passed=False,
            should_stop=True,
            guardrail_name="test",
            reason="fatal",
        )
        assert v.passed is False
        assert v.should_stop is True


# ---------------------------------------------------------------------------
# LoopContext
# ---------------------------------------------------------------------------


class TestLoopContext:
    def test_default_context(self):
        cfg = TrainingConfig(training_direction="test")
        ctx = LoopContext(config=cfg)
        assert ctx.round_num == 0
        assert ctx.train_set == []
        assert ctx.val_set == []
        assert ctx.history == []
        assert ctx.best_val_score == 0.0

    def test_mutable_state(self):
        cfg = TrainingConfig(training_direction="test")
        ctx = LoopContext(config=cfg)
        ctx.round_num = 5
        ctx.train_set = [EvalSample(input_text="q", expected_behavior="a", difficulty="easy")]
        assert len(ctx.train_set) == 1
        assert ctx.round_num == 5
