"""Unit tests for nuwa.engine.scheduler — convergence detection and round budgeting."""

from __future__ import annotations

from nuwa.core.types import (
    LoopContext,
    Reflection,
    RoundResult,
    ScoreCard,
    ScoredResult,
    TrainingConfig,
)
from nuwa.engine.scheduler import TrainingScheduler


def _score_card(mean: float) -> ScoreCard:
    """Create a minimal ScoreCard with a single result yielding *mean*."""
    from nuwa.core.types import AgentResponse, EvalSample

    s = EvalSample(input_text="q", expected_behavior="a", difficulty="easy")
    r = AgentResponse(output_text="ans", latency_ms=10.0)
    return ScoreCard(results=[ScoredResult(sample=s, response=r, score=mean, reasoning="")])


def _round(round_num: int, val_mean: float) -> RoundResult:
    """Create a round result with a given val score."""
    return RoundResult(
        round_num=round_num,
        train_scores=_score_card(val_mean + 0.05),
        val_scores=_score_card(val_mean),
        reflection=Reflection(round_num=round_num, diagnosis="test", priority="low"),
        applied=True,
    )


class TestShouldStop:
    def test_empty_history_continues(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(config=cfg)
        stop, reason = scheduler.should_stop(ctx)
        assert stop is False

    def test_max_rounds_reached(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=3)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(config=cfg, round_num=3)
        stop, reason = scheduler.should_stop(ctx)
        assert stop is True
        assert "max_rounds" in reason

    def test_score_target_reached(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(config=cfg, best_val_score=0.95)
        stop, reason = scheduler.should_stop(ctx)
        assert stop is True
        assert "score target" in reason

    def test_convergence_detected(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(
            config=cfg,
            round_num=5,
            best_val_score=0.75,
            history=[
                _round(3, 0.745),
                _round(4, 0.746),
                _round(5, 0.744),
            ],
        )
        stop, reason = scheduler.should_stop(ctx)
        assert stop is True
        assert "converged" in reason

    def test_not_converged_improving(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(
            config=cfg,
            round_num=5,
            best_val_score=0.80,
            history=[
                _round(3, 0.70),
                _round(4, 0.75),
                _round(5, 0.80),
            ],
        )
        stop, reason = scheduler.should_stop(ctx)
        assert stop is False

    def test_below_max_rounds_continues(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10)
        scheduler = TrainingScheduler(cfg)
        ctx = LoopContext(config=cfg, round_num=2)
        stop, reason = scheduler.should_stop(ctx)
        assert stop is False


class TestGetRoundBudget:
    def test_first_round_base(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=5, samples_per_round=20)
        scheduler = TrainingScheduler(cfg)
        budget = scheduler.get_round_budget(1)
        assert budget["samples_per_round"] == 20
        assert budget["multiplier"] == 1.0

    def test_later_round_increases(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=10, samples_per_round=20)
        scheduler = TrainingScheduler(cfg)
        budget = scheduler.get_round_budget(3)
        assert budget["samples_per_round"] >= 20
        assert budget["multiplier"] > 1.0

    def test_capped_at_2x(self):
        cfg = TrainingConfig(training_direction="t", max_rounds=50, samples_per_round=20)
        scheduler = TrainingScheduler(cfg)
        budget = scheduler.get_round_budget(100)
        assert budget["samples_per_round"] <= 40  # 2x cap
        assert budget["multiplier"] <= 2.0
