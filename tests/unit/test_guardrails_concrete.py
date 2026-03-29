"""Unit tests for concrete guardrails: Overfitting, Regression, Consistency."""

from __future__ import annotations

import pytest

from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    RoundResult,
    ScoreCard,
    ScoredResult,
)
from nuwa.guardrails.consistency import ConsistencyGuardrail
from nuwa.guardrails.overfitting import OverfittingGuardrail
from nuwa.guardrails.regression import RegressionGuardrail


def _sample() -> EvalSample:
    return EvalSample(input_text="q", expected_behavior="a", difficulty="easy")


def _response() -> AgentResponse:
    return AgentResponse(output_text="ans", latency_ms=10.0)


def _card(scores: list[float]) -> ScoreCard:
    return ScoreCard(
        results=[
            ScoredResult(sample=_sample(), response=_response(), score=s, reasoning="")
            for s in scores
        ]
    )


def _round(
    round_num: int,
    train_scores: list[float],
    val_scores: list[float] | None = None,
) -> RoundResult:
    return RoundResult(
        round_num=round_num,
        train_scores=_card(train_scores),
        val_scores=_card(val_scores) if val_scores else None,
        reflection=__import__("nuwa.core.types", fromlist=["Reflection"]).Reflection(
            round_num=round_num, diagnosis="test", priority="low"
        ),
        applied=True,
    )


# ===========================================================================
# OverfittingGuardrail
# ===========================================================================


class TestOverfittingGuardrail:
    def test_no_history_passes(self):
        g = OverfittingGuardrail(threshold=0.15)
        v = g.check([])
        assert v.passed is True

    def test_no_val_scores_passes(self):
        g = OverfittingGuardrail(threshold=0.15)
        rr = _round(1, train_scores=[0.9], val_scores=None)
        v = g.check([rr])
        assert v.passed is True

    def test_small_gap_passes(self):
        g = OverfittingGuardrail(threshold=0.15)
        rr = _round(1, train_scores=[0.80], val_scores=[0.75])
        v = g.check([rr])
        assert v.passed is True

    def test_large_gap_fails(self):
        g = OverfittingGuardrail(threshold=0.15)
        rr = _round(1, train_scores=[0.90], val_scores=[0.50])
        v = g.check([rr])
        assert v.passed is False
        assert "overfitting" in v.reason.lower()

    def test_threshold_boundary(self):
        g = OverfittingGuardrail(threshold=0.20)
        rr = _round(1, train_scores=[0.80], val_scores=[0.60])  # gap = 0.20
        v = g.check([rr])
        assert v.passed is False

    def test_trend_detection(self):
        """Gap widening for 2 consecutive rounds triggers trend detection."""
        g = OverfittingGuardrail(threshold=0.30, trend_rounds=2)
        r1 = _round(1, train_scores=[0.70], val_scores=[0.65])  # gap=0.05
        r2 = _round(2, train_scores=[0.80], val_scores=[0.72])  # gap=0.08
        r3 = _round(3, train_scores=[0.90], val_scores=[0.78])  # gap=0.12
        v = g.check([r1, r2, r3])
        assert v.passed is False
        assert "trend" in v.reason.lower()

    def test_custom_threshold(self):
        g = OverfittingGuardrail(threshold=0.05)
        rr = _round(1, train_scores=[0.80], val_scores=[0.70])  # gap=0.10
        v = g.check([rr])
        assert v.passed is False

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError):
            OverfittingGuardrail(threshold=-0.1)

    def test_zero_threshold_passes_tight(self):
        g = OverfittingGuardrail(threshold=0.0)
        rr = _round(1, train_scores=[0.80], val_scores=[0.79])  # gap=0.01
        v = g.check([rr])
        assert v.passed is False


# ===========================================================================
# RegressionGuardrail
# ===========================================================================


class TestRegressionGuardrail:
    def test_no_history_passes(self):
        g = RegressionGuardrail(tolerance=0.05)
        v = g.check([])
        assert v.passed is True

    def test_no_val_scores_passes(self):
        g = RegressionGuardrail(tolerance=0.05)
        rr = _round(1, train_scores=[0.8], val_scores=None)
        v = g.check([rr])
        assert v.passed is True

    def test_improving_passes(self):
        g = RegressionGuardrail(tolerance=0.05)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.70])
        r2 = _round(2, train_scores=[0.8], val_scores=[0.75])
        v = g.check([r1, r2])
        assert v.passed is True

    def test_regression_fails(self):
        g = RegressionGuardrail(tolerance=0.05)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.80])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.60])  # drop=0.20 > 0.05
        v = g.check([r1, r2])
        assert v.passed is False
        assert "regression" in v.reason.lower()

    def test_within_tolerance_passes(self):
        g = RegressionGuardrail(tolerance=0.10)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.80])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.73])  # drop=0.07 < 0.10
        v = g.check([r1, r2])
        assert v.passed is True

    def test_exact_tolerance_boundary(self):
        g = RegressionGuardrail(tolerance=0.05)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.80])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.75])  # drop=0.05
        v = g.check([r1, r2])
        # Implementation uses strict >, so exactly at tolerance still fails
        assert v.passed is False

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError):
            RegressionGuardrail(tolerance=-0.1)

    def test_recommend_rollback_round(self):
        g = RegressionGuardrail(tolerance=0.05)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.90])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.70])
        v = g.check([r1, r2])
        assert "round 1" in v.reason  # best round referenced


# ===========================================================================
# ConsistencyGuardrail
# ===========================================================================


class TestConsistencyGuardrail:
    def test_no_history_passes(self):
        g = ConsistencyGuardrail(threshold=0.8)
        v = g.check([])
        assert v.passed is True

    def test_stable_scores_pass(self):
        g = ConsistencyGuardrail(threshold=0.8, min_rounds=2)
        r1 = _round(1, train_scores=[0.75], val_scores=[0.75])
        r2 = _round(2, train_scores=[0.76], val_scores=[0.76])
        v = g.check([r1, r2])
        assert v.passed is True

    def test_oscillating_scores_fail(self):
        """Scores bouncing 0.3 -> 0.9 -> 0.3 should flag inter-round instability."""
        g = ConsistencyGuardrail(threshold=0.8, min_rounds=3)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.30])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.90])
        r3 = _round(3, train_scores=[0.7], val_scores=[0.30])
        v = g.check([r1, r2, r3])
        assert v.passed is False

    def test_intra_round_high_variance(self):
        """A single round with very spread scores should fail intra-round check."""
        g = ConsistencyGuardrail(threshold=0.8)
        # CV of [0.1, 0.9] is very high
        rr = _round(1, train_scores=[0.5], val_scores=[0.1, 0.9])
        v = g.check([rr])
        assert v.passed is False

    def test_insufficient_rounds_passes(self):
        g = ConsistencyGuardrail(threshold=0.8, min_rounds=3)
        r1 = _round(1, train_scores=[0.7], val_scores=[0.30])
        r2 = _round(2, train_scores=[0.7], val_scores=[0.90])
        v = g.check([r1, r2])  # only 2 rounds, min_rounds=3
        assert v.passed is True

    def test_threshold_bounds(self):
        with pytest.raises(ValueError):
            ConsistencyGuardrail(threshold=1.5)
        with pytest.raises(ValueError):
            ConsistencyGuardrail(threshold=-0.1)

    def test_min_rounds_minimum(self):
        with pytest.raises(ValueError):
            ConsistencyGuardrail(threshold=0.8, min_rounds=0)
