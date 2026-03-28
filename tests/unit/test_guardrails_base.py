"""Unit tests for nuwa.guardrails.base (BaseGuardrail and CompositeGuardrail)."""

from __future__ import annotations

from nuwa.core.types import GuardrailVerdict, Reflection, RoundResult, ScoreCard
from nuwa.guardrails.base import BaseGuardrail, CompositeGuardrail


def _make_round(
    round_num: int, train_mean: float = 0.0, val_mean: float | None = None
) -> RoundResult:
    """Helper to build a minimal RoundResult for testing."""
    train = ScoreCard(results=[], failure_analysis="")
    val = ScoreCard(results=[], failure_analysis="") if val_mean is not None else None
    reflection = Reflection(
        round_num=round_num,
        diagnosis="test",
        failure_patterns=[],
        proposed_changes=[],
        priority="low",
    )
    return RoundResult(
        round_num=round_num,
        train_scores=train,
        val_scores=val,
        reflection=reflection,
    )


class TestBaseGuardrail:
    """Tests for the BaseGuardrail default behaviour."""

    def test_default_name_is_class_name(self) -> None:
        g = BaseGuardrail()
        assert g.name == "BaseGuardrail"

    def test_custom_name(self) -> None:
        g = BaseGuardrail(guardrail_name="my_check")
        assert g.name == "my_check"

    def test_default_check_passes(self) -> None:
        g = BaseGuardrail()
        verdict = g.check([])
        assert verdict.passed is True
        assert verdict.should_stop is False
        assert verdict.guardrail_name == "BaseGuardrail"

    def test_default_check_with_history(self) -> None:
        g = BaseGuardrail()
        history = [_make_round(1, 0.5, 0.4)]
        verdict = g.check(history)
        assert verdict.passed is True

    def test_repr(self) -> None:
        g = BaseGuardrail(guardrail_name="test_gr")
        assert "test_gr" in repr(g)


class TestCompositeGuardrail:
    """Tests for the CompositeGuardrail aggregator."""

    def test_no_children_passes(self) -> None:
        comp = CompositeGuardrail([])
        verdict = comp.check([])
        assert verdict.passed is True
        assert "No child" in verdict.reason

    def test_single_passing_child(self) -> None:
        child = BaseGuardrail()
        comp = CompositeGuardrail([child])
        verdict = comp.check([])
        assert verdict.passed is True

    def test_single_failing_child(self) -> None:
        class AlwaysFail(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=False,
                    guardrail_name=self.name,
                    reason="Always fails",
                )

        comp = CompositeGuardrail([AlwaysFail()])
        verdict = comp.check([])
        assert verdict.passed is False
        assert "Always fails" in verdict.reason

    def test_first_failure_returned_not_second(self) -> None:
        class FailFirst(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=False,
                    guardrail_name="FailFirst",
                    reason="First failure",
                )

        class AlwaysPass(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=True,
                    guardrail_name="AlwaysPass",
                    reason="OK",
                )

        comp = CompositeGuardrail([FailFirst(), AlwaysPass()])
        verdict = comp.check([])
        assert verdict.passed is False
        assert "First failure" in verdict.reason

    def test_second_fails_not_returned_if_first_passes(self) -> None:
        """When first passes but second fails, the second's verdict is returned."""

        class AlwaysPass(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=True,
                    guardrail_name="AlwaysPass",
                    reason="OK",
                )

        class FailSecond(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=False,
                    guardrail_name="FailSecond",
                    reason="Second failure",
                )

        comp = CompositeGuardrail([AlwaysPass(), FailSecond()])
        verdict = comp.check([])
        assert verdict.passed is False
        assert "Second failure" in verdict.reason

    def test_all_pass_returns_composite_verdict(self) -> None:
        children = [BaseGuardrail(guardrail_name=f"g{i}") for i in range(5)]
        comp = CompositeGuardrail(children)
        verdict = comp.check([])
        assert verdict.passed is True
        assert verdict.guardrail_name == "CompositeGuardrail"
        assert "5 guardrails passed" in verdict.reason

    def test_should_stop_propagated(self) -> None:
        """If a passing child sets should_stop=True, it propagates."""

        class StopButPass(BaseGuardrail):
            def check(self, history: list[RoundResult]) -> GuardrailVerdict:
                return GuardrailVerdict(
                    passed=True,
                    should_stop=True,
                    guardrail_name="StopButPass",
                    reason="Should stop but pass",
                )

        comp = CompositeGuardrail([StopButPass()])
        verdict = comp.check([])
        assert verdict.passed is True
        assert verdict.should_stop is True

    def test_guardrails_property(self) -> None:
        c1 = BaseGuardrail(guardrail_name="a")
        c2 = BaseGuardrail(guardrail_name="b")
        comp = CompositeGuardrail([c1, c2])
        assert len(comp.guardrails) == 2
        assert comp.guardrails[0].name == "a"
        assert comp.guardrails[1].name == "b"

    def test_custom_name(self) -> None:
        comp = CompositeGuardrail([], guardrail_name="MyCustom")
        assert comp.name == "MyCustom"
