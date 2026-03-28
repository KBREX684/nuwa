"""Comprehensive end-to-end integration test for the Nuwa training loop.

Runs a full 3-round training loop with mock LLM backend and mock target agent,
exercises all guardrails, and validates persistence via RunLog.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    RoundResult,
    ScoreCard,
    ScoredResult,
    TrainingConfig,
    TrainingResult,
)
from nuwa.engine.loop import TrainingLoop
from nuwa.guardrails.consistency import ConsistencyGuardrail
from nuwa.guardrails.overfitting import OverfittingGuardrail
from nuwa.guardrails.regression import RegressionGuardrail
from nuwa.persistence.run_log import RunLog

# ---------------------------------------------------------------------------
# Mock model backend
# ---------------------------------------------------------------------------


class MockModelBackend:
    """Mock LLM backend that returns canned JSON responses based on prompt content."""

    def __init__(self) -> None:
        self._call_count = 0
        self._round_hint = 1  # tracks current round for score simulation

    async def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        self._call_count += 1
        content = " ".join(m.get("content", "") for m in messages)

        # Detect prompt type from content keywords
        if "dataset generation" in content.lower() or "evaluation samples" in content.lower():
            return self._dataset_response(content)
        elif "evaluation judge" in content.lower() or "objectively score" in content.lower():
            return self._scoring_response(content)
        elif "diagnostics analyst" in content.lower() or "root-cause analysis" in content.lower():
            return self._reflection_response(content)
        elif "prompt-engineering specialist" in content.lower() or "mutation" in content.lower():
            return self._mutation_response(content)
        else:
            # Fallback scoring response (used by validation stage too)
            return self._scoring_response(content)

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type,
        **kwargs: Any,
    ) -> Any:
        raw = await self.complete(messages, **kwargs)
        data = json.loads(raw)
        return response_schema.model_validate(data)

    def _dataset_response(self, content: str) -> str:
        """Return 10 EvalSample-shaped objects."""
        difficulties = ["easy", "medium", "hard"]
        samples = []
        for i in range(10):
            diff = difficulties[i % 3]
            samples.append(
                {
                    "input_text": f"Test question {i + 1}: Explain concept {i + 1} clearly.",
                    "expected_behavior": f"A clear, structured explanation of concept {i + 1} with examples.",
                    "difficulty": diff,
                    "tags": ["test", f"concept-{i + 1}", diff],
                }
            )
        return json.dumps(samples)

    def _scoring_response(self, content: str) -> str:
        """Return a score that improves with successive rounds.

        Detects round from internal state. Scores deliberately span
        below and above the 0.7 failure threshold so that reflection
        and mutation stages have failures to work with.

        After the immutable scorer blend (60% LLM + 40% immutable ~1.0),
        effective scores will be roughly: LLM*0.6 + 0.4.
        To get some failures (<0.7 after blend), LLM score must be < 0.5.
        So we use a wider range of base scores (0.1 - 0.8).
        """
        import random

        random.seed(self._call_count)

        # Base score improves per round but keeps some low values
        base = 0.1 + 0.1 * min(self._round_hint, 3)
        score = base + random.random() * 0.5
        score = max(0.0, min(1.0, score))

        return json.dumps(
            {
                "score": round(score, 3),
                "reasoning_en": f"The output demonstrates understanding (score={score:.3f}).",
                "reasoning_zh": f"输出展示了理解 (得分={score:.3f}).",
                "axis_scores": {
                    "correctness": round(min(1.0, score + 0.05), 3),
                    "completeness": round(max(0.0, score - 0.05), 3),
                    "format_compliance": 1.0,
                    "tone_style": round(score, 3),
                },
            }
        )

    def _reflection_response(self, content: str) -> str:
        """Return a structured reflection with diagnosis."""
        return json.dumps(
            {
                "diagnosis_summary_en": "Some outputs lack sufficient detail and miss edge cases.",
                "diagnosis_summary_zh": "部分输出缺乏足够细节，遗漏了边缘情况。",
                "failure_patterns": [
                    {
                        "label_en": "Insufficient detail",
                        "label_zh": "细节不足",
                        "affected_samples": [1, 3, 5],
                        "root_cause": "System prompt does not emphasize thoroughness.",
                        "severity": "high",
                    },
                    {
                        "label_en": "Missing examples",
                        "label_zh": "缺少示例",
                        "affected_samples": [2, 4],
                        "root_cause": "No instruction to include concrete examples.",
                        "severity": "medium",
                    },
                ],
                "proposed_changes": [
                    {
                        "target": "system_prompt",
                        "description_en": "Add instruction to provide detailed explanations with examples.",
                        "description_zh": "添加提供详细解释和示例的指令。",
                        "priority": "high",
                    },
                    {
                        "target": "config",
                        "description_en": "Increase temperature slightly for more creative responses.",
                        "description_zh": "稍微提高温度以获得更有创意的回复。",
                        "priority": "medium",
                    },
                ],
            }
        )

    def _mutation_response(self, content: str) -> str:
        """Return a mutation proposal with config changes."""
        return json.dumps(
            {
                "mutations": [
                    {
                        "id": "mut-001",
                        "type": "config_change",
                        "description_en": "Increase detail level in responses",
                        "description_zh": "增加响应中的细节级别",
                        "rationale_en": "Addresses the insufficient detail failure pattern.",
                        "rationale_zh": "解决细节不足的失败模式。",
                        "config_path": "detail_level",
                        "config_value": "high",
                        "expected_impact": "high",
                    },
                    {
                        "id": "mut-002",
                        "type": "prompt_insert",
                        "description_en": "Add examples instruction to system prompt",
                        "description_zh": "向系统提示添加示例指令",
                        "rationale_en": "Encourages the agent to include concrete examples.",
                        "rationale_zh": "鼓励代理包含具体示例。",
                        "new_text": "Always include at least one concrete example in your response.",
                        "expected_impact": "medium",
                    },
                ],
            }
        )


# ---------------------------------------------------------------------------
# Mock target agent
# ---------------------------------------------------------------------------


class MockTargetAgent:
    """Mock agent that subtly improves output quality after config changes."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.7,
            "detail_level": "medium",
        }
        self._config_change_count = 0
        self._invocation_count = 0

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        self._invocation_count += 1
        effective = {**self._config, **(config or {})}

        # Simulate quality improving with config changes
        quality_boost = min(self._config_change_count * 0.1, 0.3)
        detail = effective.get("detail_level", "medium")
        if detail == "high":
            quality_boost += 0.05

        base_output = f"Here is the answer to: {input_text[:60]}"
        if quality_boost > 0.1:
            base_output += (
                " This is a detailed response with concrete examples and thorough explanation."
            )
        else:
            base_output += " The answer covers the main points."

        return AgentResponse(
            output_text=base_output,
            latency_ms=50.0 + (self._invocation_count % 10) * 5.0,
            raw_metadata={"config_changes": self._config_change_count},
        )

    def get_current_config(self) -> dict[str, Any]:
        return dict(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._config_change_count += 1


# ---------------------------------------------------------------------------
# Callback for progress tracking
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Collects round-by-round progress for test assertions and summary."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, round_result: RoundResult, context: Any) -> None:
        train_mean = round_result.train_scores.mean_score
        val_mean = round_result.val_scores.mean_score if round_result.val_scores else None
        mutation_desc = round_result.mutation.description if round_result.mutation else "none"
        event = {
            "round": round_result.round_num,
            "train_score": train_mean,
            "val_score": val_mean,
            "mutation": mutation_desc,
            "applied": round_result.applied,
        }
        self.events.append(event)
        val_str = f"{val_mean:.3f}" if val_mean is not None else "N/A"
        print(
            f"  [Round {event['round']}] "
            f"train={train_mean:.3f}  "
            f"val={val_str}  "
            f"mutation={'applied' if event['applied'] else 'skipped'}  "
            f"desc={mutation_desc[:80]}"
        )


# ---------------------------------------------------------------------------
# Test: full training loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_training_loop() -> None:
    """Run a complete 3-round training loop with mocks and verify results."""
    print("\n===== FULL TRAINING LOOP INTEGRATION TEST =====\n")
    start = time.monotonic()

    backend = MockModelBackend()
    target = MockTargetAgent()
    tracker = ProgressTracker()

    config = TrainingConfig(
        training_direction="Improve the agent's ability to provide clear, detailed explanations with examples.",
        max_rounds=3,
        samples_per_round=10,
        train_val_split=0.7,
        overfitting_threshold=0.30,
        consistency_threshold=0.5,
        consistency_runs=3,
        regression_tolerance=0.10,
        objectives=[
            {"name": "correctness", "weight": 1.0, "direction": "maximize"},
            {"name": "completeness", "weight": 1.0, "direction": "maximize"},
        ],
    )

    guardrails = [
        OverfittingGuardrail(threshold=0.30),
        RegressionGuardrail(tolerance=0.10),
        ConsistencyGuardrail(threshold=0.5, min_rounds=2),
    ]

    loop = TrainingLoop(
        config=config,
        backend=backend,
        target=target,
        guardrails=guardrails,
        callbacks=[tracker],
    )

    # Advance the mock's round hint as rounds execute
    # We do this by patching the mock's _round_hint in the callback
    original_call = tracker.__call__

    def patched_call(rr: RoundResult, ctx: Any) -> None:
        original_call(rr, ctx)
        backend._round_hint = rr.round_num + 1

    tracker.__call__ = patched_call  # type: ignore[assignment]

    # Execute the loop
    result: TrainingResult = await loop.run()

    elapsed = time.monotonic() - start

    # ---- Assertions on TrainingResult ----
    assert isinstance(result, TrainingResult)
    assert len(result.rounds) > 0, "Expected at least one round"
    assert result.best_val_score > 0.0, (
        f"Expected positive best_val_score, got {result.best_val_score}"
    )
    assert result.stop_reason, "Expected a non-empty stop_reason"
    assert result.best_round >= 0
    assert result.total_duration_s > 0.0
    assert result.pareto_frontier is not None
    assert len(result.pareto_frontier) > 0

    # Check that history has round results
    for i, rr in enumerate(result.rounds):
        assert rr.train_scores is not None, f"Round {rr.round_num}: missing train_scores"
        assert rr.train_scores.mean_score > 0.0, f"Round {rr.round_num}: train score is 0"
        assert rr.reflection is not None, f"Round {rr.round_num}: missing reflection"
        assert rr.pareto_frontier_size >= 1

    # Check that the loop completed successfully - config changes may or may not happen
    # depending on whether mutations were proposed and validated
    print(f"\n  Config changes applied: {target._config_change_count}")

    # ---- Print summary ----
    print("\n===== TEST RESULTS SUMMARY =====")
    print(f"  Total rounds executed: {len(result.rounds)}")
    print(f"  Best round: {result.best_round}")
    print(f"  Best val score: {result.best_val_score:.4f}")
    print(f"  Stop reason: {result.stop_reason}")
    print(f"  Final config keys: {list(result.final_config.keys())}")
    print(f"  Total duration: {result.total_duration_s:.2f}s")
    print(f"  Backend calls: {backend._call_count}")
    print(f"  Agent invocations: {target._invocation_count}")
    print(f"  Config changes: {target._config_change_count}")

    print("\n  Round-by-round breakdown:")
    for rr in result.rounds:
        train_mean = rr.train_scores.mean_score
        val_mean = rr.val_scores.mean_score if rr.val_scores else 0.0
        mut_desc = rr.mutation.description[:60] if rr.mutation else "none"
        print(
            f"    Round {rr.round_num}: "
            f"train={train_mean:.4f}  val={val_mean:.4f}  "
            f"applied={rr.applied}  mutation={mut_desc}"
        )

    if tracker.events:
        print("\n  Mutations applied:")
        for ev in tracker.events:
            if ev["applied"]:
                print(f"    Round {ev['round']}: {ev['mutation'][:80]}")

    print(f"\n  Test elapsed time: {elapsed:.2f}s")
    print("===== END SUMMARY =====\n")


# ---------------------------------------------------------------------------
# Test: persistence with RunLog
# ---------------------------------------------------------------------------


def _make_dummy_round(round_num: int, train_score: float, val_score: float) -> RoundResult:
    """Create a minimal RoundResult for persistence testing."""
    from nuwa.core.types import Mutation, Reflection

    sample = EvalSample(
        input_text="test input",
        expected_behavior="test expected",
        difficulty="easy",
    )
    response = AgentResponse(output_text="test output", latency_ms=100.0)
    scored = ScoredResult(sample=sample, response=response, score=train_score, reasoning="test")

    train_card = ScoreCard(results=[scored], failure_analysis="")

    val_scored = ScoredResult(
        sample=sample, response=response, score=val_score, reasoning="test val"
    )
    val_card = ScoreCard(results=[val_scored], failure_analysis="")

    reflection = Reflection(
        round_num=round_num,
        diagnosis="Test diagnosis",
        failure_patterns=["pattern1"],
        proposed_changes=["change1"],
        priority="medium",
    )

    mutation = Mutation(
        description=f"Mutation for round {round_num}",
        original_config={"key": "old"},
        proposed_config={"key": "new"},
        reasoning="Test reasoning",
    )

    return RoundResult(
        round_num=round_num,
        train_scores=train_card,
        val_scores=val_card,
        reflection=reflection,
        mutation=mutation,
        applied=True,
    )


def test_persistence_run_log() -> None:
    """Test RunLog JSONL persistence and TSV append."""
    print("\n===== PERSISTENCE TEST =====\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "test_logs"
        run_log = RunLog(log_dir)

        # Create and append 3 round results
        rounds_data = [
            (1, 0.55, 0.50),
            (2, 0.65, 0.62),
            (3, 0.75, 0.72),
        ]

        for round_num, train_score, val_score in rounds_data:
            rr = _make_dummy_round(round_num, train_score, val_score)
            run_log.append_round(rr)

            # Also test TSV append
            run_log.append_tsv_line(
                round_num=round_num,
                train_score=train_score,
                val_score=val_score,
                status="kept",
                description=f"Round {round_num} completed successfully",
            )

        # Verify load_history
        history = run_log.load_history()
        assert len(history) == 3, f"Expected 3 rounds in history, got {len(history)}"

        for i, (round_num, train_score, val_score) in enumerate(rounds_data):
            rr = history[i]
            assert rr.round_num == round_num
            assert abs(rr.train_scores.mean_score - train_score) < 1e-6
            assert rr.val_scores is not None
            assert abs(rr.val_scores.mean_score - val_score) < 1e-6

        # Verify get_latest_run
        latest = run_log.get_latest_run()
        assert latest is not None
        assert latest.round_num == 3

        # Verify get_best_round
        best = run_log.get_best_round()
        assert best is not None
        assert best.round_num == 3, f"Expected best round 3, got {best.round_num}"

        # Verify TSV file
        tsv_path = run_log.tsv_path
        assert tsv_path.exists(), "TSV file should exist"
        tsv_content = tsv_path.read_text(encoding="utf-8")
        tsv_lines = tsv_content.strip().split("\n")
        assert len(tsv_lines) == 4, f"Expected 4 lines (header + 3 data), got {len(tsv_lines)}"
        assert tsv_lines[0].startswith("round_num\t")

        # Verify JSONL file exists and is readable
        log_path = run_log.log_path
        assert log_path.exists()

        print(f"  RunLog path: {log_path}")
        print(f"  TSV path: {tsv_path}")
        print(f"  Rounds persisted: {len(history)}")
        print(f"  Latest round: {latest.round_num}")
        print(f"  Best round: {best.round_num} (val={best.val_scores.mean_score:.4f})")
        print(f"  TSV lines: {len(tsv_lines)}")
        print("\n  TSV content:")
        for line in tsv_lines:
            print(f"    {line}")

    print("\n===== END PERSISTENCE TEST =====\n")


# ---------------------------------------------------------------------------
# Test: guardrails standalone
# ---------------------------------------------------------------------------


def test_guardrails_with_round_history() -> None:
    """Verify all three guardrails produce correct verdicts."""
    print("\n===== GUARDRAILS TEST =====\n")

    # Build 3 rounds with improving scores (no overfitting, no regression)
    rounds = [
        _make_dummy_round(1, 0.55, 0.50),
        _make_dummy_round(2, 0.65, 0.62),
        _make_dummy_round(3, 0.75, 0.72),
    ]

    # Overfitting: gap is small (~0.03) so should pass
    og = OverfittingGuardrail(threshold=0.15)
    verdict = og.check(rounds)
    assert verdict.passed, f"Overfitting should pass, got: {verdict.reason}"
    print(f"  OverfittingGuardrail: PASSED - {verdict.reason}")

    # Regression: scores are improving so should pass
    rg = RegressionGuardrail(tolerance=0.05)
    verdict = rg.check(rounds)
    assert verdict.passed, f"Regression should pass, got: {verdict.reason}"
    print(f"  RegressionGuardrail: PASSED - {verdict.reason}")

    # Consistency: scores are stable so should pass
    cg = ConsistencyGuardrail(threshold=0.5, min_rounds=2)
    verdict = cg.check(rounds)
    assert verdict.passed, f"Consistency should pass, got: {verdict.reason}"
    print(f"  ConsistencyGuardrail: PASSED - {verdict.reason}")

    # Now test overfitting detection with a big gap
    overfit_round = _make_dummy_round(4, 0.95, 0.60)  # gap=0.35
    rounds_overfit = rounds + [overfit_round]
    verdict = og.check(rounds_overfit)
    assert not verdict.passed, "Overfitting should be flagged with gap=0.35"
    print(f"  OverfittingGuardrail (gap=0.35): FLAGGED - {verdict.reason[:80]}")

    # Test regression detection
    regression_round = _make_dummy_round(4, 0.60, 0.40)  # drops from 0.72
    rounds_regress = rounds + [regression_round]
    verdict = rg.check(rounds_regress)
    assert not verdict.passed, "Regression should be flagged"
    print(f"  RegressionGuardrail (drop): FLAGGED - {verdict.reason[:80]}")

    print("\n===== END GUARDRAILS TEST =====\n")
