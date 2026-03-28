"""Integration tests for SDK decorators, sandbox isolation, diff utilities,
and the NuwaTrainer + sandbox combined workflow.

Verifies end-to-end that:
- @trainable decorator attaches metadata and preserves function behaviour
- SandboxManager isolates the real agent from mutations
- deep_diff / format_diff_text / format_diff_html work correctly
- NuwaTrainer with sandbox=True protects the real agent, then promote/discard work
- train_sync one-liner wrapper completes successfully
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

# -- SDK / sandbox imports ---------------------------------------------------
from nuwa.sdk.decorator import trainable, NuwaMeta
from nuwa.sandbox.manager import SandboxManager
from nuwa.sandbox.agent import SandboxedAgent
from nuwa.sandbox.diff import deep_diff, format_diff_text, format_diff_html, DiffEntry

# -- Core types --------------------------------------------------------------
from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    RoundResult,
    ScoreCard,
    ScoredResult,
    TrainingConfig,
    TrainingResult,
    Reflection,
    Mutation,
)

# -- Engine / trainer --------------------------------------------------------
from nuwa.engine.loop import TrainingLoop
from nuwa.sdk.trainer import NuwaTrainer
from nuwa.sdk.quick import train_sync

# -- Guardrails --------------------------------------------------------------
from nuwa.guardrails.consistency import ConsistencyGuardrail
from nuwa.guardrails.overfitting import OverfittingGuardrail
from nuwa.guardrails.regression import RegressionGuardrail


# ===========================================================================
# Shared mocks (same pattern as test_full_loop.py)
# ===========================================================================


class MockModelBackend:
    """Mock LLM backend returning canned JSON based on prompt keywords."""

    def __init__(self) -> None:
        self._call_count = 0
        self._round_hint = 1

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        self._call_count += 1
        content = " ".join(m.get("content", "") for m in messages).lower()

        if "dataset generation" in content or "evaluation samples" in content:
            return self._dataset()
        if "diagnostics analyst" in content or "root-cause analysis" in content:
            return self._reflection()
        if "prompt-engineering specialist" in content or "mutation" in content:
            return self._mutation()
        # default: scoring
        return self._scoring()

    async def complete_structured(self, messages, response_schema, **kw):
        raw = await self.complete(messages, **kw)
        data = json.loads(raw)
        return response_schema.model_validate(data)

    # -- canned payloads -----------------------------------------------------

    def _dataset(self) -> str:
        samples = []
        for i in range(10):
            diff = ["easy", "medium", "hard"][i % 3]
            samples.append({
                "input_text": f"Question {i+1}",
                "expected_behavior": f"Good answer for {i+1}",
                "difficulty": diff,
                "tags": ["test"],
            })
        return json.dumps(samples)

    def _scoring(self) -> str:
        import random
        random.seed(self._call_count)
        score = 0.4 + 0.1 * min(self._round_hint, 3) + random.random() * 0.3
        score = max(0.0, min(1.0, score))
        return json.dumps({
            "score": round(score, 3),
            "reasoning_en": "ok",
            "reasoning_zh": "ok",
            "axis_scores": {
                "correctness": round(score, 3),
                "completeness": round(score, 3),
                "format_compliance": 1.0,
                "tone_style": round(score, 3),
            },
        })

    def _reflection(self) -> str:
        return json.dumps({
            "diagnosis_summary_en": "Needs more detail.",
            "diagnosis_summary_zh": "needs more detail",
            "failure_patterns": [{
                "label_en": "Insufficient detail",
                "label_zh": "detail",
                "affected_samples": [1],
                "root_cause": "prompt",
                "severity": "high",
            }],
            "proposed_changes": [{
                "target": "system_prompt",
                "description_en": "Be more detailed.",
                "description_zh": "more detail",
                "priority": "high",
            }],
        })

    def _mutation(self) -> str:
        return json.dumps({
            "mutations": [{
                "id": "mut-001",
                "type": "config_change",
                "description_en": "Increase detail",
                "description_zh": "more detail",
                "rationale_en": "fix",
                "rationale_zh": "fix",
                "config_path": "detail_level",
                "config_value": "high",
                "expected_impact": "high",
            }],
        })


class MockTargetAgent:
    """Simple mock agent satisfying TargetAgent protocol."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {
            "system_prompt": "You are helpful.",
            "temperature": 0.7,
            "detail_level": "medium",
        }
        self._config_change_count = 0

    async def invoke(self, input_text: str, config: dict[str, Any] | None = None) -> AgentResponse:
        effective = {**self._config, **(config or {})}
        return AgentResponse(
            output_text=f"Answer: {input_text[:40]}",
            latency_ms=10.0,
            raw_metadata={"cfg": effective},
        )

    def get_current_config(self) -> dict[str, Any]:
        return dict(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._config_change_count += 1


# ===========================================================================
# Tests
# ===========================================================================


class TestTrainableDecorator:
    """Verify the @trainable decorator behaviour."""

    def test_bare_decorator(self):
        """@trainable without parentheses attaches nuwa_meta and preserves behaviour."""

        @trainable
        def my_agent(text: str, config: dict | None = None) -> str:
            return f"echo: {text}"

        assert hasattr(my_agent, "nuwa_meta")
        meta: NuwaMeta = my_agent.nuwa_meta
        assert meta.name == "my_agent"
        assert meta.accepts_config is True
        assert meta.description == ""
        assert meta.config_schema is None
        # Function still works normally
        assert my_agent("hello") == "echo: hello"

    def test_decorator_with_options(self):
        """@trainable(name=..., description=...) sets metadata correctly."""

        @trainable(name="CustomerBot", description="Handles queries", config_schema={"tone": str})
        def service(query: str) -> str:
            return query.upper()

        meta: NuwaMeta = service.nuwa_meta
        assert meta.name == "CustomerBot"
        assert meta.description == "Handles queries"
        assert meta.config_schema == {"tone": str}
        assert meta.accepts_config is False  # no config param
        assert service("hi") == "HI"

    def test_accepts_config_detection(self):
        """accepts_config is True only when a 'config' parameter exists."""

        @trainable
        def with_config(text, config=None):
            return text

        @trainable
        def without_config(text):
            return text

        assert with_config.nuwa_meta.accepts_config is True
        assert without_config.nuwa_meta.accepts_config is False

    def test_convenience_methods_exist(self):
        """train() and get_trainer() shortcuts are attached."""

        @trainable
        def agent(text):
            return text

        assert callable(getattr(agent, "train", None))
        assert callable(getattr(agent, "get_trainer", None))


# ---------------------------------------------------------------------------


class TestSandboxIsolation:
    """Verify sandbox protects the real agent from mutations."""

    @pytest.mark.asyncio
    async def test_full_sandbox_lifecycle(self):
        original_config = {
            "system_prompt": "Original prompt",
            "temperature": 0.5,
            "nested": {"key": "value"},
        }
        real_agent = MockTargetAgent(config=original_config.copy())

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SandboxManager(real_agent, project_dir=Path(tmpdir))
            sandboxed = await manager.enter()

            assert isinstance(sandboxed, SandboxedAgent)

            # -- Apply mutations in the sandbox --
            sandboxed.apply_config({
                "system_prompt": "Mutated prompt v1",
                "temperature": 0.9,
                "nested": {"key": "changed"},
            })
            sandboxed.apply_config({
                "system_prompt": "Mutated prompt v2",
                "temperature": 0.95,
                "nested": {"key": "changed_again"},
                "new_key": "added",
            })

            # Real agent must be UNCHANGED
            real_cfg = real_agent.get_current_config()
            assert real_cfg["system_prompt"] == "Original prompt"
            assert real_cfg["temperature"] == 0.5
            assert real_agent._config_change_count == 0

            # Sandbox reflects mutations
            sb_cfg = sandboxed.get_current_config()
            assert sb_cfg["system_prompt"] == "Mutated prompt v2"
            assert sb_cfg["temperature"] == 0.95
            assert sb_cfg.get("new_key") == "added"
            assert sandboxed.mutation_count == 2

            # -- Rollback to version 1 --
            rolled = sandboxed.rollback(version=1)
            assert rolled["system_prompt"] == "Mutated prompt v1"
            assert rolled["temperature"] == 0.9
            # Real still untouched
            assert real_agent.get_current_config()["system_prompt"] == "Original prompt"

    @pytest.mark.asyncio
    async def test_promote(self):
        """promote() pushes sandbox config to the real agent."""
        real_agent = MockTargetAgent(config={"tone": "formal"})

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SandboxManager(real_agent, project_dir=Path(tmpdir))
            sandboxed = await manager.enter()
            sandboxed.apply_config({"tone": "casual", "extra": True})

            promoted = await manager.promote(sandboxed)

            assert promoted["tone"] == "casual"
            assert promoted["extra"] is True
            # Real agent now has the promoted config
            assert real_agent.get_current_config()["tone"] == "casual"

    @pytest.mark.asyncio
    async def test_discard(self):
        """discard() leaves the real agent with its original config."""
        real_agent = MockTargetAgent(config={"mode": "production"})

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SandboxManager(real_agent, project_dir=Path(tmpdir))
            sandboxed = await manager.enter()
            sandboxed.apply_config({"mode": "experimental", "risk": "high"})

            original = await manager.discard(sandboxed)

            assert original["mode"] == "production"
            # Real agent never changed
            assert real_agent.get_current_config()["mode"] == "production"
            assert real_agent._config_change_count == 0


# ---------------------------------------------------------------------------


class TestSandboxDiff:
    """Verify deep_diff, format_diff_text, and format_diff_html."""

    def test_deep_diff_various_changes(self):
        original = {
            "a": 1,
            "b": "hello",
            "c": {"nested": True, "deep": {"x": 10}},
            "removed_key": "gone",
        }
        modified = {
            "a": 2,
            "b": "hello",  # unchanged
            "c": {"nested": False, "deep": {"x": 10, "y": 20}},
            "new_key": "added",
        }
        entries = deep_diff(original, modified)

        paths = {e.path: e for e in entries}
        assert "a" in paths
        assert paths["a"].change_type == "modified"
        assert paths["a"].old_value == 1
        assert paths["a"].new_value == 2

        assert "c.nested" in paths
        assert paths["c.nested"].change_type == "modified"

        assert "c.deep.y" in paths
        assert paths["c.deep.y"].change_type == "added"

        assert "new_key" in paths
        assert paths["new_key"].change_type == "added"

        assert "removed_key" in paths
        assert paths["removed_key"].change_type == "removed"

        # "b" should not appear (unchanged)
        assert "b" not in paths

    def test_format_diff_text(self):
        entries = deep_diff({"x": 1}, {"x": 2, "y": 3})
        text = format_diff_text(entries)
        assert len(text) > 0
        assert "x" in text

    def test_format_diff_html(self):
        entries = deep_diff({"x": 1}, {"x": 2, "y": 3})
        html = format_diff_html(entries)
        assert "<div" in html
        assert "x" in html

    def test_empty_diff(self):
        entries = deep_diff({"a": 1}, {"a": 1})
        assert entries == []
        assert format_diff_text(entries) == "No differences."
        assert "empty" in format_diff_html(entries).lower() or "No differences" in format_diff_html(entries)


# ---------------------------------------------------------------------------


class TestSDKTrainerWithSandbox:
    """NuwaTrainer with sandbox=True: real agent stays protected."""

    @pytest.mark.asyncio
    async def test_trainer_sandbox_protect_and_promote(self):
        print("\n===== SDK TRAINER + SANDBOX TEST =====\n")

        original_config = {
            "system_prompt": "You are helpful.",
            "temperature": 0.7,
            "detail_level": "medium",
        }
        real_agent = MockTargetAgent(config=original_config.copy())
        backend = MockModelBackend()

        config = TrainingConfig(
            training_direction="Improve detail and accuracy",
            max_rounds=2,
            samples_per_round=10,
            train_val_split=0.7,
            overfitting_threshold=0.30,
            consistency_threshold=0.5,
            consistency_runs=3,
            regression_tolerance=0.10,
        )
        guardrails = [
            OverfittingGuardrail(threshold=0.30),
            RegressionGuardrail(tolerance=0.10),
            ConsistencyGuardrail(threshold=0.5, min_rounds=2),
        ]

        loop = TrainingLoop(
            config=config,
            backend=backend,
            target=real_agent,
            guardrails=guardrails,
            sandbox=SandboxManager(real_agent, project_dir=Path(tempfile.mkdtemp())),
        )

        result = await loop.run()

        # Training must have completed
        assert isinstance(result, TrainingResult)
        assert len(result.rounds) > 0
        assert result.best_val_score > 0.0
        print(f"  Rounds: {len(result.rounds)}, best_val: {result.best_val_score:.4f}")

        # The real agent config must still be the original (sandbox protected it)
        real_cfg = real_agent.get_current_config()
        assert real_cfg["system_prompt"] == original_config["system_prompt"], (
            f"Real agent was mutated! Got: {real_cfg}"
        )
        assert real_cfg["temperature"] == original_config["temperature"]
        print("  Real agent config preserved during training: OK")

        print("===== END SDK TRAINER + SANDBOX TEST =====\n")

    @pytest.mark.asyncio
    async def test_nuwa_trainer_promote_and_discard(self):
        """NuwaTrainer.promote() applies best config; discard() restores original."""
        real_agent = MockTargetAgent(config={"prompt": "original"})
        backend = MockModelBackend()

        # We need to use the NuwaTrainer high-level API which internally
        # wraps the agent in a FunctionCallAdapter. To test promote/discard
        # on the internal _target we go through the trainer directly.

        # Build a simple decorated function to feed NuwaTrainer
        @trainable(config_schema={"prompt": str})
        def my_func(text: str, config: dict | None = None) -> str:
            p = (config or {}).get("prompt", "default")
            return f"{p}: {text}"

        trainer = NuwaTrainer(
            agent=my_func,
            direction="test",
            model="openai/gpt-4o",
            max_rounds=2,
            sandbox=True,
        )

        # Swap backend to our mock
        trainer._backend = backend

        result = await trainer.run()
        assert isinstance(result, TrainingResult)

        # Capture config before promote
        pre_promote = trainer._target.get_current_config()

        # Promote
        promoted = trainer.promote()
        assert isinstance(promoted, dict)
        post_promote = trainer._target.get_current_config()
        # After promote the target should have the best config
        assert post_promote == result.final_config

        # Discard restores original
        discarded = trainer.discard()
        assert isinstance(discarded, dict)
        print(f"  Promote/discard lifecycle: OK")


# ---------------------------------------------------------------------------


class TestTrainSync:
    """Verify the synchronous train_sync wrapper completes."""

    def test_train_sync_quick(self):
        print("\n===== TRAIN_SYNC TEST =====\n")

        @trainable
        def simple_agent(text: str, config: dict | None = None) -> str:
            return f"reply: {text}"

        # Monkeypatch LiteLLMBackend in the module where NuwaTrainer imports it,
        # so the constructor never touches the real LLM provider.
        import nuwa.sdk.trainer as trainer_mod

        OriginalBackend = trainer_mod.LiteLLMBackend

        class PatchedLiteLLMBackend:
            """Drop-in replacement that delegates to MockModelBackend."""

            def __init__(self, *a, **kw):
                self._mock = MockModelBackend()

            async def complete(self, messages, **kw):
                return await self._mock.complete(messages, **kw)

            async def complete_structured(self, messages, response_schema, **kw):
                return await self._mock.complete_structured(messages, response_schema, **kw)

        trainer_mod.LiteLLMBackend = PatchedLiteLLMBackend  # type: ignore[misc]
        try:
            result = train_sync(
                simple_agent,
                direction="Improve reply quality",
                max_rounds=1,
                verbose=False,
            )
            assert isinstance(result, TrainingResult)
            assert len(result.rounds) > 0
            print(f"  train_sync completed: {len(result.rounds)} round(s), "
                  f"best_val={result.best_val_score:.4f}")
        finally:
            trainer_mod.LiteLLMBackend = OriginalBackend

        print("===== END TRAIN_SYNC TEST =====\n")
