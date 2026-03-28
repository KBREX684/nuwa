"""End-to-end integration test: full training loop against a real LLM.

This test validates that the entire Nuwa pipeline (dataset gen → execute →
evaluate → reflect → mutate → validate) works end-to-end with real LLM API
calls.  It is skipped automatically when no API key is available, so it
won't break CI runs on machines without credentials.

Run manually with:

    OPENAI_API_KEY=sk-... pytest tests/integration/test_e2e_training.py -v

Or with any LiteLLM-compatible provider:

    NUWA_E2E_MODEL=anthropic/claude-sonnet-4-20250514 \
    ANTHROPIC_API_KEY=sk-ant-... \
    pytest tests/integration/test_e2e_training.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip condition: only run when an API key is explicitly provided.
# ---------------------------------------------------------------------------

_HAS_API_KEY = bool(
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("AZURE_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
)

_E2E_MODEL = os.environ.get("NUWA_E2E_MODEL", "openai/gpt-4o-mini")

skip_no_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="No LLM API key set (set OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY to run).",
)


# ---------------------------------------------------------------------------
# A trivial trainable agent for E2E testing
# ---------------------------------------------------------------------------


def _echo_agent(user_input: str, config: dict[str, Any] | None = None) -> str:
    """Minimal agent that echoes input with a configurable prefix."""
    prefix = (config or {}).get("system_prompt", "Echo:")
    return f"{prefix} {user_input}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_key
@pytest.mark.asyncio
async def test_e2e_full_loop_completes() -> None:
    """Full training loop should complete without errors and return results."""
    import nuwa

    result = await nuwa.train(
        _echo_agent,
        training_direction="让回复更简洁",
        model=_E2E_MODEL,
        max_rounds=2,
        samples_per_round=3,
        train_val_split=0.7,
        auto_promote=False,
        verbose=False,
    )

    # Should complete successfully.
    assert result is not None
    # Scheduler returns e.g. "max_rounds (2) reached" or "converged: ..."
    sr = result.stop_reason
    assert (
        "max_rounds" in sr
        or "converged" in sr
        or "target" in sr
    ), f"Unexpected stop_reason: {sr}"
    assert len(result.rounds) > 0
    assert len(result.rounds) <= 2


@skip_no_key
@pytest.mark.asyncio
async def test_e2e_trainer_promote_discard() -> None:
    """NuwaTrainer promote/discard lifecycle should work."""
    from nuwa.sdk.trainer import NuwaTrainer

    trainer = NuwaTrainer(
        agent=_echo_agent,
        training_direction="让回复更友好",
        model=_E2E_MODEL,
        max_rounds=1,
        samples_per_round=2,
        sandbox=True,
        verbose=False,
    )

    result = await trainer.run()
    assert result is not None

    # Promote should apply config.
    if result.best_val_score >= 0.1:
        promoted = trainer.promote()
        assert isinstance(promoted, dict)

    # Discard should restore original config.
    restored = trainer.discard()
    assert isinstance(restored, dict)


@skip_no_key
@pytest.mark.asyncio
async def test_e2e_trainable_decorator() -> None:
    """@trainable decorated function should work with train()."""
    import nuwa

    @nuwa.trainable(name="EchoBot", config_schema={"system_prompt": str})
    def my_bot(user_input: str, config: dict[str, Any] | None = None) -> str:
        prompt = (config or {}).get("system_prompt", "Hi")
        return f"[{prompt}] {user_input}"

    assert hasattr(my_bot, "nuwa_meta")
    assert my_bot.nuwa_meta.name == "EchoBot"

    # Use async train() since we're inside an existing event loop.
    result = await nuwa.train(
        my_bot,
        training_direction="提升回复质量",
        model=_E2E_MODEL,
        max_rounds=1,
        samples_per_round=2,
        auto_promote=False,
        verbose=False,
    )

    assert result is not None
    assert len(result.rounds) == 1
