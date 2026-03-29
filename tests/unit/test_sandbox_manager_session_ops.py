"""Unit tests for SandboxManager session-level APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from nuwa.core.types import AgentResponse
from nuwa.sandbox.manager import SandboxManager


class _MockTargetAgent:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {"mode": "prod", "temp": 0.5})

    async def invoke(self, input_text: str, config: dict[str, Any] | None = None) -> AgentResponse:
        return AgentResponse(
            output_text=input_text, latency_ms=1.0, raw_metadata={"config": config}
        )

    def get_current_config(self) -> dict[str, Any]:
        return dict(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        self._config = dict(config)


def test_promote_and_discard_session_by_id() -> None:
    real_agent = _MockTargetAgent({"mode": "prod", "temp": 0.5})

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SandboxManager(real_agent, project_dir=Path(tmpdir))
        sandboxed = manager.enter_sync()

        promoted = manager.promote_session(
            sandboxed.session_id,
            config_override={"mode": "exp", "temp": 0.9},
        )
        assert promoted["mode"] == "exp"
        assert real_agent.get_current_config()["mode"] == "exp"

        discarded = manager.discard_session(sandboxed.session_id)
        assert discarded["mode"] == "prod"
