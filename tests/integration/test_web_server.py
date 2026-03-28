"""Integration tests for Nuwa web server API behavior."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import nuwa.web.server as web_server
from nuwa.connectors.registry import create_connector


@pytest.fixture
def client() -> TestClient:
    """Provide a clean TestClient with reset global web state."""
    web_server._reset_state()
    web_server._state["current_config"] = None
    with TestClient(web_server.app) as test_client:
        yield test_client
    task = web_server._state.get("training_task")
    if task is not None and not task.done():
        task.cancel()
    web_server._reset_state()
    web_server._state["current_config"] = None


def _base_config_payload() -> dict[str, object]:
    return {
        "llm_model": "openai/gpt-4o-mini",
        "connector_type": "cli",
        "connector_params": {
            "command": "python3",
            "args": ["/tmp/nuwa_real_agent.py"],
            "input_mode": "stdin",
            "timeout": 20,
        },
        "training_direction": "test",
        "max_rounds": 2,
        "samples_per_round": 5,
        "train_val_split": 0.7,
        "overfitting_threshold": 0.15,
        "regression_tolerance": 0.05,
        "consistency_threshold": 0.8,
    }


def test_static_missing_asset_returns_404(client: TestClient) -> None:
    """Missing files under /static should return 404, not SPA index.html."""
    resp = client.get("/static/does-not-exist.js")
    assert resp.status_code == 404


def test_function_connector_supports_module_reference(tmp_path: Path) -> None:
    """Registry should accept module-path strings for function connectors."""
    module_file = tmp_path / "demo_agent.py"
    module_file.write_text(
        "def run_agent(input_text, config=None):\n    return {'output': f'echo:{input_text}'}\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        adapter = create_connector("function", module="demo_agent:run_agent")
        response = asyncio.run(adapter.invoke("hello"))
        assert response.output_text == "echo:hello"
    finally:
        sys.path.pop(0)
        sys.modules.pop("demo_agent", None)


def test_train_stop_cancellation_sets_non_running_status(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the background task should not leave status stuck on running."""

    class FakeTrainingLoop:
        def __init__(self, **_: object) -> None:
            pass

        async def run(self) -> object:
            await asyncio.sleep(30)
            return object()

    monkeypatch.setattr(web_server, "TrainingLoop", FakeTrainingLoop)

    cfg_resp = client.post("/api/config", json=_base_config_payload())
    assert cfg_resp.status_code == 200

    start_resp = client.post("/api/train/start")
    assert start_resp.status_code == 200

    stop_resp = client.post("/api/train/stop")
    assert stop_resp.status_code == 200

    time.sleep(0.2)
    status_resp = client.get("/api/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["training_status"] == "completed"


def test_extend_uses_incremental_rounds(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`extend` should add extra rounds on top of the current max_rounds."""

    async def fake_run_training(config: object, backend: object, target: object) -> None:
        web_server._state["training_status"] = "completed"

    monkeypatch.setattr(web_server, "_run_training", fake_run_training)

    payload = _base_config_payload()
    payload["max_rounds"] = 10
    cfg_resp = client.post("/api/config", json=payload)
    assert cfg_resp.status_code == 200

    extend_resp = client.post(
        "/api/approve",
        json={"decision": "extend", "extra_rounds": 3},
    )
    assert extend_resp.status_code == 200
    current_cfg = web_server._state["current_config"]
    assert current_cfg.max_rounds == 13
