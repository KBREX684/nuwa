"""Unit tests for nuwa.connectors.cli_adapter and nuwa.connectors.http_api."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuwa.connectors.cli_adapter import CliAdapter
from nuwa.connectors.http_api import HttpApiAdapter
from nuwa.core.exceptions import ConnectorError

# ---------------------------------------------------------------------------
# CliAdapter tests
# ---------------------------------------------------------------------------


class TestCliAdapter:
    """Tests for the CLI subprocess adapter."""

    def test_default_name(self) -> None:
        adapter = CliAdapter(command="echo")
        assert "echo" in repr(adapter)

    @pytest.mark.asyncio
    async def test_invoke_echo(self) -> None:
        """Test basic invocation with echo command."""
        adapter = CliAdapter(command="echo", input_mode="arg")
        resp = await adapter.invoke("hello world")
        assert "hello world" in resp.output_text
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_invoke_stdin(self) -> None:
        """Test stdin input mode with cat."""
        adapter = CliAdapter(command="cat", input_mode="stdin")
        resp = await adapter.invoke("test input")
        assert "test input" in resp.output_text

    @pytest.mark.asyncio
    async def test_invoke_with_config(self, tmp_path: Path) -> None:
        """Config should be written before invocation."""
        config_file = tmp_path / "config.json"
        adapter = CliAdapter(command="echo", input_mode="arg", config_file=str(config_file))
        _ = await adapter.invoke("test", config={"key": "value"})
        assert config_file.exists()
        loaded = json.loads(config_file.read_text())
        assert loaded == {"key": "value"}

    @pytest.mark.asyncio
    async def test_command_not_found(self) -> None:
        adapter = CliAdapter(command="nonexistent_command_xyz_12345")
        with pytest.raises(ConnectorError, match="not found"):
            await adapter.invoke("test")

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        adapter = CliAdapter(command="sleep", args=["10"], timeout=1)
        with pytest.raises(ConnectorError, match="timed out"):
            await adapter.invoke("test")

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self) -> None:
        adapter = CliAdapter(command="false")
        resp = await adapter.invoke("test")
        assert resp.raw_metadata.get("returncode", 0) != 0

    def test_get_current_config_no_file(self) -> None:
        adapter = CliAdapter(command="echo")
        assert adapter.get_current_config() == {}

    def test_get_current_config_with_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"model": "v1"}), encoding="utf-8")
        adapter = CliAdapter(command="echo", config_file=str(config_file))
        config = adapter.get_current_config()
        assert config == {"model": "v1"}

    def test_apply_config_writes_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text("{}", encoding="utf-8")
        adapter = CliAdapter(command="echo", config_file=str(config_file))
        adapter.apply_config({"temperature": 0.5})

        loaded = json.loads(config_file.read_text())
        assert loaded == {"temperature": 0.5}

    def test_yaml_config_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: v1\ntemp: 0.5\n", encoding="utf-8")
        adapter = CliAdapter(command="echo", config_file=str(config_file))
        config = adapter.get_current_config()
        assert config["model"] == "v1"

    def test_repr(self) -> None:
        adapter = CliAdapter(command="echo", input_mode="stdin", timeout=60)
        r = repr(adapter)
        assert "echo" in r
        assert "stdin" in r


# ---------------------------------------------------------------------------
# HttpApiAdapter tests (with mocked aiohttp)
# ---------------------------------------------------------------------------


class TestHttpApiAdapter:
    """Tests for the HTTP API adapter (using mocked network)."""

    def test_default_repr(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat")
        assert "localhost" in repr(adapter)

    def test_get_current_config_default(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat")
        assert adapter.get_current_config() == {}

    def test_apply_config_caches(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat")
        adapter.apply_config({"temperature": 0.9})
        assert adapter.get_current_config() == {"temperature": 0.9}

    def test_extract_output_default_field(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat", output_field="reply")
        body = {"reply": "Hello there", "extra": True}
        assert adapter._extract_output(body) == "Hello there"

    def test_extract_output_fallback_keys(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat", output_field="custom")
        body = {"response": "fallback output"}
        assert adapter._extract_output(body) == "fallback output"

    def test_extract_output_non_dict(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat")
        assert adapter._extract_output("plain text") == "plain text"

    def test_extract_output_none_value(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat", output_field="result")
        body = {"no_result_here": True}
        result = adapter._extract_output(body)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        adapter = HttpApiAdapter(url="http://localhost:5000/chat")
        session = await adapter._get_session()
        assert not session.closed
        await adapter.close()
        assert session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with HttpApiAdapter(url="http://localhost:5000/chat") as adapter:
            session = await adapter._get_session()
            assert not session.closed
        # After exiting context, session should be closed
        assert session.closed
