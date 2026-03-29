"""Unit tests for nuwa.connectors.function_call — FunctionCallAdapter."""

from __future__ import annotations

import pytest

from nuwa.connectors.function_call import FunctionCallAdapter
from nuwa.core.exceptions import ConnectorError


class TestFunctionCallAdapterInit:
    def test_sync_function(self):
        def agent(input_text: str) -> str:
            return f"echo: {input_text}"

        adapter = FunctionCallAdapter(func=agent)
        assert not adapter._is_async
        assert adapter._accepts_config is False

    def test_async_function(self):
        async def agent(input_text: str) -> str:
            return f"async: {input_text}"

        adapter = FunctionCallAdapter(func=agent)
        assert adapter._is_async

    def test_config_aware_function(self):
        def agent(input_text: str, config: dict | None = None) -> str:
            return f"config: {config}"

        adapter = FunctionCallAdapter(func=agent)
        assert adapter._accepts_config is True

    def test_non_callable_rejected(self):
        with pytest.raises(TypeError, match="callable"):
            FunctionCallAdapter(func="not a function")

    def test_repr(self):
        def my_func(x: str) -> str:
            return x

        adapter = FunctionCallAdapter(func=my_func)
        assert "my_func" in repr(adapter)


class TestFunctionCallAdapterInvoke:
    @pytest.mark.asyncio
    async def test_sync_invoke(self):
        def agent(input_text: str) -> str:
            return f"got: {input_text}"

        adapter = FunctionCallAdapter(func=agent)
        response = await adapter.invoke("hello")
        assert response.output_text == "got: hello"
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        async def agent(input_text: str) -> str:
            return f"async: {input_text}"

        adapter = FunctionCallAdapter(func=agent)
        response = await adapter.invoke("test")
        assert response.output_text == "async: test"

    @pytest.mark.asyncio
    async def test_invoke_with_config(self):
        def agent(input_text: str, config: dict | None = None) -> str:
            prompt = config.get("system_prompt", "default") if config else "default"
            return f"[{prompt}] {input_text}"

        adapter = FunctionCallAdapter(
            func=agent,
            config={"system_prompt": "be helpful"},
        )
        response = await adapter.invoke("hi")
        assert "be helpful" in response.output_text
        assert "hi" in response.output_text

    @pytest.mark.asyncio
    async def test_invoke_config_override(self):
        def agent(input_text: str, config: dict | None = None) -> str:
            return str(config)

        adapter = FunctionCallAdapter(func=agent, config={"a": 1})
        response = await adapter.invoke("x", config={"b": 2})
        # Merged config should contain both
        assert "a" in response.output_text
        assert "b" in response.output_text

    @pytest.mark.asyncio
    async def test_invoke_exception_raises_connector_error(self):
        def agent(input_text: str) -> str:
            raise ValueError("boom")

        adapter = FunctionCallAdapter(func=agent)
        with pytest.raises(ConnectorError, match="boom"):
            await adapter.invoke("test")


class TestFunctionCallAdapterConfig:
    def test_get_current_config(self):
        adapter = FunctionCallAdapter(func=lambda x: x, config={"key": "val"})
        cfg = adapter.get_current_config()
        assert cfg == {"key": "val"}
        # Should be a copy
        cfg["key"] = "modified"
        assert adapter.get_current_config()["key"] == "val"

    def test_apply_config(self):
        adapter = FunctionCallAdapter(func=lambda x: x, config={"old": True})
        adapter.apply_config({"new": True})
        assert adapter.get_current_config() == {"new": True}


class TestFunctionCallAdapterOutputNormalisation:
    @pytest.mark.asyncio
    async def test_string_output(self):
        adapter = FunctionCallAdapter(func=lambda x: "plain text")
        r = await adapter.invoke("x")
        assert r.output_text == "plain text"

    @pytest.mark.asyncio
    async def test_dict_output_extracts_text(self):
        def agent(x: str) -> dict:
            return {"output": "from dict", "meta": "ignored"}

        adapter = FunctionCallAdapter(func=agent)
        r = await adapter.invoke("x")
        assert r.output_text == "from dict"

    @pytest.mark.asyncio
    async def test_dict_output_fallback_to_str(self):
        def agent(x: str) -> dict:
            return {"custom_key": "value"}

        adapter = FunctionCallAdapter(func=agent)
        r = await adapter.invoke("x")
        # No standard key, so full dict as string
        assert "custom_key" in r.output_text

    @pytest.mark.asyncio
    async def test_int_output(self):
        adapter = FunctionCallAdapter(func=lambda x: 42)
        r = await adapter.invoke("x")
        assert r.output_text == "42"
