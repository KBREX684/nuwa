"""Unit tests for nuwa.llm.backend — LiteLLMBackend with circuit breaker."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nuwa.core.exceptions import LLMError
from nuwa.llm.backend import LiteLLMBackend


def _mock_response(content: str = "Hello!") -> MagicMock:
    """Create a mock litellm response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    return response


class TestLiteLLMBackendInit:
    def test_defaults(self):
        b = LiteLLMBackend()
        assert b.model == "openai/gpt-4o"
        assert b.temperature == 0.7
        assert b.api_key is None
        assert b.base_url is None
        assert b._cb_failures == 0

    def test_custom_model(self):
        b = LiteLLMBackend(model="deepseek/deepseek-chat", api_key="sk-test")
        assert b.model == "deepseek/deepseek-chat"
        assert b.api_key == "sk-test"


class TestComplete:
    @pytest.mark.asyncio
    async def test_success(self):
        b = LiteLLMBackend()
        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("Hi there"))
            result = await b.complete([{"role": "user", "content": "Hello"}])
            assert result == "Hi there"

    @pytest.mark.asyncio
    async def test_null_content_raises(self):
        b = LiteLLMBackend()
        msg = MagicMock()
        msg.content = None
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=response)
            with pytest.raises(LLMError, match="null content"):
                await b.complete([{"role": "user", "content": "Hi"}])


class TestCircuitBreaker:
    def _make_rate_limit_error(self):
        """Create a RateLimitError type for testing."""

        class RateLimitError(Exception):
            pass

        return RateLimitError("rate limited")

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """After 5 consecutive failures, circuit breaker should open."""
        b = LiteLLMBackend()
        error = self._make_rate_limit_error()

        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=error)
            for _ in range(5):
                with pytest.raises(LLMError):
                    await b.complete([{"role": "user", "content": "test"}])

            assert b._cb_failures >= 5
            assert b._cb_open_until > 0

    @pytest.mark.asyncio
    async def test_blocks_when_open(self):
        b = LiteLLMBackend()
        b._cb_failures = 5
        b._cb_open_until = time.monotonic() + 60.0  # open for 60s

        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            with pytest.raises(LLMError, match="Circuit breaker open"):
                await b.complete([{"role": "user", "content": "test"}])
            mock_litellm.acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_half_open_allows_one(self):
        b = LiteLLMBackend()
        b._cb_failures = 5
        b._cb_open_until = time.monotonic() - 1.0  # expired

        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("recovered"))
            result = await b.complete([{"role": "user", "content": "test"}])
            assert result == "recovered"
            assert b._cb_failures == 0

    @pytest.mark.asyncio
    async def test_resets_on_success(self):
        b = LiteLLMBackend()
        b._cb_failures = 3

        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("ok"))
            await b.complete([{"role": "user", "content": "test"}])
            assert b._cb_failures == 0


class TestTokenUsage:
    @pytest.mark.asyncio
    async def test_cumulative_tracking(self):
        b = LiteLLMBackend()
        with patch("nuwa.llm.backend.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("ok"))
            await b.complete([{"role": "user", "content": "test"}])
            assert b._total_prompt_tokens == 10
            assert b._total_completion_tokens == 20

            await b.complete([{"role": "user", "content": "test"}])
            assert b._total_prompt_tokens == 20
            assert b._total_completion_tokens == 40
