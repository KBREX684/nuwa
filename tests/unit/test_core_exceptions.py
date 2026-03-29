"""Unit tests for nuwa.core.exceptions — custom exception hierarchy."""

from __future__ import annotations

import pytest

from nuwa.core.exceptions import (
    ConfigError,
    ConnectorError,
    GuardrailTriggered,
    LLMError,
    NuwaError,
    TrainingAborted,
    ValidationError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from NuwaError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigError,
            ConnectorError,
            LLMError,
            GuardrailTriggered,
            TrainingAborted,
            ValidationError,
        ],
    )
    def test_inherits_base(self, exc_cls):
        assert issubclass(exc_cls, NuwaError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigError,
            ConnectorError,
            LLMError,
            GuardrailTriggered,
            TrainingAborted,
            ValidationError,
        ],
    )
    def test_is_exception(self, exc_cls):
        assert issubclass(exc_cls, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(NuwaError):
            raise ConfigError("bad config")

        with pytest.raises(NuwaError):
            raise ConnectorError("connection failed")

        with pytest.raises(NuwaError):
            raise LLMError("LLM timeout")

    def test_specific_catch(self):
        with pytest.raises(LLMError):
            raise LLMError("api error")

        # ConfigError should NOT catch LLMError
        with pytest.raises(LLMError):
            try:
                raise LLMError("api error")
            except ConfigError:
                pass  # should not reach here

    def test_message_preserved(self):
        try:
            raise ConfigError("missing field: api_key")
        except ConfigError as e:
            assert "missing field: api_key" in str(e)

    def test_guardrail_triggered(self):
        with pytest.raises(GuardrailTriggered):
            raise GuardrailTriggered("overfitting detected")

    def test_training_aborted(self):
        with pytest.raises(TrainingAborted):
            raise TrainingAborted("user stopped")

    def test_validation_error(self):
        with pytest.raises(ValidationError):
            raise ValidationError("schema mismatch")

    def test_catch_base_catches_all(self):
        """Catching NuwaError catches all subclasses."""
        errors = [
            ConfigError("a"),
            ConnectorError("b"),
            LLMError("c"),
            GuardrailTriggered("d"),
            TrainingAborted("e"),
            ValidationError("f"),
        ]
        for err in errors:
            with pytest.raises(NuwaError):
                raise err
