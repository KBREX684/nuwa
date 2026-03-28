"""Custom exception hierarchy for the Nuwa AI Trainer framework."""

from __future__ import annotations


class NuwaError(Exception):
    """Base exception for all Nuwa-related errors."""


class ConfigError(NuwaError):
    """Raised when a training or agent configuration is invalid or missing."""


class ConnectorError(NuwaError):
    """Raised when communication with an external service or backend fails."""


class LLMError(NuwaError):
    """Raised when an LLM call fails, times out, or returns unparseable output."""


class GuardrailTriggered(NuwaError):
    """Raised when a guardrail check fails and requires the loop to react."""


class TrainingAborted(NuwaError):
    """Raised when the training loop is intentionally stopped early."""


class ValidationError(NuwaError):
    """Raised when data validation (outside of Pydantic) fails at runtime."""
