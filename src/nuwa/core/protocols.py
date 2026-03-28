"""Protocol (structural typing) interfaces for the Nuwa AI Trainer framework.

Every protocol is decorated with ``@runtime_checkable`` so that ``isinstance``
checks work at runtime, enabling lightweight dependency-injection validation
without requiring concrete inheritance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nuwa.core.types import (
    AgentResponse,
    GuardrailVerdict,
    LoopContext,
    RoundResult,
)

# ---------------------------------------------------------------------------
# LLM / model backend
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelBackend(Protocol):
    """Async interface for calling an LLM.

    Implementations wrap vendor-specific SDKs (OpenAI, Anthropic, local
    models, etc.) behind a uniform completion API.
    """

    async def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Return a plain-text completion for the given message list.

        Args:
            messages: Chat-style message dicts (role / content at minimum).
            **kwargs: Backend-specific overrides (temperature, max_tokens, …).

        Returns:
            The model's text response.
        """
        ...

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_schema: type,
        **kwargs: Any,
    ) -> Any:
        """Return a completion parsed into *response_schema*.

        Args:
            messages: Chat-style message dicts.
            response_schema: A Pydantic model (or similar) describing the
                expected response structure.
            **kwargs: Backend-specific overrides.

        Returns:
            An instance of *response_schema* populated from the model output.
        """
        ...


# ---------------------------------------------------------------------------
# Target agent under training
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetAgent(Protocol):
    """Interface for the agent whose behaviour is being improved."""

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Run the agent on a single input and return its response.

        Args:
            input_text: The user-facing prompt to send to the agent.
            config: Optional configuration overrides for this call.

        Returns:
            An :class:`AgentResponse` capturing the output and metadata.
        """
        ...

    def get_current_config(self) -> dict[str, Any]:
        """Return a snapshot of the agent's current configuration."""
        ...

    def apply_config(self, config: dict[str, Any]) -> None:
        """Mutate the agent's live configuration in place.

        Args:
            config: A (possibly partial) configuration dict whose keys
                overwrite the current values.
        """
        ...


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


@runtime_checkable
class Guardrail(Protocol):
    """A safety / sanity check executed between training rounds."""

    def check(self, history: list[RoundResult]) -> GuardrailVerdict:
        """Evaluate the training history and return a verdict.

        Args:
            history: All round results accumulated so far.

        Returns:
            A :class:`GuardrailVerdict` indicating whether to continue.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier for this guardrail."""
        ...


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@runtime_checkable
class Stage(Protocol):
    """A single step in the training-loop pipeline.

    Stages are composed sequentially; each one receives the shared
    :class:`LoopContext`, performs its work (possibly mutating the context),
    and returns it for the next stage.
    """

    async def execute(self, context: LoopContext) -> LoopContext:
        """Run this stage's logic.

        Args:
            context: The mutable loop state.

        Returns:
            The (possibly mutated) context to pass downstream.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier for this stage."""
        ...
