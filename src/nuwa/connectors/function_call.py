"""In-process Python callable adapter for target agents."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from nuwa.core.exceptions import ConnectorError
from nuwa.core.types import AgentResponse

logger = logging.getLogger(__name__)


class _LatencyTimer:
    """Tiny context-manager for measuring wall-clock milliseconds."""

    __slots__ = ("_start", "elapsed_ms")

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class FunctionCallAdapter:
    """Wraps an arbitrary Python callable as a target agent.

    Both synchronous and asynchronous callables are supported.  Synchronous
    functions are executed in the default event-loop executor so they never
    block the loop.

    Implements the ``TargetAgent`` protocol defined in
    :mod:`nuwa.core.protocols`.

    Parameters
    ----------
    func:
        The callable to invoke.  It must accept at least one positional
        argument (the input text).  An optional second keyword argument
        ``config`` will receive the active configuration dict when present.
    config:
        Initial configuration dict.  Passed to the callable on each
        invocation if the callable's signature accepts a ``config`` parameter.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func).__name__}")
        self._func = func
        self._config: dict[str, Any] = dict(config) if config else {}
        self._is_async = asyncio.iscoroutinefunction(func)
        self._accepts_config = self._check_accepts_config(func)

    # ------------------------------------------------------------------
    # TargetAgent protocol
    # ------------------------------------------------------------------

    async def invoke(
        self,
        input_text: str,
        config: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Call the wrapped function with *input_text* and return the result.

        If *config* is provided it is merged into the stored configuration
        for this call only.
        """
        effective_config = {**self._config, **(config or {})}
        kwargs: dict[str, Any] = {}
        if self._accepts_config:
            kwargs["config"] = effective_config

        try:
            with _LatencyTimer() as timer:
                if self._is_async:
                    result = await self._func(input_text, **kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._func(input_text, **kwargs),
                    )

            output_text = self._normalise_output(result)
            raw = result if isinstance(result, dict) else {"result": result}
            return AgentResponse(
                output_text=output_text,
                latency_ms=timer.elapsed_ms,
                raw_metadata=raw,
            )

        except Exception as exc:
            logger.error("Function %s raised: %s", self._func_name, exc)
            raise ConnectorError(f"Function {self._func_name} failed: {exc}") from exc

    def get_current_config(self) -> dict[str, Any]:
        """Return a copy of the current configuration dict."""
        return dict(self._config)

    def apply_config(self, config: dict[str, Any]) -> None:
        """Replace the stored configuration with *config*."""
        self._config = dict(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _func_name(self) -> str:
        return getattr(self._func, "__qualname__", repr(self._func))

    @staticmethod
    def _check_accepts_config(func: Callable[..., Any]) -> bool:
        """Return *True* if *func* has a ``config`` parameter."""
        try:
            sig = inspect.signature(func)
            return "config" in sig.parameters
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _normalise_output(result: Any) -> str:
        """Coerce an arbitrary return value into a string."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # Prefer common output keys
            for key in ("output", "response", "text", "result", "message"):
                if key in result:
                    return str(result[key])
            return str(result)
        return str(result)

    def __repr__(self) -> str:
        return f"FunctionCallAdapter(func={self._func_name!r})"
