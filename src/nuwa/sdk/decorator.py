"""The ``@trainable`` decorator for marking agent functions as Nuwa-trainable.

Usage::

    import nuwa

    @nuwa.trainable
    def my_chatbot(user_input: str, config: dict | None = None) -> str:
        system_prompt = config.get("system_prompt", "You are helpful.") if config else "You are helpful."
        return call_llm(system_prompt, user_input)

    # With options:
    @nuwa.trainable(name="CustomerBot", config_schema={"system_prompt": str})
    def customer_service(query: str, config: dict | None = None) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class NuwaMeta:
    """Metadata attached to a ``@trainable``-decorated function."""

    name: str
    """Human-readable name for this trainable agent."""

    description: str
    """Optional description of what the agent does."""

    config_schema: dict[str, Any] | None
    """Optional schema describing the config keys the agent accepts."""

    accepts_config: bool
    """Whether the decorated function has a ``config`` parameter."""

    original_func: Callable[..., Any]
    """Reference to the unwrapped original function."""


def _detect_accepts_config(func: Callable[..., Any]) -> bool:
    """Return *True* if *func* has a ``config`` parameter in its signature."""
    try:
        sig = inspect.signature(func)
        return "config" in sig.parameters
    except (ValueError, TypeError):
        return False


def _attach_convenience_methods(wrapper: Any, meta: NuwaMeta) -> None:
    """Add ``.train()`` and ``.get_trainer()`` shortcuts to *wrapper*."""

    def get_trainer(**kwargs: Any) -> Any:
        """Return a :class:`~nuwa.sdk.trainer.NuwaTrainer` bound to this agent.

        All keyword arguments are forwarded to the ``NuwaTrainer`` constructor.
        A ``direction`` keyword argument is required.
        """
        from nuwa.sdk.trainer import NuwaTrainer

        return NuwaTrainer(agent=wrapper, **kwargs)

    def train(direction: str, **kwargs: Any) -> Any:
        """Shortcut: create a trainer and call ``await trainer.run()``.

        Returns
        -------
        Coroutine that resolves to a :class:`~nuwa.core.types.TrainingResult`.
        """
        from nuwa.sdk.quick import train as _train

        return _train(wrapper, direction=direction, **kwargs)

    wrapper.get_trainer = get_trainer
    wrapper.train = train


def _make_wrapper(func: Callable[..., Any], meta: NuwaMeta) -> Callable[..., Any]:
    """Wrap *func* so it keeps its original behaviour but carries Nuwa metadata."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    wrapper.nuwa_meta = meta  # type: ignore[attr-defined]
    _attach_convenience_methods(wrapper, meta)
    return wrapper


# -----------------------------------------------------------------------
# Public decorator
# -----------------------------------------------------------------------


@overload
def trainable(func: F) -> F: ...


@overload
def trainable(
    func: None = None,
    *,
    name: str | None = None,
    config_schema: dict[str, Any] | None = None,
    description: str = "",
) -> Callable[[F], F]: ...


def trainable(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    config_schema: dict[str, Any] | None = None,
    description: str = "",
) -> Any:
    """Decorator that marks a function as a Nuwa-trainable agent.

    Can be used with or without arguments::

        @nuwa.trainable
        def my_agent(text, config=None): ...

        @nuwa.trainable(name="My Agent", config_schema={"system_prompt": str})
        def my_agent(text, config=None): ...

    The decorated function retains its original calling behaviour and gains:

    * ``nuwa_meta`` -- a :class:`NuwaMeta` dataclass with name, config_schema,
      description, and introspection results.
    * ``train(direction, **kw)`` -- shortcut that returns an awaitable
      :class:`~nuwa.core.types.TrainingResult`.
    * ``get_trainer(**kw)`` -- returns a :class:`~nuwa.sdk.trainer.NuwaTrainer`
      instance for full control.

    Parameters
    ----------
    func:
        The function to decorate.  When called without parentheses the
        function is passed directly; when called *with* keyword arguments
        this is ``None`` and the return value is a secondary decorator.
    name:
        Human-readable name.  Defaults to ``func.__name__``.
    config_schema:
        Optional dict describing the configuration keys the agent accepts.
    description:
        Free-text description of the agent's purpose.
    """

    def decorator(fn: Callable[..., Any]) -> Any:
        resolved_name = name if name is not None else getattr(fn, "__name__", "unnamed")
        meta = NuwaMeta(
            name=resolved_name,
            description=description,
            config_schema=config_schema,
            accepts_config=_detect_accepts_config(fn),
            original_func=fn,
        )
        return _make_wrapper(fn, meta)

    # Called as @trainable (no parentheses) -- ``func`` is the decorated function.
    if func is not None:
        return decorator(func)

    # Called as @trainable(...) -- return the real decorator.
    return decorator
