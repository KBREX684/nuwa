"""One-liner convenience functions for Nuwa agent training.

Usage::

    import nuwa

    # Async
    result = await nuwa.train(my_agent, direction="提升回答质量", model="openai/gpt-4o")

    # Sync (for scripts / notebooks without an event loop)
    result = nuwa.train_sync(my_agent, direction="提升回答质量")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from nuwa.core.protocols import TargetAgent
from nuwa.core.types import TrainingResult

logger = logging.getLogger(__name__)


async def train(
    agent: Callable[..., Any] | TargetAgent,
    direction: str,
    *,
    model: str = "openai/gpt-4o",
    api_key: str | None = None,
    max_rounds: int = 5,
    auto_promote: bool = False,
    promote_threshold: float = 0.8,
    verbose: bool = True,
    **kwargs: Any,
) -> TrainingResult:
    """One-liner training function.

    Creates a :class:`~nuwa.sdk.trainer.NuwaTrainer`, runs the full training
    loop, and optionally auto-promotes the best config when the validation
    score exceeds *promote_threshold*.

    Parameters
    ----------
    agent:
        The agent to train -- a plain callable, a ``@trainable``-decorated
        function, or any ``TargetAgent`` implementation.
    direction:
        Natural-language training goal (e.g. ``"提升回答质量"``).
    model:
        LLM model identifier in ``provider/model`` format.
    api_key:
        Optional API key override for the LLM provider.
    max_rounds:
        Maximum number of training rounds.
    auto_promote:
        When *True*, automatically apply the best config to the agent if
        the best validation score meets or exceeds *promote_threshold*.
    promote_threshold:
        Minimum validation score required for auto-promotion.
    verbose:
        Enable verbose logging output.
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`~nuwa.sdk.trainer.NuwaTrainer`.

    Returns
    -------
    TrainingResult
        Complete summary of the training run.
    """
    from nuwa.sdk.trainer import NuwaTrainer

    trainer = NuwaTrainer(
        agent=agent,
        direction=direction,
        model=model,
        api_key=api_key,
        max_rounds=max_rounds,
        verbose=verbose,
        **kwargs,
    )

    result = await trainer.run()

    if auto_promote and result.best_val_score >= promote_threshold:
        trainer.promote()
        logger.info(
            "Auto-promoted config (best_val_score=%.4f >= threshold=%.4f).",
            result.best_val_score,
            promote_threshold,
        )
    elif auto_promote:
        logger.info(
            "Auto-promote skipped: best_val_score=%.4f < threshold=%.4f. "
            "Original config retained.",
            result.best_val_score,
            promote_threshold,
        )

    return result


def train_sync(
    agent: Callable[..., Any] | TargetAgent,
    direction: str,
    **kwargs: Any,
) -> TrainingResult:
    """Synchronous version of :func:`train` for non-async contexts.

    Internally calls ``asyncio.run(train(...))``.  Suitable for scripts,
    notebooks, and CLI tools that do not already have a running event loop.

    Parameters
    ----------
    agent:
        The agent to train.
    direction:
        Natural-language training goal.
    **kwargs:
        All other keyword arguments are forwarded to :func:`train`.

    Returns
    -------
    TrainingResult
        Complete summary of the training run.
    """
    return asyncio.run(train(agent, direction, **kwargs))
