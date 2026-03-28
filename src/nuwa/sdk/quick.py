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
from collections.abc import Callable
from typing import Any

from nuwa.core.protocols import TargetAgent
from nuwa.core.types import TrainingResult

logger = logging.getLogger(__name__)


async def train(
    agent: Callable[..., Any] | TargetAgent,
    training_direction: str,
    *,
    model: str = "openai/gpt-4o",
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    max_rounds: int = 10,
    auto_promote: bool = False,
    promote_threshold: float = 0.8,
    verbose: bool = True,
    # Backward-compatible aliases
    direction: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
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
    training_direction:
        Natural-language training goal (e.g. ``"提升回答质量"``).
    model:
        LLM model identifier in ``provider/model`` format.
    llm_api_key:
        Optional API key override for the LLM provider.
    llm_base_url:
        Optional base URL override for the LLM provider.
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
    # Resolve backward-compatible aliases
    if direction is not None:
        training_direction = direction
    if api_key is not None and llm_api_key is None:
        llm_api_key = api_key
    if base_url is not None and llm_base_url is None:
        llm_base_url = base_url

    from nuwa.sdk.trainer import NuwaTrainer

    trainer = NuwaTrainer(
        agent=agent,
        training_direction=training_direction,
        model=model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
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
    training_direction: str = "",
    *,
    direction: str | None = None,
    **kwargs: Any,
) -> TrainingResult:
    """Synchronous version of :func:`train` for non-async contexts.

    Internally calls ``asyncio.run(train(...))``.  Suitable for scripts,
    notebooks, and CLI tools that do not already have a running event loop.

    Parameters
    ----------
    agent:
        The agent to train.
    training_direction:
        Natural-language training goal.
    direction:
        Backward-compatible alias for *training_direction*.
    **kwargs:
        All other keyword arguments are forwarded to :func:`train`.

    Returns
    -------
    TrainingResult
        Complete summary of the training run.
    """
    if direction is not None and not training_direction:
        training_direction = direction
    return asyncio.run(train(agent, training_direction, **kwargs))
