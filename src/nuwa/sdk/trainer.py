"""High-level ``NuwaTrainer`` API for programmatic agent training.

Usage::

    from nuwa.sdk import NuwaTrainer

    trainer = NuwaTrainer(
        agent=my_agent,
        direction="提升回答准确率和友好度",
        model="openai/gpt-4o",
        max_rounds=5,
    )
    result = await trainer.run()

    if result.best_val_score > 0.8:
        trainer.promote()
    else:
        trainer.discard()
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nuwa.connectors.function_call import FunctionCallAdapter
from nuwa.core.protocols import TargetAgent
from nuwa.core.types import TrainingConfig, TrainingResult
from nuwa.engine.loop import TrainingLoop
from nuwa.guardrails.consistency import ConsistencyGuardrail
from nuwa.guardrails.overfitting import OverfittingGuardrail
from nuwa.guardrails.regression import RegressionGuardrail
from nuwa.llm.backend import LiteLLMBackend
from nuwa.sandbox.manager import SandboxManager
from nuwa.sdk.decorator import NuwaMeta

logger = logging.getLogger(__name__)


def _extract_initial_config(agent: Any) -> dict[str, Any]:
    """Best-effort extraction of initial config from a decorated or plain agent."""
    # If the agent is already a TargetAgent protocol implementation, ask it.
    if isinstance(agent, TargetAgent):
        return agent.get_current_config()

    # If decorated with @trainable, check for nuwa_meta.
    meta: NuwaMeta | None = getattr(agent, "nuwa_meta", None)
    if meta is not None and meta.config_schema is not None:
        # Build a default config from the schema types (empty defaults).
        return {k: "" if v is str else 0.0 for k, v in meta.config_schema.items()}

    return {}


def _wrap_as_target_agent(
    agent: Any,
    initial_config: dict[str, Any] | None = None,
) -> FunctionCallAdapter:
    """Wrap a plain callable (or @trainable-decorated function) as a TargetAgent."""
    # Unwrap to the real callable if it was decorated.
    func = agent
    meta: NuwaMeta | None = getattr(agent, "nuwa_meta", None)
    if meta is not None:
        func = meta.original_func

    config = initial_config if initial_config is not None else _extract_initial_config(agent)
    return FunctionCallAdapter(func=func, config=config)


class NuwaTrainer:
    """High-level API for training agents programmatically.

    Accepts both ``@trainable``-decorated functions and objects implementing the
    :class:`~nuwa.core.protocols.TargetAgent` protocol.

    Parameters
    ----------
    agent:
        The agent to train.  Can be a plain callable, a ``@trainable``-decorated
        function, or any object satisfying the ``TargetAgent`` protocol.
    direction:
        Natural-language description of the desired training goal (e.g.
        ``"提升回答准确率"``).
    model:
        LLM model identifier in ``provider/model`` format.
    api_key:
        API key for the LLM provider.  Falls back to environment variables when
        not supplied.
    base_url:
        Optional base URL override (Azure, self-hosted, etc.).
    max_rounds:
        Maximum number of train-eval-reflect-mutate rounds.
    samples_per_round:
        Number of evaluation samples generated per round.
    train_val_split:
        Fraction of samples allocated to the training set.
    overfitting_threshold:
        Maximum allowed train/val score gap before overfitting is flagged.
    regression_tolerance:
        Allowed score drop from historical best before regression is flagged.
    consistency_threshold:
        Minimum consistency ratio across repeated evaluations.
    sandbox:
        When *True* (default), the original agent config is preserved and
        mutations are applied only to an internal copy.  Call :meth:`promote`
        to apply the best config to the real agent.
    project_dir:
        Directory for persisted artefacts (logs, configs, run history).
    on_round_end:
        Optional callback invoked at the end of each training round.  Receives
        ``(round_result, context)`` as positional arguments.
    verbose:
        Enable verbose logging output.
    """

    def __init__(
        self,
        agent: Callable[..., Any] | TargetAgent,
        training_direction: str = "",
        *,
        model: str = "openai/gpt-4o",
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        max_rounds: int = 10,
        samples_per_round: int = 20,
        train_val_split: float = 0.7,
        overfitting_threshold: float = 0.15,
        regression_tolerance: float = 0.05,
        consistency_threshold: float = 0.8,
        sandbox: bool = True,
        project_dir: str | Path = ".nuwa",
        on_round_end: Callable[..., Any] | None = None,
        verbose: bool = True,
        # Backward-compatible aliases (preferred names above take precedence)
        direction: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Resolve backward-compatible aliases
        if direction is not None and not training_direction:
            training_direction = direction
        if not training_direction:
            raise ValueError(
                "training_direction is required. Pass training_direction='...' or direction='...'."
            )
        if api_key is not None and llm_api_key is None:
            llm_api_key = api_key
        if base_url is not None and llm_base_url is None:
            llm_base_url = base_url

        self._raw_agent = agent
        self._direction = training_direction
        self._sandbox = sandbox
        self._verbose = verbose
        self._project_dir = Path(project_dir)

        # Configure logging level based on verbose flag.
        if verbose:
            logging.getLogger("nuwa").setLevel(logging.INFO)
        else:
            logging.getLogger("nuwa").setLevel(logging.WARNING)

        # Resolve the TargetAgent implementation.
        if isinstance(agent, TargetAgent):
            self._target: TargetAgent = agent
        else:
            self._target = _wrap_as_target_agent(agent)

        # Snapshot the original config for sandbox / discard.
        self._original_config: dict[str, Any] = copy.deepcopy(self._target.get_current_config())

        # Build the LLM backend.
        self._backend = LiteLLMBackend(
            model=model,
            api_key=llm_api_key,
            base_url=llm_base_url,
        )

        # Build the training config.
        self._training_config = TrainingConfig(
            training_direction=training_direction,
            max_rounds=max_rounds,
            samples_per_round=samples_per_round,
            train_val_split=train_val_split,
            overfitting_threshold=overfitting_threshold,
            regression_tolerance=regression_tolerance,
            consistency_threshold=consistency_threshold,
        )

        # Standard guardrails.
        self._guardrails = [
            OverfittingGuardrail(threshold=overfitting_threshold),
            RegressionGuardrail(tolerance=regression_tolerance),
            ConsistencyGuardrail(threshold=consistency_threshold),
        ]

        # Callbacks.
        self._callbacks: list[Callable[..., Any]] = []
        if on_round_end is not None:
            self._callbacks.append(on_round_end)

        # Optional sandbox manager. When enabled, TrainingLoop will train
        # against an isolated config copy and keep the real target untouched.
        self._sandbox_manager: SandboxManager | None = None
        if self._sandbox:
            self._sandbox_manager = SandboxManager(
                self._target,
                project_dir=self._project_dir,
            )

        # Result placeholder -- populated after run().
        self._result: TrainingResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> TrainingResult:
        """Run the full training loop and return a :class:`TrainingResult`.

        The training loop executes up to ``max_rounds`` iterations of:
        dataset generation, agent execution, LLM-based evaluation,
        reflection, mutation proposal, and validation.

        Guardrails (overfitting, regression, consistency) are checked
        between rounds and may terminate training early.
        """
        loop = TrainingLoop(
            config=self._training_config,
            backend=self._backend,
            target=self._target,
            guardrails=self._guardrails,
            callbacks=self._callbacks,
            sandbox=self._sandbox_manager,
        )

        logger.info(
            "Starting Nuwa training: direction=%r, max_rounds=%d",
            self._direction,
            self._training_config.max_rounds,
        )

        self._result = await loop.run()

        logger.info(
            "Training complete: best_val_score=%.4f (round %d), stop_reason=%r",
            self._result.best_val_score,
            self._result.best_round,
            self._result.stop_reason,
        )

        return self._result

    def promote(self) -> dict[str, Any]:
        """Apply the best config found during training to the real agent.

        Only meaningful when ``sandbox=True``.  After calling this method the
        agent will use the optimised configuration for all future invocations.

        Returns
        -------
        dict
            The promoted (best) configuration dict.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet, or no best config exists.
        """
        if self._result is None:
            raise RuntimeError("Cannot promote before training. Call await trainer.run() first.")

        best = self._result.final_config
        if not best:
            raise RuntimeError("No best config was recorded during training.")

        if self._sandbox_manager is not None and self._result.sandbox_session_id:
            promoted = self._sandbox_manager.promote_session(
                self._result.sandbox_session_id,
                config_override=best,
            )
            logger.info(
                "Promoted best config via sandbox session %s.",
                self._result.sandbox_session_id,
            )
            return dict(promoted)

        self._target.apply_config(best)
        logger.info("Promoted best config to agent directly: %s", best)
        return dict(best)

    def discard(self) -> dict[str, Any]:
        """Discard all training changes and restore the original agent config.

        Returns
        -------
        dict
            The original configuration dict that was restored.
        """
        session_id = self._result.sandbox_session_id if self._result is not None else None
        if self._sandbox_manager is not None and session_id:
            original = self._sandbox_manager.discard_session(session_id)
            self._target.apply_config(copy.deepcopy(original))
            logger.info("Discarded sandbox session %s and restored original config.", session_id)
            return dict(original)

        self._target.apply_config(copy.deepcopy(self._original_config))
        logger.info("Discarded training changes; restored original config.")
        return dict(self._original_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_config(self) -> dict[str, Any] | None:
        """The best configuration found during training, or *None* if not yet run."""
        if self._result is None:
            return None
        return dict(self._result.final_config) if self._result.final_config else None

    @property
    def original_config(self) -> dict[str, Any] | None:
        """Snapshot of the agent's configuration before training began."""
        return dict(self._original_config) if self._original_config else None

    @property
    def result(self) -> TrainingResult | None:
        """The full :class:`TrainingResult`, or *None* if training has not run."""
        return self._result

    def __repr__(self) -> str:
        status = "trained" if self._result is not None else "pending"
        return (
            f"NuwaTrainer(direction={self._direction!r}, "
            f"max_rounds={self._training_config.max_rounds}, status={status!r})"
        )

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> NuwaTrainer:
        return self

    async def __aexit__(self, *_: object) -> None:
        """Clean up resources (e.g. sandbox session) when used as a context manager."""
        if self._sandbox_manager is not None:
            try:
                self.discard()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to clean up sandbox on exit: %s", exc)
            logger.debug("NuwaTrainer context exited, sandbox cleaned up.")
