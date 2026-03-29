"""Main training loop orchestrator for the Nuwa AI Trainer."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from nuwa.core.exceptions import (
    ConfigError,
    ConnectorError,
    GuardrailTriggered,
    LLMError,
    TrainingAborted,
)
from nuwa.core.protocols import Guardrail, ModelBackend, Stage, TargetAgent
from nuwa.core.types import (
    LoopContext,
    RoundResult,
    TrainingConfig,
    TrainingResult,
)
from nuwa.engine.objectives.pareto import ParetoTracker
from nuwa.engine.objectives.types import Objective, ObjectiveSet
from nuwa.engine.parallel.evaluator import EnsembleStrategy, JudgeConfig
from nuwa.engine.parallel.stage import (
    ParallelEvaluationStage,
    ParallelExecutionStage,
    ParallelValidationStage,
)
from nuwa.engine.scheduler import TrainingScheduler
from nuwa.engine.stages.dataset_gen import DatasetGenStage
from nuwa.engine.stages.evaluation import EvaluationStage
from nuwa.engine.stages.execution import ExecutionStage
from nuwa.engine.stages.mutation import MutationStage
from nuwa.engine.stages.reflection import ReflectionStage
from nuwa.engine.stages.validation import ValidationStage
from nuwa.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Orchestrates the full train-eval-reflect-mutate-validate loop."""

    def __init__(
        self,
        config: TrainingConfig,
        backend: ModelBackend,
        target: TargetAgent,
        guardrails: Sequence[Guardrail],
        callbacks: list[Callable[..., Any]] | None = None,
        sandbox: SandboxManager | None = None,
        parallel_config: dict[str, Any] | None = None,
        start_round: int = 1,
        initial_history: list[RoundResult] | None = None,
        initial_best_config: dict[str, Any] | None = None,
        initial_best_val_score: float = 0.0,
    ) -> None:
        self._config = config
        self._backend = backend
        self._target = target
        self._guardrails = list(guardrails)
        self._callbacks = callbacks or []
        self._sandbox = sandbox
        self._start_round = max(1, start_round)
        self._initial_history = list(initial_history) if initial_history else []
        self._initial_best_config = (
            dict(initial_best_config) if initial_best_config is not None else None
        )
        self._initial_best_val_score = max(0.0, float(initial_best_val_score))
        self._scheduler = TrainingScheduler(config)
        self._objective_set: ObjectiveSet | None = None
        self._pareto_tracker: ParetoTracker | None = None
        self._init_multi_objective(config)

        # Initialise the pipeline stages.
        # When *parallel_config* is provided, swap the execution, evaluation,
        # and validation stages for their high-throughput parallel variants.
        #
        # parallel_config format:
        #   {
        #       "max_concurrency": int (default 10),
        #       "judges": list[JudgeConfig],
        #       "strategy": str | EnsembleStrategy (default "mean"),
        #   }
        if parallel_config is not None:
            stages = self._build_parallel_stages(parallel_config)
            self._pre_validation_stages: list[Stage] = stages["pre_validation"]
            self._validation_stage: Stage = stages["validation"]
            logger.info(
                "Parallel mode enabled (concurrency=%s, judges=%d, strategy=%s)",
                parallel_config.get("max_concurrency", 10),
                len(parallel_config.get("judges", [])),
                parallel_config.get("strategy", "mean"),
            )
        else:
            self._pre_validation_stages = [
                DatasetGenStage(),
                ExecutionStage(),
                EvaluationStage(),
                ReflectionStage(),
                MutationStage(),
            ]
            self._validation_stage = ValidationStage()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> TrainingResult:
        """Execute the full training run and return the result summary.

        When a :class:`SandboxManager` is configured, the target agent is
        automatically wrapped in a :class:`SandboxedAgent` so that **no
        mutations reach the real agent** during training.  The caller (CLI /
        UI / SDK) is responsible for applying a final decision after the run
        (e.g. by calling ``SandboxManager.promote_session(result.sandbox_session_id)`` or
        ``SandboxManager.discard_session(result.sandbox_session_id)``).
        """
        start_time = time.monotonic()

        # --- sandbox wrapping ------------------------------------------------
        sandbox_session_id: str | None = None
        if self._sandbox is not None:
            sandboxed = await self._sandbox.enter()
            sandbox_session_id = sandboxed.session_id
            active_target: Any = sandboxed
            logger.info(
                "Sandbox mode enabled (session %s). "
                "The real agent will NOT be modified during training.",
                sandbox_session_id,
            )
        else:
            active_target = self._target

        starting_config = active_target.get_current_config()
        if self._initial_best_config is not None:
            active_target.apply_config(self._initial_best_config)
            starting_config = dict(self._initial_best_config)

        context = LoopContext(
            config=self._config,
            backend_ref=self._backend,
            target_ref=active_target,
            round_num=max(0, self._start_round - 1),
            current_config=starting_config,
            best_config=starting_config,
            best_val_score=self._initial_best_val_score,
        )
        if self._initial_history:
            context.history = list(self._initial_history)

        if self._initial_best_config is not None:
            context.best_config = dict(self._initial_best_config)

        if context.history and context.best_val_score <= 0.0:
            inferred_best = max(
                (
                    rr.val_scores.mean_score
                    for rr in context.history
                    if rr.val_scores is not None
                ),
                default=0.0,
            )
            context.best_val_score = inferred_best

        stop_reason = "max_rounds reached"
        if self._start_round > self._config.max_rounds:
            stop_reason = (
                f"resume no-op: start_round ({self._start_round}) > "
                f"max_rounds ({self._config.max_rounds})"
            )

        try:
            for round_num in range(self._start_round, self._config.max_rounds + 1):
                context.round_num = round_num
                logger.info("===== Round %d / %d =====", round_num, self._config.max_rounds)

                # Optionally adjust budget for this round.
                budget = self._scheduler.get_round_budget(round_num)
                context.config = self._config.model_copy(
                    update={"samples_per_round": budget["samples_per_round"]}
                )

                # Run pre-validation stages.
                try:
                    for stage in self._pre_validation_stages:
                        logger.debug("Running stage: %s", stage.name)
                        context = await stage.execute(context)
                except (TrainingAborted, GuardrailTriggered) as exc:
                    stop_reason = f"aborted during stage: {exc}"
                    logger.warning("Training aborted in round %d: %s", round_num, exc)
                    break
                except (LLMError, ConnectorError, ConfigError) as exc:
                    # Recoverable: skip this round, keep best config, continue.
                    logger.error(
                        "Round %d: recoverable error in pipeline: [%s] %s. "
                        "Skipping round, retaining best config.",
                        round_num,
                        type(exc).__name__,
                        exc,
                    )
                    from nuwa.core.types import Reflection as _Reflection
                    from nuwa.core.types import ScoreCard as _ScoreCard

                    fallback_scores = context.train_scores or _ScoreCard(
                        results=[],
                        failure_analysis=f"Round skipped: {type(exc).__name__}: {exc}",
                    )
                    fallback_reflection = context.reflection or _Reflection(
                        round_num=round_num,
                        diagnosis=f"Round skipped: {exc}",
                        failure_patterns=[],
                        proposed_changes=[],
                        priority="high",
                    )
                    round_result = RoundResult(
                        round_num=round_num,
                        train_scores=fallback_scores,
                        val_scores=context.val_scores,
                        reflection=fallback_reflection,
                        mutation=None,
                        applied=False,
                        error=str(exc),
                    )
                    context.history.append(round_result)
                    continue
                except Exception:
                    logger.exception("Fatal unhandled error in round %d", round_num)
                    raise

                # If a mutation was proposed, apply it, run validation, then decide.
                if context.proposed_mutation is not None:
                    saved_config = dict(context.current_config)
                    proposed_config = dict(context.proposed_mutation.proposed_config)

                    active_target.apply_config(proposed_config)
                    context.current_config = proposed_config

                    logger.info(
                        "Round %d: applying proposed mutation for validation",
                        round_num,
                    )

                    try:
                        context = await self._validation_stage.execute(context)
                    except (TrainingAborted, GuardrailTriggered) as exc:
                        stop_reason = f"aborted during validation: {exc}"
                        logger.warning("Training aborted in round %d: %s", round_num, exc)
                        active_target.apply_config(saved_config)
                        context.current_config = saved_config
                        break
                    except (LLMError, ConnectorError) as exc:
                        logger.error(
                            "Round %d: validation failed with recoverable error: "
                            "[%s] %s. Rolling back mutation.",
                            round_num,
                            type(exc).__name__,
                            exc,
                        )
                        active_target.apply_config(saved_config)
                        context.current_config = saved_config
                    except Exception:
                        logger.exception("Fatal unhandled error in validation round %d", round_num)
                        active_target.apply_config(saved_config)
                        context.current_config = saved_config
                        raise

                    # If validation failed (score regressed), rollback.
                    val_mean = context.val_scores.mean_score if context.val_scores else 0.0
                    regression = context.best_val_score - val_mean
                    if (
                        regression > self._config.regression_tolerance
                        and context.best_val_score > 0.0
                    ):
                        logger.warning(
                            "Round %d: validation score %.3f regressed from best %.3f "
                            "(tolerance=%.3f). Rolling back mutation.",
                            round_num,
                            val_mean,
                            context.best_val_score,
                            self._config.regression_tolerance,
                        )
                        active_target.apply_config(saved_config)
                        context.current_config = saved_config
                    else:
                        logger.info(
                            "Round %d: validation passed (score=%.3f), keeping mutation.",
                            round_num,
                            val_mean,
                        )
                else:
                    # No mutation proposed; still run validation with current config.
                    try:
                        context = await self._validation_stage.execute(context)
                    except (TrainingAborted, GuardrailTriggered) as exc:
                        stop_reason = f"aborted during validation: {exc}"
                        logger.warning("Training aborted in round %d: %s", round_num, exc)
                        break
                    except (LLMError, ConnectorError) as exc:
                        logger.error(
                            "Round %d: validation failed with recoverable error: "
                            "[%s] %s. Skipping round.",
                            round_num,
                            type(exc).__name__,
                            exc,
                        )
                    except Exception:
                        logger.exception("Fatal unhandled error in validation round %d", round_num)
                        raise

                # Build RoundResult for this round.
                if context.train_scores is None:
                    raise TrainingAborted(
                        f"round {round_num} missing train_scores after evaluation stage"
                    )
                if context.reflection is None:
                    raise TrainingAborted(
                        f"round {round_num} missing reflection after reflection stage"
                    )
                round_result = RoundResult(
                    round_num=round_num,
                    train_scores=context.train_scores,
                    val_scores=context.val_scores,
                    reflection=context.reflection,
                    mutation=context.proposed_mutation,
                    applied=context.proposed_mutation is not None,
                )
                if self._pareto_tracker is not None and self._objective_set is not None:
                    objective_scores = self._build_objective_scores(context)
                    on_frontier = self._pareto_tracker.add(
                        round_num=round_num,
                        config=context.current_config,
                        scores=objective_scores,
                    )
                    round_result.pareto_frontier_size = len(self._pareto_tracker.frontier)
                    logger.info(
                        "Round %d: multi-objective tracked (frontier=%d, "
                        "on_frontier=%s, scores=%s)",
                        round_num,
                        round_result.pareto_frontier_size,
                        on_frontier,
                        {k: round(v, 4) for k, v in objective_scores.items()},
                    )
                # Trim history to prevent unbounded memory growth.
                context.history.append(round_result)
                if len(context.history) > context.max_history_size:
                    context.history = context.history[-context.max_history_size :]

                # Track best config / score.
                val_mean = context.val_scores.mean_score if context.val_scores else 0.0
                if val_mean > context.best_val_score:
                    context.best_val_score = val_mean
                    context.best_config = dict(context.current_config)
                    logger.info("Round %d: new best val score %.3f", round_num, val_mean)
                else:
                    logger.info(
                        "Round %d: val score %.3f did not beat best %.3f",
                        round_num,
                        val_mean,
                        context.best_val_score,
                    )

                # Check guardrails.
                should_halt, halt_reason = self._check_guardrails(context)
                if should_halt:
                    stop_reason = halt_reason
                    logger.info("Guardrail triggered stop: %s", halt_reason)
                    break

                # Check scheduler convergence.
                if self._pareto_tracker is not None:
                    converged, conv_reason = self._scheduler.should_stop_multi_objective(
                        context,
                        self._pareto_tracker,
                    )
                else:
                    converged, conv_reason = self._scheduler.should_stop(context)
                if converged:
                    stop_reason = conv_reason
                    logger.info("Scheduler stop: %s", conv_reason)
                    break

                # Fire callbacks for UI / monitoring.
                self._fire_callbacks(round_result, context)

        finally:
            # Log sandbox session state on any exit path.
            if self._sandbox is not None and sandbox_session_id is not None:
                logger.info(
                    "Training loop finished (session %s). Sandbox state preserved "
                    "for promote/discard.",
                    sandbox_session_id,
                )

        elapsed = time.monotonic() - start_time
        return self._finalize(context, stop_reason, elapsed, sandbox_session_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parallel_stages(
        parallel_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct parallel stage instances from *parallel_config*.

        Parameters
        ----------
        parallel_config:
            Dictionary with keys ``max_concurrency`` (int), ``judges``
            (list of :class:`JudgeConfig`), and ``strategy`` (str or
            :class:`EnsembleStrategy`).

        Returns
        -------
        dict
            ``{"pre_validation": [...stages], "validation": stage}``
        """
        max_concurrency: int = parallel_config.get("max_concurrency", 10)
        judges: list[JudgeConfig] = parallel_config.get("judges", [])
        raw_strategy = parallel_config.get("strategy", "mean")

        if isinstance(raw_strategy, str):
            strategy = EnsembleStrategy(raw_strategy)
        else:
            strategy = raw_strategy

        # If no judges were supplied, fall back to the default sequential
        # evaluation stage but still use parallel execution.
        if judges:
            eval_stage: Stage = ParallelEvaluationStage(judges=judges, strategy=strategy)
        else:
            eval_stage = EvaluationStage()

        pre_validation: list[Stage] = [
            DatasetGenStage(),
            ParallelExecutionStage(max_concurrency=max_concurrency),
            eval_stage,
            ReflectionStage(),
            MutationStage(),
        ]

        validation: Stage = ParallelValidationStage(
            judges=judges if judges else None,
            max_concurrency=max_concurrency,
        )

        return {"pre_validation": pre_validation, "validation": validation}

    def _check_guardrails(self, context: LoopContext) -> tuple[bool, str]:
        """Run all guardrails and return (should_stop, reason)."""
        for guardrail in self._guardrails:
            try:
                verdict = guardrail.check(context.history)
                if verdict.should_stop:
                    return True, (f"Guardrail '{verdict.guardrail_name}': {verdict.reason}")
                if not verdict.passed:
                    logger.warning(
                        "Guardrail '%s' flagged (non-fatal): %s",
                        verdict.guardrail_name,
                        verdict.reason,
                    )
            except Exception:
                logger.exception("Guardrail '%s' raised an exception", guardrail.name)
        return False, ""

    def _init_multi_objective(self, config: TrainingConfig) -> None:
        """Initialise multi-objective tracking from training config."""
        raw_objectives = config.objectives or []
        if not raw_objectives:
            return

        objectives: list[Objective] = []
        for idx, item in enumerate(raw_objectives):
            if not isinstance(item, dict):
                logger.warning(
                    "Ignoring invalid objective at index %d (expected dict, got %s)",
                    idx,
                    type(item).__name__,
                )
                continue
            try:
                objectives.append(Objective.model_validate(item))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Ignoring invalid objective at index %d: %s",
                    idx,
                    exc,
                )

        if not objectives:
            logger.warning("Multi-objective mode requested but no valid objectives were parsed.")
            return

        self._objective_set = ObjectiveSet(objectives=objectives)
        self._pareto_tracker = ParetoTracker(self._objective_set)
        logger.info(
            "Multi-objective mode enabled with objectives=%s",
            [obj.name for obj in objectives],
        )

    def _build_objective_scores(self, context: LoopContext) -> dict[str, float]:
        """Build per-objective scores for Pareto tracking in the current round."""
        if self._objective_set is None:
            return {}

        val_mean = context.val_scores.mean_score if context.val_scores else 0.0
        from_card = context.val_scores.objective_scores if context.val_scores else None

        scores: dict[str, float] = {}
        for obj in self._objective_set.objectives:
            raw = val_mean
            if from_card and obj.name in from_card:
                raw = from_card[obj.name]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = val_mean
            scores[obj.name] = max(0.0, min(1.0, value))
        return scores

    def _finalize(
        self,
        context: LoopContext,
        stop_reason: str,
        elapsed_s: float,
        sandbox_session_id: str | None = None,
    ) -> TrainingResult:
        """Build the final TrainingResult from accumulated context."""
        best_round = 0
        best_score = 0.0
        for rr in context.history:
            vs = rr.val_scores.mean_score if rr.val_scores else 0.0
            if vs > best_score:
                best_score = vs
                best_round = rr.round_num

        frontier_payload: list[dict[str, Any]] | None = None
        if self._pareto_tracker is not None:
            frontier_payload = [
                {
                    "round_num": p.round_num,
                    "config": p.config,
                    "scores": p.scores,
                    "weighted_score": p.weighted_score,
                }
                for p in self._pareto_tracker.frontier
            ]

        return TrainingResult(
            rounds=context.history,
            best_round=best_round,
            best_val_score=best_score,
            final_config=context.best_config,
            stop_reason=stop_reason,
            pareto_frontier=frontier_payload,
            total_duration_s=round(elapsed_s, 2),
            sandbox_session_id=sandbox_session_id,
        )

    def _fire_callbacks(self, round_result: RoundResult, context: LoopContext) -> None:
        """Invoke all registered callbacks as ``cb(round_result, context)``."""
        for cb in self._callbacks:
            try:
                cb(round_result, context)
            except Exception:
                logger.exception("Callback %s raised an exception", cb)
