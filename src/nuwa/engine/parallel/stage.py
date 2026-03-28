"""Drop-in parallel replacements for the default sequential pipeline stages.

Each class implements the :class:`~nuwa.core.protocols.Stage` protocol and
can be used interchangeably with its sequential counterpart.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jinja2 import Template

from nuwa.core.exceptions import LLMError
from nuwa.core.types import (
    AgentResponse,
    EvalSample,
    LoopContext,
    ScoreCard,
    ScoredResult,
)
from nuwa.engine.parallel.evaluator import (
    EnsembleStrategy,
    JudgeConfig,
    ParallelEvaluator,
)
from nuwa.engine.parallel.executor import ParallelExecutor
from nuwa.engine.stages.immutable_scorer import ImmutableScorer
from nuwa.llm.prompts import OUTPUT_SCORING
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE = Template(OUTPUT_SCORING)
_PASS_THRESHOLD = 0.7
_INVOKE_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# ParallelExecutionStage
# ---------------------------------------------------------------------------


class ParallelExecutionStage:
    """Drop-in replacement for
    :class:`~nuwa.engine.stages.execution.ExecutionStage` using
    :class:`ParallelExecutor`.

    Implements the :class:`~nuwa.core.protocols.Stage` protocol.

    Parameters
    ----------
    max_concurrency:
        Maximum number of concurrent agent invocations.
    max_retries:
        Per-sample retry count for transient failures.
    timeout_per_sample:
        Per-invocation timeout in seconds.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        max_retries: int = 2,
        timeout_per_sample: float = 120.0,
    ) -> None:
        self._executor = ParallelExecutor(
            max_concurrency=max_concurrency,
            timeout_per_sample=timeout_per_sample,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:  # noqa: D102
        return "parallel_execution"

    async def execute(self, context: LoopContext) -> LoopContext:
        """Run the target agent on every sample in *train_set* concurrently."""
        target = context.target_ref
        samples = context.train_set

        if not samples:
            logger.warning(
                "Round %d: train_set is empty, skipping parallel execution.",
                context.round_num,
            )
            context.train_results = []
            return context

        logger.info(
            "Round %d: parallel execution of %d samples",
            context.round_num,
            len(samples),
        )

        results = await self._executor.execute_batch(
            target=target,
            samples=samples,
            config=context.current_config or None,
        )

        # Store as ScoredResult placeholders (score=0, placeholder reasoning) so
        # downstream stages can work with a uniform type.
        context.train_results = [
            ScoredResult(
                sample=sample,
                response=response,
                score=0.0,
                reasoning="(pending evaluation)",
            )
            for sample, response in results
        ]

        logger.info(
            "Round %d: parallel execution complete -- %d results collected",
            context.round_num,
            len(context.train_results),
        )
        return context


# ---------------------------------------------------------------------------
# ParallelEvaluationStage
# ---------------------------------------------------------------------------


class ParallelEvaluationStage:
    """Drop-in replacement for
    :class:`~nuwa.engine.stages.evaluation.EvaluationStage` using
    :class:`ParallelEvaluator`.

    Implements the :class:`~nuwa.core.protocols.Stage` protocol.

    Parameters
    ----------
    judges:
        List of :class:`JudgeConfig` instances describing the ensemble.
    strategy:
        Ensemble aggregation strategy.
    immutable_scorer:
        Optional deterministic scorer override.
    llm_weight:
        Weight for the LLM ensemble score.
    immutable_weight:
        Weight for the immutable (deterministic) score.
    """

    def __init__(
        self,
        judges: list[JudgeConfig],
        strategy: EnsembleStrategy = EnsembleStrategy.MEAN,
        immutable_scorer: ImmutableScorer | None = None,
        llm_weight: float = 0.7,
        immutable_weight: float = 0.3,
    ) -> None:
        self._evaluator = ParallelEvaluator(
            judges=judges,
            strategy=strategy,
            immutable_scorer=immutable_scorer,
            immutable_weight=immutable_weight,
            llm_weight=llm_weight,
        )

    @property
    def name(self) -> str:  # noqa: D102
        return "parallel_evaluation"

    async def execute(self, context: LoopContext) -> LoopContext:
        """Score every result using the multi-judge ensemble."""
        if not context.train_results:
            logger.warning(
                "Round %d: no train_results to evaluate.", context.round_num
            )
            context.train_scores = ScoreCard(results=[], failure_analysis="")
            return context

        # Unpack ScoredResult placeholders into (sample, response) pairs
        # for the evaluator.
        pairs: list[tuple[EvalSample, AgentResponse]] = [
            (sr.sample, sr.response) for sr in context.train_results
        ]

        logger.info(
            "Round %d: parallel evaluation of %d results",
            context.round_num,
            len(pairs),
        )

        card = await self._evaluator.evaluate_batch(pairs)
        context.train_scores = card
        context.train_results = card.results  # update with real scores

        logger.info(
            "Round %d: train mean_score=%.3f  pass_rate=%.2f",
            context.round_num,
            card.mean_score,
            card.pass_rate,
        )
        return context


# ---------------------------------------------------------------------------
# ParallelValidationStage
# ---------------------------------------------------------------------------


class ParallelValidationStage:
    """Drop-in replacement for
    :class:`~nuwa.engine.stages.validation.ValidationStage` using the
    parallel execution and evaluation infrastructure.

    Implements the :class:`~nuwa.core.protocols.Stage` protocol.

    When *judges* is provided, the multi-judge ensemble is used for scoring
    the validation set.  Otherwise, a single judge backed by the context's
    ``backend_ref`` is used (mirroring the default ``ValidationStage``
    behaviour but with higher concurrency).

    Parameters
    ----------
    judges:
        Optional list of judge configs for ensemble scoring.
    max_concurrency:
        Maximum number of concurrent agent invocations for running the
        validation set.
    """

    def __init__(
        self,
        judges: list[JudgeConfig] | None = None,
        max_concurrency: int = 10,
    ) -> None:
        self._judges = judges
        self._executor = ParallelExecutor(
            max_concurrency=max_concurrency,
            timeout_per_sample=_INVOKE_TIMEOUT_S,
            max_retries=1,
        )

    @property
    def name(self) -> str:  # noqa: D102
        return "parallel_validation"

    async def execute(self, context: LoopContext) -> LoopContext:
        """Execute and score the held-out *val_set* in parallel."""
        target = context.target_ref
        val_samples = context.val_set

        if not val_samples:
            logger.warning(
                "Round %d: val_set is empty, skipping validation.",
                context.round_num,
            )
            context.val_scores = ScoreCard(
                results=[], failure_analysis="no validation samples"
            )
            return context

        logger.info(
            "Round %d: parallel validation on %d held-out samples",
            context.round_num,
            len(val_samples),
        )

        # --- Execute agent on val_set ------------------------------------
        exec_results = await self._executor.execute_batch(
            target=target,
            samples=val_samples,
            config=context.current_config or None,
        )

        # --- Score -------------------------------------------------------
        if self._judges:
            # Multi-judge ensemble scoring.
            evaluator = ParallelEvaluator(judges=self._judges)
            card = await evaluator.evaluate_batch(exec_results)
        else:
            # Fallback: single-judge scoring using context backend.
            card = await self._score_with_context_backend(
                context, exec_results
            )

        context.val_scores = card

        logger.info(
            "Round %d: val mean_score=%.3f  pass_rate=%.2f",
            context.round_num,
            card.mean_score,
            card.pass_rate,
        )
        return context

    # ------------------------------------------------------------------
    # Fallback single-judge scoring
    # ------------------------------------------------------------------

    @staticmethod
    async def _score_with_context_backend(
        context: LoopContext,
        results: list[tuple[EvalSample, AgentResponse]],
    ) -> ScoreCard:
        """Score validation results using the context's default backend.

        This mirrors the original ``ValidationStage`` logic but runs all
        scoring calls concurrently with a semaphore.
        """
        backend = context.backend_ref
        sem = asyncio.Semaphore(10)

        async def _score_one(
            sample: EvalSample, response: AgentResponse
        ) -> ScoredResult:
            async with sem:
                prompt = _SCORING_TEMPLATE.render(
                    input_text=sample.input_text,
                    expected_behavior=sample.expected_behavior,
                    actual_output=response.output_text,
                    scoring_rubric=None,
                )
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": "You are an objective evaluation judge.",
                    },
                    {"role": "user", "content": prompt},
                ]

                try:
                    raw = await backend.complete(messages, temperature=0.1)
                    data = parse_json_response(raw)
                    score = float(data.get("score", 0.0))  # type: ignore[union-attr]
                    score = max(0.0, min(1.0, score))
                    reasoning_en = data.get("reasoning_en", "")  # type: ignore[union-attr]
                    reasoning_zh = data.get("reasoning_zh", "")  # type: ignore[union-attr]
                    reasoning = str(
                        reasoning_en or reasoning_zh or "(no reasoning)"
                    )
                except (LLMError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Validation scoring failed: %s", exc)
                    score = 0.0
                    reasoning = f"Scoring error: {exc}"

                return ScoredResult(
                    sample=sample,
                    response=response,
                    score=score,
                    reasoning=reasoning,
                )

        scored: list[ScoredResult] = list(
            await asyncio.gather(
                *(_score_one(sample, resp) for sample, resp in results)
            )
        )

        failures = [s for s in scored if s.score < _PASS_THRESHOLD]
        failure_analysis = ""
        if failures:
            lines = [
                f"- [{f.score:.2f}] input={f.sample.input_text[:80]!r}"
                for f in failures
            ]
            failure_analysis = (
                f"Validation: {len(failures)}/{len(scored)} below "
                f"{_PASS_THRESHOLD}:\n" + "\n".join(lines)
            )

        return ScoreCard(results=scored, failure_analysis=failure_analysis)
