"""Execution stage -- runs the target agent on all training samples."""

from __future__ import annotations

import asyncio
import logging
import time

from nuwa.core.types import AgentResponse, EvalSample, LoopContext, ScoredResult

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = 5
_INVOKE_TIMEOUT_S = 120.0


class ExecutionStage:
    """Run the target agent on every sample in *train_set* concurrently."""

    @property
    def name(self) -> str:
        return "execution"

    async def execute(self, context: LoopContext) -> LoopContext:
        target = context.target_ref
        samples = context.train_set

        if not samples:
            logger.warning("Round %d: train_set is empty, skipping execution.", context.round_num)
            context.train_results = []
            return context

        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _run_one(sample: EvalSample) -> tuple[EvalSample, AgentResponse]:
            async with sem:
                start = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        target.invoke(sample.input_text, config=context.current_config or None),
                        timeout=_INVOKE_TIMEOUT_S,
                    )
                    return sample, response
                except TimeoutError:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    logger.warning(
                        "Sample %s timed out after %.0f ms", sample.id, elapsed_ms
                    )
                    return sample, AgentResponse(
                        output_text="[ERROR: agent timed out]",
                        latency_ms=elapsed_ms,
                        raw_metadata={"error": "timeout"},
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    logger.warning(
                        "Sample %s failed with %s: %s",
                        sample.id,
                        type(exc).__name__,
                        exc,
                    )
                    return sample, AgentResponse(
                        output_text=f"[ERROR: {type(exc).__name__}: {exc}]",
                        latency_ms=elapsed_ms,
                        raw_metadata={"error": str(exc)},
                    )

        logger.info(
            "Round %d: executing %d samples (concurrency=%d)",
            context.round_num,
            len(samples),
            _MAX_CONCURRENCY,
        )

        results: list[tuple[EvalSample, AgentResponse]] = await asyncio.gather(
            *(_run_one(s) for s in samples)
        )

        # Store as ScoredResult placeholders (score=0, placeholder reasoning) so
        # downstream stages can work with a uniform type.  The EvaluationStage
        # will overwrite scores.
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
            "Round %d: execution complete -- %d results collected",
            context.round_num,
            len(context.train_results),
        )
        return context
