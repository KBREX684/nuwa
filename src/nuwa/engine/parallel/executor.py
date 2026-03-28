"""High-throughput parallel executor for running the target agent.

Unlike the default :class:`~nuwa.engine.stages.execution.ExecutionStage`
(capped at 5 concurrent invocations), this executor supports configurable
concurrency, automatic retry with exponential back-off, per-sample timeouts,
and optional progress callbacks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from nuwa.core.protocols import TargetAgent
from nuwa.core.types import AgentResponse, EvalSample

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY_LIMIT = 50


class ParallelExecutor:
    """High-throughput parallel executor for running the target agent.

    Unlike the default ExecutionStage (max 5 concurrent), this supports:

    * Configurable concurrency (default 10, up to 50).
    * Batch execution with progress tracking via an optional callback.
    * Automatic retry with exponential back-off for transient failures.
    * Per-sample timeout with graceful degradation (returns an error
      ``AgentResponse`` rather than propagating the exception).

    Parameters
    ----------
    max_concurrency:
        Maximum number of agent invocations running simultaneously.
    timeout_per_sample:
        Per-invocation wall-clock timeout in seconds.
    max_retries:
        How many times to retry a failed invocation before giving up.
    retry_delay:
        Base delay (in seconds) between retries; doubled each attempt.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        timeout_per_sample: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        if max_concurrency < 1 or max_concurrency > _MAX_CONCURRENCY_LIMIT:
            raise ValueError(
                f"max_concurrency must be in [1, {_MAX_CONCURRENCY_LIMIT}], got {max_concurrency}"
            )
        self._max_concurrency = max_concurrency
        self._timeout = timeout_per_sample
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_batch(
        self,
        target: TargetAgent,
        samples: list[EvalSample],
        config: dict[str, Any] | None = None,
        on_progress: Callable[[int, int], object] | None = None,
    ) -> list[tuple[EvalSample, AgentResponse]]:
        """Run all *samples* through the target agent in parallel.

        Parameters
        ----------
        target:
            Object implementing the ``TargetAgent`` protocol.
        samples:
            Evaluation samples to execute.
        config:
            Optional configuration dict forwarded to each ``target.invoke`` call.
        on_progress:
            Optional callback invoked as ``on_progress(completed, total)``
            after each sample finishes.

        Returns
        -------
        list[tuple[EvalSample, AgentResponse]]
            ``(sample, response)`` pairs in the **same order** as *samples*.
        """
        if not samples:
            return []

        total = len(samples)
        sem = asyncio.Semaphore(self._max_concurrency)
        completed = 0
        lock = asyncio.Lock()

        async def _run_one(sample: EvalSample) -> tuple[EvalSample, AgentResponse]:
            nonlocal completed
            async with sem:
                response = await self._invoke_with_retry(target, sample, config)
                async with lock:
                    completed += 1
                    if on_progress is not None:
                        try:
                            on_progress(completed, total)
                        except Exception as cb_exc:  # noqa: BLE001
                            logger.warning(
                                "Progress callback raised %s: %s; ignoring.",
                                type(cb_exc).__name__,
                                cb_exc,
                            )
                return sample, response

        logger.info(
            "ParallelExecutor: launching %d samples (concurrency=%d, timeout=%.1fs, retries=%d)",
            total,
            self._max_concurrency,
            self._timeout,
            self._max_retries,
        )

        results: list[tuple[EvalSample, AgentResponse]] = list(
            await asyncio.gather(*(_run_one(s) for s in samples))
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _invoke_with_retry(
        self,
        target: TargetAgent,
        sample: EvalSample,
        config: dict[str, Any] | None,
    ) -> AgentResponse:
        """Invoke the target agent with retry + exponential back-off."""
        delay = self._retry_delay
        last_exc: Exception | None = None
        elapsed_ms = 0.0

        for attempt in range(1, self._max_retries + 2):  # +2: first try + retries
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    target.invoke(sample.input_text, config=config),
                    timeout=self._timeout,
                )
                return response
            except TimeoutError:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                last_exc = TimeoutError(f"Timed out after {elapsed_ms:.0f}ms")
                logger.warning(
                    "Sample %s attempt %d/%d timed out after %.0fms",
                    sample.id,
                    attempt,
                    self._max_retries + 1,
                    elapsed_ms,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                last_exc = exc
                logger.warning(
                    "Sample %s attempt %d/%d failed (%s): %s",
                    sample.id,
                    attempt,
                    self._max_retries + 1,
                    type(exc).__name__,
                    exc,
                )

            # If we still have retries left, back off before the next attempt.
            if attempt <= self._max_retries:
                await asyncio.sleep(delay)
                delay *= 2  # exponential back-off

        # All attempts exhausted -- return a degraded response.
        error_msg = str(last_exc) if last_exc else "unknown error"
        return AgentResponse(
            output_text=f"[ERROR: {error_msg} after {self._max_retries + 1} attempts]",
            latency_ms=elapsed_ms,
            raw_metadata={"error": error_msg, "attempts": self._max_retries + 1},
        )
