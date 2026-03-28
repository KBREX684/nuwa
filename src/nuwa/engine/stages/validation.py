"""Validation stage -- runs held-out val_set to detect overfitting."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from jinja2 import Template

from nuwa.core.exceptions import LLMError
from nuwa.core.types import AgentResponse, LoopContext, ScoreCard, ScoredResult
from nuwa.llm.prompts import OUTPUT_SCORING
from nuwa.llm.response_parser import parse_json_response

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE = Template(OUTPUT_SCORING)

_MAX_CONCURRENCY = 5
_INVOKE_TIMEOUT_S = 120.0
_PASS_THRESHOLD = 0.7


class ValidationStage:
    """Execute and score the held-out val_set as an overfitting check."""

    @property
    def name(self) -> str:
        return "validation"

    async def execute(self, context: LoopContext) -> LoopContext:
        target = context.target_ref
        backend = context.backend_ref
        val_samples = context.val_set

        if not val_samples:
            logger.warning("Round %d: val_set is empty, skipping validation.", context.round_num)
            context.val_scores = ScoreCard(results=[], failure_analysis="no validation samples")
            return context

        # --- Execute agent on val_set ------------------------------------
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _run_and_score(
            sample_input: str,
            expected: str,
        ) -> tuple[AgentResponse, float, str, dict[str, float] | None]:
            async with sem:
                start = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        target.invoke(sample_input, config=context.current_config or None),
                        timeout=_INVOKE_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    response = AgentResponse(
                        output_text="[ERROR: agent timed out]",
                        latency_ms=elapsed,
                        raw_metadata={"error": "timeout"},
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = (time.perf_counter() - start) * 1000.0
                    response = AgentResponse(
                        output_text=f"[ERROR: {type(exc).__name__}: {exc}]",
                        latency_ms=elapsed,
                        raw_metadata={"error": str(exc)},
                    )

                # Score the response.
                score, reasoning, axis_scores = await self._score_one(
                    backend, sample_input, expected, response.output_text
                )
                return response, score, reasoning, axis_scores

        logger.info(
            "Round %d: validating on %d held-out samples",
            context.round_num,
            len(val_samples),
        )

        tasks = [
            _run_and_score(s.input_text, s.expected_behavior)
            for s in val_samples
        ]
        results = await asyncio.gather(*tasks)

        scored: list[ScoredResult] = []
        axis_rows: list[dict[str, float]] = []
        for sample, (response, score, reasoning, axis_scores) in zip(val_samples, results):
            scored.append(
                ScoredResult(
                    sample=sample,
                    response=response,
                    score=score,
                    reasoning=reasoning,
                )
            )
            if axis_scores:
                axis_rows.append(axis_scores)

        objective_scores: dict[str, float] | None = None
        if axis_rows:
            keys: set[str] = set()
            for row in axis_rows:
                keys.update(row.keys())
            objective_scores = {}
            for key in sorted(keys):
                vals = [row[key] for row in axis_rows if key in row]
                if vals:
                    objective_scores[key] = sum(vals) / len(vals)

        failures = [s for s in scored if s.score < _PASS_THRESHOLD]
        failure_analysis = ""
        if failures:
            lines = [
                f"- [{f.score:.2f}] input={f.sample.input_text[:80]!r}"
                for f in failures
            ]
            failure_analysis = (
                f"Validation: {len(failures)}/{len(scored)} below {_PASS_THRESHOLD}:\n"
                + "\n".join(lines)
            )

        context.val_scores = ScoreCard(
            results=scored,
            failure_analysis=failure_analysis,
            objective_scores=objective_scores,
        )

        logger.info(
            "Round %d: val mean_score=%.3f  pass_rate=%.2f",
            context.round_num,
            context.val_scores.mean_score,
            context.val_scores.pass_rate,
        )
        return context

    # ------------------------------------------------------------------

    @staticmethod
    async def _score_one(
        backend: Any,
        input_text: str,
        expected_behavior: str,
        actual_output: str,
    ) -> tuple[float, str, dict[str, float] | None]:
        """Score a single validation triple.

        Returns:
            (score, reasoning, axis_scores)
        """
        prompt = _SCORING_TEMPLATE.render(
            input_text=input_text,
            expected_behavior=expected_behavior,
            actual_output=actual_output,
            scoring_rubric=None,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an objective evaluation judge."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await backend.complete(messages, temperature=0.1)
            data = parse_json_response(raw)
            score = float(data.get("score", 0.0))  # type: ignore[union-attr]
            score = max(0.0, min(1.0, score))
            reasoning_en = data.get("reasoning_en", "")  # type: ignore[union-attr]
            reasoning_zh = data.get("reasoning_zh", "")  # type: ignore[union-attr]
            axis_raw = data.get("axis_scores", {})  # type: ignore[union-attr]
            axis_scores: dict[str, float] | None = None
            if isinstance(axis_raw, dict):
                axis_scores = {}
                for k, v in axis_raw.items():
                    try:
                        axis_scores[str(k)] = max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        continue
                if not axis_scores:
                    axis_scores = None
            return score, str(reasoning_en or reasoning_zh or "(no reasoning)"), axis_scores
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Validation scoring failed: %s", exc)
            return 0.0, f"Scoring error: {exc}", None
