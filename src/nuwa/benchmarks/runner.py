"""Benchmark runner."""

from __future__ import annotations

import asyncio
from typing import Any

from nuwa.benchmarks.models import BenchmarkCase, BenchmarkCaseResult, BenchmarkRunResult
from nuwa.benchmarks.registry import get_benchmark
from nuwa.core.protocols import TargetAgent


def _score_case(case: BenchmarkCase, output_text: str) -> tuple[float, dict[str, float | str]]:
    text = output_text.lower()

    keyword_score = 1.0
    if case.expected_keywords:
        found = sum(1 for k in case.expected_keywords if k.lower() in text)
        keyword_score = found / len(case.expected_keywords)

    forbidden_penalty = 1.0
    if case.forbidden_keywords:
        hit = sum(1 for k in case.forbidden_keywords if k.lower() in text)
        forbidden_penalty = max(0.0, 1.0 - hit / len(case.forbidden_keywords))

    length_score = 1.0 if len(output_text) >= case.min_output_chars else 0.0

    score = 0.6 * keyword_score + 0.2 * forbidden_penalty + 0.2 * length_score
    score = max(0.0, min(1.0, score))
    details: dict[str, float | str] = {
        "keyword_score": round(keyword_score, 4),
        "forbidden_penalty": round(forbidden_penalty, 4),
        "length_score": round(length_score, 4),
    }
    return score, details


async def run_benchmark(
    target: TargetAgent,
    suite_name: str,
    *,
    config: dict[str, Any] | None = None,
    max_concurrency: int = 5,
) -> BenchmarkRunResult:
    """Run one benchmark suite against a target agent."""
    suite = get_benchmark(suite_name)
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_case(case: BenchmarkCase) -> BenchmarkCaseResult:
        async with sem:
            response = await target.invoke(case.input_text, config=config)
            score, details = _score_case(case, response.output_text)
            return BenchmarkCaseResult(
                case_id=case.id,
                score=score,
                output_text=response.output_text,
                latency_ms=response.latency_ms,
                details=details,
            )

    results = await asyncio.gather(*(_run_case(case) for case in suite.cases))
    return BenchmarkRunResult(suite_name=suite.name, cases=list(results))
