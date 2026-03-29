"""Unit tests for benchmark runner scoring and execution."""

from __future__ import annotations

from uuid import uuid4

import pytest

from nuwa.benchmarks.models import BenchmarkCase, BenchmarkSuite
from nuwa.benchmarks.registry import register_benchmark
from nuwa.benchmarks.runner import run_benchmark
from nuwa.core.types import AgentResponse


class _MockBenchmarkTarget:
    def __init__(self) -> None:
        self.received_configs: list[dict[str, object] | None] = []

    async def invoke(
        self,
        input_text: str,
        config: dict[str, object] | None = None,
    ) -> AgentResponse:
        self.received_configs.append(config)
        if "first" in input_text:
            return AgentResponse(output_text="foo bar answer", latency_ms=10.0)
        return AgentResponse(output_text="forbidden text", latency_ms=20.0)

    def get_current_config(self) -> dict[str, object]:
        return {}

    def apply_config(self, config: dict[str, object]) -> None:
        _ = config


@pytest.mark.asyncio
async def test_run_benchmark_scores_and_pass_rate() -> None:
    suffix = uuid4().hex[:8]
    suite_name = f"suite_{suffix}"

    register_benchmark(
        suite_name,
        BenchmarkSuite(
            name=suite_name,
            description="test suite",
            cases=[
                BenchmarkCase(
                    id="first",
                    input_text="first case",
                    expected_keywords=["foo", "bar"],
                ),
                BenchmarkCase(
                    id="second",
                    input_text="second case",
                    expected_keywords=["baz"],
                    forbidden_keywords=["forbidden"],
                ),
            ],
        ),
    )

    target = _MockBenchmarkTarget()
    result = await run_benchmark(
        target,
        suite_name=suite_name,
        config={"mode": "benchmark"},
        max_concurrency=2,
    )

    assert result.suite_name == suite_name
    assert len(result.cases) == 2
    assert result.cases[0].score == pytest.approx(1.0)
    assert result.cases[1].score == pytest.approx(0.2)
    assert result.mean_score == pytest.approx(0.6)
    assert result.pass_rate == pytest.approx(0.5)
    assert target.received_configs == [{"mode": "benchmark"}, {"mode": "benchmark"}]
