"""Benchmark suite registry."""

from __future__ import annotations

from nuwa.benchmarks.models import BenchmarkCase, BenchmarkSuite

_REGISTRY: dict[str, BenchmarkSuite] = {
    "customer_support_cn": BenchmarkSuite(
        name="customer_support_cn",
        description="中文客服场景基础能力基准集。",
        cases=[
            BenchmarkCase(
                id="cs-001",
                input_text="请帮我查询订单退款进度。",
                expected_keywords=["退款", "进度"],
            ),
            BenchmarkCase(
                id="cs-002",
                input_text="我想修改收货地址，现在还能改吗？",
                expected_keywords=["地址", "修改"],
            ),
            BenchmarkCase(
                id="cs-003",
                input_text="商品有质量问题，怎么处理？",
                expected_keywords=["质量", "处理"],
            ),
        ],
    ),
    "reasoning_basic_en": BenchmarkSuite(
        name="reasoning_basic_en",
        description="English reasoning and explanation baseline.",
        cases=[
            BenchmarkCase(
                id="rs-001",
                input_text="Explain why unit tests matter for backend services.",
                expected_keywords=["test", "reliability"],
            ),
            BenchmarkCase(
                id="rs-002",
                input_text="How would you reduce latency in an API service?",
                expected_keywords=["latency", "cache"],
            ),
            BenchmarkCase(
                id="rs-003",
                input_text="Describe a safe rollback strategy for deployments.",
                expected_keywords=["rollback", "safe"],
            ),
        ],
    ),
}


def register_benchmark(name: str, suite: BenchmarkSuite) -> None:
    """Register or overwrite a benchmark suite."""
    _REGISTRY[name] = suite


def get_benchmark(name: str) -> BenchmarkSuite:
    """Get benchmark suite by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark suite: {name}")
    return _REGISTRY[name]


def list_benchmarks() -> list[str]:
    """List available benchmark suite names."""
    return sorted(_REGISTRY)
