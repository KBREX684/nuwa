"""Nuwa benchmark suite APIs."""

from nuwa.benchmarks.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkRunResult,
    BenchmarkSuite,
)
from nuwa.benchmarks.registry import get_benchmark, list_benchmarks, register_benchmark
from nuwa.benchmarks.runner import run_benchmark

__all__ = [
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkRunResult",
    "BenchmarkSuite",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "run_benchmark",
]
