"""Data models for benchmark suites and run results."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class BenchmarkCase(BaseModel):
    """Single benchmark case."""

    id: str
    input_text: str
    expected_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    min_output_chars: int = 1


class BenchmarkSuite(BaseModel):
    """Named benchmark suite."""

    name: str
    description: str = ""
    cases: list[BenchmarkCase] = Field(default_factory=list)


class BenchmarkCaseResult(BaseModel):
    """Result of one benchmark case execution."""

    case_id: str
    score: float = Field(ge=0.0, le=1.0)
    output_text: str
    latency_ms: float = Field(ge=0.0)
    details: dict[str, float | str] = Field(default_factory=dict)


class BenchmarkRunResult(BaseModel):
    """Aggregate result for one benchmark suite run."""

    suite_name: str
    cases: list[BenchmarkCaseResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        passed = sum(1 for c in self.cases if c.score >= 0.7)
        return passed / len(self.cases)
