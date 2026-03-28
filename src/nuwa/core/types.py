"""Pydantic v2 data models for the Nuwa AI Trainer framework.

All domain objects used across the training loop -- configuration, evaluation
samples, scoring artifacts, reflections, mutations, round/training results,
guardrail verdicts, and the mutable loop context -- are defined here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Top-level knobs that govern a single training run."""

    training_direction: str = Field(
        ...,
        description="Natural-language description of the desired training goal.",
    )
    max_rounds: int = Field(
        default=10,
        ge=1,
        description="Maximum number of train-eval-reflect-mutate rounds.",
    )
    samples_per_round: int = Field(
        default=20,
        ge=1,
        description="Number of evaluation samples generated each round.",
    )
    train_val_split: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="Fraction of samples allocated to the training set.",
    )
    overfitting_threshold: float = Field(
        default=0.15,
        ge=0.0,
        description=(
            "Maximum allowed gap between train and validation scores "
            "before overfitting is flagged."
        ),
    )
    consistency_runs: int = Field(
        default=3,
        ge=1,
        description="Number of repeated evaluations used to measure consistency.",
    )
    consistency_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of consistent results across repeated runs.",
    )
    regression_tolerance: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Allowed score regression between rounds before a mutation is reverted."
        ),
    )
    objectives: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional list of objective dicts for multi-objective optimisation. "
            "Each dict should contain at least a 'name' key plus optional "
            "'weight', 'direction', 'target', and 'description' fields."
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation primitives
# ---------------------------------------------------------------------------


class EvalSample(BaseModel):
    """A single evaluation prompt with its expected behaviour."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this sample.",
    )
    input_text: str = Field(
        ...,
        description="The prompt or input sent to the target agent.",
    )
    expected_behavior: str = Field(
        ...,
        description="Human-readable description of the correct/desired response.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Subjective difficulty bucket.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for slicing analysis.",
    )


class AgentResponse(BaseModel):
    """Raw output returned by the target agent for a single sample."""

    output_text: str = Field(
        ...,
        description="The text produced by the agent.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock latency of the agent call in milliseconds.",
    )
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata returned alongside the response.",
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_PASS_THRESHOLD: float = 0.7


class ScoredResult(BaseModel):
    """An evaluation sample paired with its agent response and LLM-assigned score."""

    sample: EvalSample
    response: AgentResponse
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised quality score assigned by the judge LLM.",
    )
    reasoning: str = Field(
        ...,
        description="Free-text explanation of why this score was given.",
    )


class ScoreCard(BaseModel):
    """Aggregate view over a batch of scored results."""

    results: list[ScoredResult] = Field(
        default_factory=list,
        description="Individual scored results that make up this card.",
    )
    failure_analysis: str = Field(
        default="",
        description="LLM-generated summary of common failure modes.",
    )
    objective_scores: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional per-objective score breakdown when multi-objective "
            "optimisation is enabled."
        ),
    )

    @computed_field
    @property
    def mean_score(self) -> float:
        """Arithmetic mean of all individual scores (0.0 when empty)."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """Fraction of results whose score meets or exceeds the pass threshold."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.score >= _PASS_THRESHOLD)
        return passed / len(self.results)


# ---------------------------------------------------------------------------
# Reflection & mutation
# ---------------------------------------------------------------------------


class Reflection(BaseModel):
    """Structured self-diagnosis produced after scoring a round."""

    round_num: int = Field(..., ge=0, description="The round this reflection covers.")
    diagnosis: str = Field(
        ...,
        description="High-level summary of what went wrong or right.",
    )
    failure_patterns: list[str] = Field(
        default_factory=list,
        description="Recurring failure modes identified by the reflector.",
    )
    proposed_changes: list[str] = Field(
        default_factory=list,
        description="Concrete suggestions for the next mutation.",
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Urgency level guiding how aggressively to mutate.",
    )


class Mutation(BaseModel):
    """A proposed configuration change derived from a reflection."""

    description: str = Field(
        ...,
        description="Human-readable summary of what is being changed.",
    )
    original_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of the config before the mutation.",
    )
    proposed_config: dict[str, Any] = Field(
        default_factory=dict,
        description="The new config to apply.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this change should improve performance.",
    )


# ---------------------------------------------------------------------------
# Round / training results
# ---------------------------------------------------------------------------


class RoundResult(BaseModel):
    """Complete record of a single train-eval-reflect-mutate round."""

    round_num: int = Field(..., ge=0)
    train_scores: ScoreCard
    val_scores: ScoreCard | None = Field(
        default=None,
        description="Validation scores; may be absent if validation was skipped.",
    )
    reflection: Reflection
    mutation: Mutation | None = Field(
        default=None,
        description="The mutation applied after this round, if any.",
    )
    applied: bool = Field(
        default=False,
        description="Whether the proposed mutation was actually applied.",
    )
    pareto_frontier_size: int = Field(
        default=0,
        description="Number of Pareto-optimal points after this round (multi-objective mode).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the round completed.",
    )


class TrainingResult(BaseModel):
    """Summary returned when the entire training run finishes."""

    rounds: list[RoundResult] = Field(default_factory=list)
    best_round: int = Field(
        ...,
        ge=0,
        description="Index of the round that achieved the best validation score.",
    )
    best_val_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Highest validation mean score observed.",
    )
    final_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent configuration snapshot at the end of training.",
    )
    stop_reason: str = Field(
        ...,
        description="Why the training loop terminated (e.g. converged, max_rounds).",
    )
    pareto_frontier: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Final Pareto frontier when multi-objective optimisation was used. "
            "Each dict contains 'round_num', 'config', 'scores', and 'weighted_score'."
        ),
    )
    total_duration_s: float = Field(
        ...,
        ge=0.0,
        description="Total wall-clock duration of the training run in seconds.",
    )
    sandbox_session_id: str | None = Field(
        default=None,
        description=(
            "Sandbox session identifier when training ran in sandbox mode. "
            "None if sandbox was not used."
        ),
    )


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class GuardrailVerdict(BaseModel):
    """Decision produced by a single guardrail check."""

    passed: bool = Field(
        ...,
        description="Whether the guardrail considers the current state acceptable.",
    )
    should_stop: bool = Field(
        default=False,
        description="If True, the training loop should terminate immediately.",
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the verdict.",
    )
    guardrail_name: str = Field(
        ...,
        description="Name of the guardrail that produced this verdict.",
    )


# ---------------------------------------------------------------------------
# Mutable loop context
# ---------------------------------------------------------------------------


class LoopContext(BaseModel):
    """Mutable state container threaded through every pipeline stage.

    Stages read from and write to this context so that each stage remains
    decoupled from the others while sharing a single coherent state object.
    """

    # -- identifiers / references (serialised as opaque handles) -----------
    config: TrainingConfig
    backend_ref: Any = Field(
        default=None,
        description="Reference to the active ModelBackend instance.",
    )
    target_ref: Any = Field(
        default=None,
        description="Reference to the active TargetAgent instance.",
    )

    # -- round bookkeeping ------------------------------------------------
    round_num: int = Field(default=0, ge=0)

    # -- dataset splits ----------------------------------------------------
    train_set: list[EvalSample] = Field(default_factory=list)
    val_set: list[EvalSample] = Field(default_factory=list)

    # -- per-round artefacts (written then consumed within a round) --------
    train_results: list[ScoredResult] = Field(default_factory=list)
    train_scores: ScoreCard | None = None
    val_scores: ScoreCard | None = None
    reflection: Reflection | None = None
    proposed_mutation: Mutation | None = None

    # -- cross-round state -------------------------------------------------
    history: list[RoundResult] = Field(default_factory=list)
    max_history_size: int = Field(default=100, ge=1, help="Cap history size to prevent unbounded memory growth in long training runs.")
    current_config: dict[str, Any] = Field(default_factory=dict)
    best_config: dict[str, Any] = Field(default_factory=dict)
    best_val_score: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}
