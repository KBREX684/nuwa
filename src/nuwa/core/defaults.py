"""Centralised default values for the Nuwa framework.

All magic numbers, default model names, thresholds, and timeouts live here
so that every module references the same source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = "openai/gpt-4o"
"""Default LLM model identifier (provider/model format)."""

DEFAULT_LLM_TEMPERATURE: float = 0.7
"""Default sampling temperature for general LLM calls."""

# Per-stage temperatures (carefully tuned for each task).
TEMPERATURE_DATASET_GEN: float = 0.9
TEMPERATURE_REFLECTION: float = 0.3
TEMPERATURE_MUTATION: float = 0.4
TEMPERATURE_SCORING: float = 0.1

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_ROUNDS: int = 10
DEFAULT_SAMPLES_PER_ROUND: int = 20
DEFAULT_TRAIN_VAL_SPLIT: float = 0.7
DEFAULT_OVERFITTING_THRESHOLD: float = 0.15
DEFAULT_REGRESSION_TOLERANCE: float = 0.05
DEFAULT_CONSISTENCY_THRESHOLD: float = 0.8
DEFAULT_CONSISTENCY_RUNS: int = 3

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

PASS_THRESHOLD: float = 0.7
"""Minimum score to consider an individual sample as passing."""

FAILURE_SCORE_THRESHOLD: float = 0.7
"""Score below which a sample is considered a failure for reflection."""

# ---------------------------------------------------------------------------
# Scoring weights (immutable scorer vs LLM)
# ---------------------------------------------------------------------------

IMMUTABLE_SCORER_WEIGHT: float = 0.4
LLM_SCORER_WEIGHT: float = 0.6

# ---------------------------------------------------------------------------
# Concurrency & timeout defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_CONCURRENCY: int = 5
"""Maximum concurrent agent invocations within a stage."""

MAX_CONCURRENCY_LIMIT: int = 50
"""Hard upper limit on concurrency for parallel stages."""

INVOKE_TIMEOUT_S: float = 120.0
"""Per-invocation timeout in seconds."""

JUDGE_CONCURRENCY: int = 5
"""Per-judge concurrency for parallel evaluation."""

# ---------------------------------------------------------------------------
# LLM backend retry
# ---------------------------------------------------------------------------

LLM_MAX_RETRIES: int = 3
LLM_INITIAL_BACKOFF_S: float = 1.0
LLM_BACKOFF_MULTIPLIER: float = 2.0

# ---------------------------------------------------------------------------
# Scheduler convergence
# ---------------------------------------------------------------------------

CONVERGENCE_WINDOW: int = 3
MIN_IMPROVEMENT: float = 0.01
DEFAULT_SCORE_TARGET: float = 0.95
PARETO_STALL_WINDOW: int = 3

# ---------------------------------------------------------------------------
# Reflection context-window discipline
# ---------------------------------------------------------------------------

MAX_FAILURES_FOR_PROMPT: int = 10
MAX_PASSES_FOR_PROMPT: int = 3
MAX_OUTPUT_CHARS: int = 500
MAX_FAILURE_PATTERTS: int = 15

# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

MAX_MUTATIONS_PER_PROPOSAL: int = 5

# ---------------------------------------------------------------------------
# History & memory
# ---------------------------------------------------------------------------

MAX_HISTORY_SIZE: int = 100
"""Maximum number of round results kept in LoopContext.history."""

# ---------------------------------------------------------------------------
# Immutable scorer defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_OUTPUT_LENGTH: int = 1
DEFAULT_MAX_OUTPUT_LENGTH: int = 10_000
DEFAULT_MAX_LATENCY_MS: float = 30_000.0

# ---------------------------------------------------------------------------
# Web / server
# ---------------------------------------------------------------------------

DEFAULT_WEB_HOST: str = "0.0.0.0"
DEFAULT_WEB_PORT: int = 8080
SSE_KEEPALIVE_TIMEOUT_S: float = 15.0
SHUTDOWN_TIMEOUT_S: float = 5.0
DEMO_MAX_ROUNDS: int = 5

# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

DEFAULT_HTTP_TIMEOUT: int = 30
DEFAULT_CLI_TIMEOUT: int = 60

# ---------------------------------------------------------------------------
# Git tracker
# ---------------------------------------------------------------------------

GIT_OPERATION_TIMEOUT: int = 30
