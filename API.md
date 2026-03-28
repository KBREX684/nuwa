# Nuwa REST API Documentation | REST API 文档

**Version**: 0.2.0
**Base URL**: `http://<host>:8080`
**License**: BSL 1.1

---

## Authentication | 认证

When the `NUWA_API_KEY` environment variable is set, all `/api/*` endpoints
require authentication. Provide the key via either header:

```http
X-API-Key: your-secret-key
```

```http
Authorization: Bearer your-secret-key
```

When `NUWA_API_KEY` is not set, authentication is disabled.

---

## Common Error Responses | 通用错误响应

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| 400 | Bad Request | Invalid request body or configuration error. |
| 401 | Unauthorized | Missing or incorrect API key. |
| 404 | Not Found | Requested resource or static file not found. |
| 409 | Conflict | Training is already running or conflicting state. |
| 429 | Too Many Requests | Rate limit exceeded (120 requests / 60 seconds per IP). |
| 501 | Not Implemented | SSE support unavailable (`sse-starlette` not installed). |

All error responses follow the format:

```json
{
  "error": "Description of the error."
}
```

Internal exception details are never exposed to clients; check server logs for
full diagnostics.

---

## Endpoints | 接口列表

---

### `GET /api/health`

Health check for load balancers and monitoring.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Response

```json
{
  "status": "ok",
  "version": "0.2.0"
}
```

---

### `GET /api/status`

Return the current training dashboard state.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Response

```json
{
  "training_status": "idle",
  "current_round": 0,
  "current_stage": "",
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `training_status` | `string` | One of: `idle`, `running`, `completed`, `error` |
| `current_round` | `integer` | Current round number (0 when idle) |
| `current_stage` | `string` | Current pipeline stage name (e.g. `dataset_gen`, `evaluation`) |
| `error` | `string \| null` | Error message if status is `error` |

---

### `POST /api/config`

Validate and store a training configuration. Must be called before
`POST /api/train/start`.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Request Body (`ConfigRequest`)

```json
{
  "llm_model": "openai/gpt-4o",
  "llm_api_key": "sk-...",
  "llm_base_url": null,
  "connector_type": "function",
  "connector_params": {},
  "training_direction": "Improve the target agent's response quality.",
  "max_rounds": 10,
  "samples_per_round": 20,
  "train_val_split": 0.7,
  "overfitting_threshold": 0.15,
  "regression_tolerance": 0.05,
  "consistency_threshold": 0.8
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_model` | `string` | `"openai/gpt-4o"` | Model identifier in `provider/model` format |
| `llm_api_key` | `string \| null` | `null` | API key for the LLM provider |
| `llm_base_url` | `string \| null` | `null` | Optional base URL override for the provider |
| `connector_type` | `"http" \| "cli" \| "function"` | `"function"` | Connector adapter type |
| `connector_params` | `object` | `{}` | Keyword arguments forwarded to the connector |
| `training_direction` | `string` | `"Improve the target agent's response quality."` | Natural-language training goal |
| `max_rounds` | `integer` | `10` | Maximum training rounds (>= 1) |
| `samples_per_round` | `integer` | `20` | Evaluation samples per round (>= 1) |
| `train_val_split` | `float` | `0.7` | Train/validation split ratio (0.0 < x < 1.0) |
| `overfitting_threshold` | `float` | `0.15` | Max train-val score gap before overfitting alert (>= 0.0) |
| `regression_tolerance` | `float` | `0.05` | Allowed score regression before rollback (>= 0.0) |
| `consistency_threshold` | `float` | `0.8` | Minimum consistency ratio (0.0-1.0) |

#### Response

**Success** (200):

```json
{
  "ok": true
}
```

**Validation error** (400):

```json
{
  "error": "Request failed. Check server logs for details."
}
```

---

### `POST /api/train/start`

Start a training run in a background asyncio task. Requires a configuration to
be set via `POST /api/config` first.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Request Body

None (empty body).

#### Response

**Success** (200):

```json
{
  "ok": true
}
```

**No configuration** (400):

```json
{
  "error": "No configuration set. POST /api/config first."
}
```

**Already running** (409):

```json
{
  "error": "Training is already running."
}
```

---

### `POST /api/train/stop`

Request the training loop to stop after the current round. Cancels the
background training task.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Request Body

None (empty body).

#### Response

```json
{
  "ok": true
}
```

---

### `GET /api/train/events`

Server-Sent Events (SSE) endpoint for real-time training updates. Keeps the
connection open and pushes events as training progresses.

**Authentication**: Required if `NUWA_API_KEY` is set.

**Requires**: `sse-starlette` package installed. Returns HTTP 501 if missing.

#### SSE Event Types | SSE 事件类型

All events are delivered as JSON in the `data` field. Some events carry a
custom `event` field.

##### `round_start`

Emitted when a new round begins.

```json
{
  "type": "round_start",
  "round": 1,
  "max_rounds": 10
}
```

##### `stage`

Emitted when a pipeline stage starts within a round.

```json
{
  "type": "stage",
  "round": 1,
  "stage": "dataset_gen"
}
```

Stage names: `dataset_gen`, `execution`, `evaluation`, `reflection`, `mutation`.

##### `round_end`

Emitted when a round completes.

```json
{
  "type": "round_end",
  "round": 1,
  "train_score": 0.5234,
  "val_score": 0.4812,
  "reflection": "The agent improved on detail but still misses edge cases.",
  "mutation": "Enhance system prompt with detail-level=high",
  "applied": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `round` | `integer` | Round number |
| `train_score` | `float` | Mean training score (0.0-1.0, rounded to 4 decimals) |
| `val_score` | `float` | Mean validation score (0.0-1.0, rounded to 4 decimals) |
| `reflection` | `string` | Reflection diagnosis text |
| `mutation` | `string` | Mutation description text |
| `applied` | `boolean` | Whether the mutation was applied |

##### `training_complete`

Emitted when the training run finishes (successfully, stopped by user, or
converged).

```json
{
  "type": "training_complete",
  "best_round": 7,
  "best_score": 0.8923,
  "stop_reason": "max_rounds reached"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `best_round` | `integer` | Round with the best validation score |
| `best_score` | `float` | Best validation score achieved |
| `stop_reason` | `string` | Reason for stopping (e.g. `max_rounds reached`, `stopped by user`, `Guardrail '...': ...`) |

##### `error`

Emitted when training fails.

```json
{
  "type": "error",
  "message": "Training error. Check server logs."
}
```

##### `ping` (keepalive)

Sent automatically every 15 seconds when no other events are queued.

```
event: ping
data:
```

#### Connection Behavior

- The server checks `request.is_disconnected()` on each iteration.
- The connection stays open until the client disconnects.
- Keepalive pings are sent every 15 seconds of inactivity.

---

### `GET /api/results`

Return the full `TrainingResult` of the most recent training run.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Response

**Result available** (200):

```json
{
  "rounds": [
    {
      "round_num": 1,
      "train_scores": {
        "results": [
          {
            "sample": { "id": "...", "input_text": "...", "expected_behavior": "...", "difficulty": "medium", "tags": [] },
            "response": { "output_text": "...", "latency_ms": 120.0, "raw_metadata": {} },
            "score": 0.65,
            "reasoning": "..."
          }
        ],
        "failure_analysis": "...",
        "objective_scores": null,
        "mean_score": 0.65,
        "pass_rate": 0.5714
      },
      "val_scores": { "...": "..." },
      "reflection": {
        "round_num": 1,
        "diagnosis": "...",
        "failure_patterns": ["..."],
        "proposed_changes": ["..."],
        "priority": "medium"
      },
      "mutation": {
        "description": "...",
        "original_config": {},
        "proposed_config": {},
        "reasoning": "..."
      },
      "applied": true,
      "pareto_frontier_size": 0,
      "timestamp": "2026-03-28T12:00:00Z",
      "error": null
    }
  ],
  "best_round": 7,
  "best_val_score": 0.8923,
  "final_config": { "system_prompt": "..." },
  "stop_reason": "max_rounds reached",
  "pareto_frontier": null,
  "total_duration_s": 125.4,
  "sandbox_session_id": null
}
```

**No result yet** (200):

```json
null
```

---

### `GET /api/results/rounds`

Return round-by-round summary data suitable for charting. A flattened,
lightweight version of the full results.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Response

**Result available** (200):

```json
[
  {
    "round": 1,
    "train_score": 0.5234,
    "val_score": 0.4812,
    "train_pass_rate": 0.5714,
    "val_pass_rate": 0.3333,
    "reflection": "The agent improved on detail...",
    "mutation": "Enhance system prompt...",
    "applied": true,
    "timestamp": "2026-03-28T12:00:00Z"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `round` | `integer` | Round number |
| `train_score` | `float` | Mean training score |
| `val_score` | `float \| null` | Mean validation score (null if validation was skipped) |
| `train_pass_rate` | `float` | Fraction of training samples with score >= 0.7 |
| `val_pass_rate` | `float \| null` | Fraction of validation samples with score >= 0.7 |
| `reflection` | `string` | Reflection diagnosis |
| `mutation` | `string` | Mutation description |
| `applied` | `boolean` | Whether the mutation was applied |
| `timestamp` | `string` | ISO 8601 UTC timestamp |

**No result yet** (200):

```json
[]
```

---

### `POST /api/approve`

Handle human-in-the-loop decisions on training results.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Request Body (`ApproveRequest`)

```json
{
  "decision": "accept",
  "extra_rounds": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decision` | `"accept" \| "reject" \| "extend"` | (required) | The human decision on the training result |
| `extra_rounds` | `integer` | `5` | Additional rounds when `decision` is `"extend"` (>= 1) |

#### Decision Behavior

| Decision | Behavior |
|----------|----------|
| `accept` | Saves the final config to disk (`.nuwa/accepted_config.json`) |
| `reject` | Discards the training result and resets status to `idle` |
| `extend` | Adds `extra_rounds` to `max_rounds` and restarts training |

#### Response

**Success** (200):

```json
{
  "ok": true
}
```

**No result to accept** (400):

```json
{
  "error": "No training result to accept."
}
```

**Already running on extend** (409):

```json
{
  "error": "Training is already running."
}
```

---

### `GET /api/history`

Return past training round results from the on-disk `RunLog`. This endpoint
reads from persisted log files and is independent of the in-memory training
state.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Response

```json
[
  {
    "round_num": 1,
    "train_scores": { "..." },
    "val_scores": { "..." },
    "reflection": { "..." },
    "mutation": { "..." },
    "applied": true,
    "pareto_frontier_size": 0,
    "timestamp": "2026-03-28T12:00:00Z",
    "error": null
  }
]
```

Returns a list of `RoundResult` objects from all previous training runs in the
project directory. Returns an empty list if no history is found.

**Error** (500):

```json
{
  "error": "Request failed. Check server logs for details."
}
```

---

### `POST /api/demo/start`

Start a mock training run with simulated data for UI testing. No real LLM API
keys are required. Runs 5 simulated rounds with artificial delays.

**Authentication**: Required if `NUWA_API_KEY` is set.

#### Request Body

None (empty body).

#### Response

**Success** (200):

```json
{
  "ok": true
}
```

**Already running** (409):

```json
{
  "error": "Training is already running."
}
```

#### Demo Behavior

- Runs 5 simulated rounds with 0.5-second delays per stage.
- Generates improving (but randomized) scores.
- Emits the same SSE event types as real training (`round_start`, `stage`,
  `round_end`, `training_complete`).
- Produces a `TrainingResult` that can be queried via `GET /api/results`.
- Can be stopped with `POST /api/train/stop`.

---

## Pydantic Model Schemas | Pydantic 模型定义

The API uses the following Pydantic v2 models for request/response validation.
All models use strict type checking.

### `ConfigRequest`

```python
class ConfigRequest(BaseModel):
    llm_model: str = "openai/gpt-4o"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    connector_type: Literal["http", "cli", "function"] = "function"
    connector_params: dict[str, Any] = {}
    training_direction: str = "Improve the target agent's response quality."
    max_rounds: int = 10
    samples_per_round: int = 20
    train_val_split: float = 0.7
    overfitting_threshold: float = 0.15
    regression_tolerance: float = 0.05
    consistency_threshold: float = 0.8
```

### `ApproveRequest`

```python
class ApproveRequest(BaseModel):
    decision: Literal["accept", "reject", "extend"]
    extra_rounds: int = Field(default=5, ge=1)
```

### `TrainingResult`

```python
class TrainingResult(BaseModel):
    rounds: list[RoundResult] = []
    best_round: int          # Round with highest validation score
    best_val_score: float    # 0.0 - 1.0
    final_config: dict       # Final agent configuration
    stop_reason: str         # Why training ended
    pareto_frontier: list | None  # Multi-objective frontier (if enabled)
    total_duration_s: float  # Wall-clock duration in seconds
    sandbox_session_id: str | None  # Sandbox session (if sandbox mode used)
```

### `RoundResult`

```python
class RoundResult(BaseModel):
    round_num: int
    train_scores: ScoreCard
    val_scores: ScoreCard | None
    reflection: Reflection
    mutation: Mutation | None
    applied: bool
    pareto_frontier_size: int = 0
    timestamp: datetime      # ISO 8601 UTC
    error: str | None
```

### `ScoreCard`

```python
class ScoreCard(BaseModel):
    results: list[ScoredResult] = []
    failure_analysis: str = ""
    objective_scores: dict[str, float] | None = None
    # Computed fields:
    mean_score: float        # Arithmetic mean of all scores
    pass_rate: float         # Fraction of scores >= 0.7
```

### `Reflection`

```python
class Reflection(BaseModel):
    round_num: int
    diagnosis: str
    failure_patterns: list[str] = []
    proposed_changes: list[str] = []
    priority: Literal["low", "medium", "high", "critical"]
```

### `Mutation`

```python
class Mutation(BaseModel):
    description: str
    original_config: dict[str, Any] = {}
    proposed_config: dict[str, Any] = {}
    reasoning: str
```

---

## Environment Variables | 环境变量

| Variable | Default | Description |
|----------|---------|-------------|
| `NUWA_API_KEY` | *(unset)* | API key for authentication. Unset = no auth. |
| `NUWA_CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | Comma-separated allowed CORS origins. Use `*` for all (dev only). |

---

## Quick Start Example | 快速上手示例

```bash
# Start the server with authentication
export NUWA_API_KEY="my-secret-key"
nuwa web --port 8080

# Configure training
curl -X POST http://localhost:8080/api/config \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "llm_model": "openai/gpt-4o",
    "llm_api_key": "sk-...",
    "training_direction": "Improve chatbot response quality",
    "max_rounds": 5
  }'

# Start training
curl -X POST http://localhost:8080/api/train/start \
  -H "X-API-Key: my-secret-key"

# Stream events (SSE)
curl -N http://localhost:8080/api/train/events \
  -H "X-API-Key: my-secret-key"

# Get results
curl http://localhost:8080/api/results \
  -H "X-API-Key: my-secret-key"

# Approve results
curl -X POST http://localhost:8080/api/approve \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"decision": "accept"}'
```
