# Troubleshooting Guide | 故障排除指南

Common issues and solutions for Nuwa v0.2.0.

Nuwa v0.2.0 常见问题与解决方案。

---

## Table of Contents | 目录

- [LLM & API Errors](#llm--api-errors)
- [Training Loop Issues](#training-loop-issues)
- [Web Dashboard Issues](#web-dashboard-issues)
- [Sandbox Issues](#sandbox-issues)
- [Installation & Import Errors](#installation--import-errors)
- [Demo Mode](#demo-mode)
- [Debug Logging](#debug-logging)
- [Performance Tuning](#performance-tuning)

---

## LLM & API Errors

### `LLM call failed after 3 attempts`

**Symptoms**: Training fails immediately or intermittently with this error in
the logs.

**Causes & Solutions**:

1. **Invalid or missing API key**

   Ensure the correct API key is provided:

   ```bash
   # Via environment variable (provider-specific)
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export DEEPSEEK_API_KEY="..."

   # Or via config file / POST /api/config
   # Set the llm_api_key field
   ```

   When using the web API, pass the key in the config request body:
   ```json
   {
     "llm_api_key": "sk-...",
     "llm_model": "openai/gpt-4o"
   }
   ```

2. **Rate limiting by the provider**

   LiteLLM retries on `RateLimitError` with exponential backoff, but 3 retries
   may not be enough during heavy usage.

   Solutions:
   - Reduce `samples_per_round` to lower concurrent LLM calls.
   - Increase `LLM_MAX_RETRIES` by setting it before importing Nuwa (advanced).
   - Wait a few minutes and retry.
   - Upgrade your provider API tier for higher rate limits.

3. **Network connectivity**

   Verify network access to the LLM provider:

   ```bash
   # Test connectivity (example for OpenAI)
   curl -I https://api.openai.com/v1/models

   # If using a custom base URL, verify it is reachable
   curl -I https://your-llm-proxy.example.com/v1/models
   ```

   If behind a proxy, ensure `HTTP_PROXY` / `HTTPS_PROXY` environment variables
   are set.

4. **Incorrect model identifier**

   LiteLLM uses `provider/model` format. Verify your model string:
   - Correct: `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `deepseek/deepseek-chat`
   - Wrong: `gpt-4o` (missing provider prefix)

---

### `Circuit breaker open: 5 consecutive failures`

**Symptoms**: LLM calls are blocked even though the API key and network appear
correct.

**Cause**: The circuit breaker trips after 5 consecutive LLM failures and
blocks calls for 60 seconds.

**Solutions**:

1. **Wait 60 seconds** for the circuit breaker to enter half-open state and
   retry automatically.
2. **Check the LLM provider's health** -- there may be an ongoing outage:
   - [OpenAI Status](https://status.openai.com)
   - [Anthropic Status](https://status.anthropic.com)
3. **Restart the server** to reset the circuit breaker state (it is
   in-memory).
4. **Review server logs** for the original 5 failures that triggered the
   breaker.

---

## Training Loop Issues

### `Training aborted during stage`

**Symptoms**: Training stops early with a message like
`aborted during stage: ...`.

**Causes & Solutions**:

1. **Missing or unclear `training_direction`**

   The `training_direction` guides LLM-driven evaluation and mutation. If it
   is vague, the LLM may produce invalid outputs.

   ```json
   {
     "training_direction": "Improve the chatbot's ability to answer technical questions about Python accurately and concisely."
   }
   ```

   Be specific about the desired improvement, not just "make it better."

2. **Invalid model configuration**

   Ensure the connector parameters match your agent:
   - For `http` connector: verify the `url` in `connector_params`.
   - For `cli` connector: verify the `command` in `connector_params`.
   - For `function` connector: verify the `func` in `connector_params`.

3. **Guardrail triggered**

   Training may be halted by a guardrail (overfitting, regression, consistency).
   Check the `stop_reason` in the training result:
   ```json
   { "stop_reason": "Guardrail 'OverfittingGuardrail': train-val gap exceeded threshold" }
   ```
   Adjust thresholds in the config:
   - `overfitting_threshold`: increase to allow more train-val gap.
   - `regression_tolerance`: increase to allow larger score drops.
   - `consistency_threshold`: decrease to allow more variance.

---

### `No configuration set. POST /api/config first.`

**Symptoms**: `POST /api/train/start` returns HTTP 400.

**Cause**: The server has no stored configuration. Configuration must be set
before starting training.

**Solution**:

```bash
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "llm_model": "openai/gpt-4o",
    "training_direction": "Improve response quality",
    "max_rounds": 5
  }'
```

Then start training:

```bash
curl -X POST http://localhost:8080/api/train/start
```

---

## Web Dashboard Issues

### `Rate limit exceeded. Try again later.`

**Symptoms**: API returns HTTP 429.

**Cause**: More than 120 requests per 60 seconds from the same IP address.

**Solutions**:

1. **Reduce request frequency** -- avoid polling endpoints in tight loops. Use
   the SSE stream (`GET /api/train/events`) for real-time updates instead of
   polling `GET /api/status`.
2. **Wait 60 seconds** for the rate limit window to reset.
3. For production deployments, configure rate limiting at the reverse-proxy
   level instead.

---

### SSE stream not connecting

**Symptoms**: `GET /api/train/events` returns HTTP 501 or no events.

**Causes & Solutions**:

1. **`sse-starlette` not installed**

   Error response:
   ```json
   { "error": "sse-starlette is not installed." }
   ```

   Install the web dependencies:
   ```bash
   pip install "nuwa-trainer[web]"
   # or
   pip install sse-starlette>=2.0
   ```

2. **Proxy / firewall buffering**

   SSE requires streaming responses. Some reverse proxies buffer responses by
   default. Add the following headers in your proxy config:
   ```
   X-Accel-Buffering: no
   Cache-Control: no-cache
   Connection: keep-alive
   ```

3. **Browser EventSource limitations**

   Browser `EventSource` does not support custom headers. Use a polyfill like
   `eventsource-parser` or `@microsoft/fetch-event-source` if you need to pass
   API key headers.

---

### CORS errors in browser

**Symptoms**: Browser console shows CORS policy errors.

**Solutions**:

```bash
# Allow your dashboard origin
export NUWA_CORS_ORIGINS="https://your-dashboard.example.com"

# Or for local development
export NUWA_CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
```

Restart the server after changing environment variables.

---

## Sandbox Issues

### `Sandbox session not found` / `Unknown sandbox session`

**Symptoms**: Calling `sandbox.promote()` or `sandbox.discard()` raises
`ValueError`.

**Causes & Solutions**:

1. **Session already promoted or discarded**

   Sandbox sessions are one-shot. Once `promote()` or `discard()` is called,
   the session is finalized. Check your code to ensure you are not calling these
   methods twice on the same session.

2. **Different `SandboxManager` instance**

   Sessions are tracked per `SandboxManager` instance. Ensure the same instance
   that called `enter()` is used for `promote()` / `discard()`.

3. **Server restart**

   Sandbox sessions are stored in memory. Restarting the server loses all
   active sessions.

---

## Installation & Import Errors

### `ImportError: No module named 'litellm'`

**Symptoms**: Python cannot import `litellm` or other Nuwa dependencies.

**Solution**:

Nuwa must be installed with its dependencies:

```bash
# Install from source (recommended for GitHub installation)
git clone https://github.com/KBREX684/nuwa.git
cd nuwa
pip install -e .

# Verify installation
python -c "import nuwa; print(nuwa.__version__)"
```

To install web dependencies separately:

```bash
pip install -e ".[web]"
```

To install development dependencies:

```bash
pip install -e ".[dev]"
```

### `ModuleNotFoundError: No module named 'nuwa'`

**Cause**: Nuwa is not installed, or the virtual environment is not activated.

**Solution**:

```bash
# Ensure your virtual environment is active
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install nuwa
pip install -e .
```

### `pydantic` version conflict

**Symptoms**: Errors related to Pydantic v1/v2 mismatch.

**Solution**: Nuwa requires Pydantic v2:

```bash
pip install "pydantic>=2.0"
```

---

## Demo Mode

Demo mode runs a simulated training loop without real LLM API keys, useful for
testing the web dashboard UI.

### Using the CLI

```bash
# No special CLI flag for demo; use the web dashboard's demo endpoint
nuwa web --port 8080
```

### Using the API

```bash
# Start demo training
curl -X POST http://localhost:8080/api/demo/start

# Stream demo events
curl -N http://localhost:8080/api/train/events

# Check results
curl http://localhost:8080/api/results

# Stop demo early
curl -X POST http://localhost:8080/api/train/stop
```

### Demo Behavior

- Runs **5 simulated rounds** with 0.5-second delays per stage.
- Generates improving (randomized) scores starting around 0.45 and increasing.
- Produces mock reflections, mutations, and a final `TrainingResult`.
- The demo result can be accepted/rejected/extended via `POST /api/approve`.
- No API keys or external services are required.

---

## Debug Logging

Enable verbose logging to diagnose issues:

### CLI

```bash
# Enable verbose output
nuwa train --verbose
nuwa run --config nuwa-config.yaml --verbose
```

### Python Logging

```python
import logging

# Enable DEBUG logging for all Nuwa modules
logging.basicConfig(level=logging.DEBUG)

# Or enable for specific modules
logging.getLogger("nuwa.llm").setLevel(logging.DEBUG)      # LLM calls
logging.getLogger("nuwa.engine").setLevel(logging.DEBUG)    # Training loop
logging.getLogger("nuwa.web").setLevel(logging.DEBUG)       # Web server
logging.getLogger("nuwa.sandbox").setLevel(logging.DEBUG)   # Sandbox
```

### Web Server

```bash
# Start with auto-reload and verbose logging
nuwa web --reload

# Or set log level via environment variable
export LOG_LEVEL=DEBUG
nuwa web
```

### Key Log Messages

| Log Message | Meaning |
|-------------|---------|
| `Token usage -- prompt: N, completion: N, total: N` | Successful LLM call |
| `LLM call attempt X/3 failed` | Retrying after a transient error |
| `Circuit breaker tripped after 5 failures` | LLM calls are now blocked |
| `Round N: validation score regressed from best` | Mutation rolled back |
| `Guardrail triggered stop` | Training halted by a guardrail |
| `Sandbox session X created` | Sandbox isolation active |

---

## Performance Tuning | 性能调优

### Concurrency | 并发

The training loop supports parallel execution via `parallel_config`:

```python
from nuwa.engine.loop import TrainingLoop

loop = TrainingLoop(
    config=training_config,
    backend=backend,
    target=target,
    guardrails=guardrails,
    parallel_config={
        "max_concurrency": 10,    # Max concurrent agent invocations (default: 5, max: 50)
        "judges": [],             # Multi-judge evaluation configs
        "strategy": "mean",       # Ensemble strategy: "mean", "median", "max", "weighted"
    },
)
```

**Guidelines**:
- Start with `max_concurrency=5` (the default).
- Increase gradually; too high concurrency triggers provider rate limits.
- The hard limit is 50 concurrent invocations.

### Samples Per Round | 每轮样本数

`samples_per_round` controls how many evaluation samples are generated and
scored each round.

| Value | LLM Calls per Round | Use Case |
|-------|---------------------|----------|
| 5-10 | ~20-40 | Quick iteration, testing |
| 20 (default) | ~80-100 | Balanced quality and speed |
| 50+ | ~200+ | High-confidence results, slow |

**Trade-off**: Higher values produce more reliable scores but cost more tokens
and take longer. The training scheduler automatically adjusts the budget per
round based on convergence behavior.

### Training Rounds | 训练轮数

- Start with `max_rounds=5` for experimentation.
- Increase to `10-20` for production training.
- The scheduler may stop early if convergence is detected (no improvement over
  3 consecutive rounds).

### Other Knobs | 其他参数

| Parameter | Default | Effect |
|-----------|---------|--------|
| `train_val_split` | 0.7 | Higher = more training data, less validation coverage |
| `overfitting_threshold` | 0.15 | Lower = more sensitive to overfitting detection |
| `regression_tolerance` | 0.05 | Higher = allows more score fluctuation between rounds |
| `consistency_threshold` | 0.8 | Higher = requires more stable results across runs |
| `llm_temperature` | 0.7 | Lower (0.1-0.3) = more deterministic; higher (0.8-1.0) = more creative |

### Memory Usage | 内存使用

- Round history is capped at 100 entries (`max_history_size`) to prevent
  unbounded memory growth in long runs.
- ScoreCard results (individual scored samples) are retained in the
  `TrainingResult`. For very large `samples_per_round` values, consider the
  memory footprint.

### LLM Cost Optimization | LLM 成本优化

1. **Use cheaper models for iteration**: Start with `openai/gpt-4o-mini` or
   `deepseek/deepseek-chat` for fast, cheap experimentation.
2. **Reduce `samples_per_round`**: Fewer samples = fewer LLM calls.
3. **Enable the circuit breaker**: Already enabled by default; prevents wasted
   calls when the provider is degraded.
4. **Monitor token usage**: Check server logs for `Token usage` lines to
   understand cumulative costs.
