# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-03-29

### Fixed

- SSE reconnect now uses exponential backoff (max 8 retries, 30s cap) instead of silent disconnect
- SSE keepalive uses proper SSE comment format instead of named `ping` event that was ignored by browsers
- API error responses now pass through `ConfigError`, `ConnectorError`, `LLMError` messages verbatim instead of masking with generic text
- Config diff display: `original_config` and `proposed_config` now included in `/api/results` and `/api/results/rounds` responses
- Reject approval (`POST /api/approve` with `decision=reject`) now fully resets all training state
- Default web port changed from 8080 to 9090 to avoid conflict with Docker proxy on common Linux setups

### Changed

- Alpine.js and ECharts bundled as local vendor files with CDN fallback for offline reliability
- Configurable API base URL via `<meta name="api-base">` tag in index.html
- CORS default origins updated to include both port 9090 and 8080
- README reviewed and updated: added TOC, CI badge, 30-second demo, environment requirements, SDK example fixes, version/port corrections

## [0.2.0] - 2026-03-28

### Added

- Circuit breaker pattern in LLM backend: automatic backoff after 5 consecutive failures, 60s recovery window
- Rate limiting on Web API endpoints: 120 requests per minute per client IP
- Health check endpoint `GET /api/health` for load balancer and monitoring integration
- Path traversal protection in static file serving
- Jitter in LLM retry exponential backoff to prevent thundering herd
- Recoverable error handling in training loop: LLM/connector/config errors skip the round instead of crashing
- `error` field on `RoundResult` to record skipped round information
- New documentation: [SECURITY.md](SECURITY.md), [API.md](API.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- DeepSeek API support in E2E tests and demo example

### Changed

- **Breaking**: Exception handling in training loop refined — `LLMError`, `ConnectorError`, `ConfigError` are now recoverable (round skipped), only truly fatal errors crash the run
- Execution stage no longer masks system-level errors (`ConfigError`, `LLMError`) as agent failures — these now propagate correctly
- Mutation stage `_apply_config_path` now raises `ValueError` if a dot-path would overwrite a non-dict value, preventing config corruption
- Parallel executor progress callback exceptions now logged at WARNING level with full exception info
- Training loop uses `try/finally` to ensure sandbox session state is always logged on exit
- Web SPA fallback now validates resolved paths to prevent directory traversal
- Version bumped to 0.2.0, development status changed to Beta
- License classifier corrected to "Other/Proprietary License"
- README updated: removed PyPI references, added architecture diagram, added documentation links, improved formatting
- GitHub-only distribution (no PyPI publishing)

### Security

- Per-IP rate limiting prevents API abuse (120 req/min)
- Circuit breaker prevents LLM API cascade failures
- Path traversal attack vector closed in static file serving
- Recoverable error handling prevents training data loss on transient failures

## [0.1.0] - 2025-03-28

### Added

- Core training loop with 6-stage pipeline (dataset generation, execution, evaluation, reflection, mutation, validation)
- LLM backend via LiteLLM with retry and exponential backoff
- Multi-judge parallel evaluation framework
- Guardrails: overfitting detection, regression detection, consistency checks
- Sandbox isolation for safe agent training
- Three agent connectors: function call, HTTP API, CLI
- Web UI with SSE streaming for real-time training monitoring
- CLI via Typer for command-line training
- SDK: `@trainable` decorator, `NuwaTrainer`, one-liner `train()` / `train_sync()`
- Artifact persistence (JSONL run log, config snapshots)
- Centralized defaults in `core/defaults.py`
- Immutable scorer for deterministic evaluation metrics
- Async context manager support on `NuwaTrainer`
- 101 unit + integration tests

### Changed

- License changed from MIT to Business Source License 1.1 (BSL 1.1)

### Security

- API key authentication for web server (`NUWA_API_KEY`)
- CORS origin validation from environment variable
- Input sanitization on git branch names and commit messages
- Sanitized error responses (no internal details leaked)

### Infrastructure

- GitHub Actions CI (lint, test, typecheck, security scan, coverage)
- Pre-commit hooks (ruff, mypy, trailing whitespace, merge conflict checks)
- Makefile for common dev workflows
- `.env.example` for environment variable documentation
