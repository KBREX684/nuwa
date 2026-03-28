# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- 98 unit + integration tests

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
