# Contributing to Nuwa

Thank you for your interest in contributing to Nuwa! This guide covers the development setup and contribution workflow.

## Development Setup

### Prerequisites

- Python 3.11+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/KBREX684/nuwa.git
cd nuwa

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
make install-dev
# Or manually:
# pip install -e ".[dev]"
```

### Environment Variables

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

## Development Workflow

### Common Commands

```bash
make lint          # Run ruff linter
make format        # Auto-format code with ruff
make typecheck     # Run mypy type checking
make test          # Run all tests
make test-unit     # Run unit tests only
make test-integration  # Run integration tests only
make test-cov      # Run tests with coverage report
make check         # lint + typecheck + test (run before PRs)
make clean         # Remove build artifacts and caches
```

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit:

```bash
# Install hooks (done by make install-dev)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Hooks include:
- **ruff** — linting and formatting
- **mypy** — type checking
- **trailing-whitespace** — whitespace cleanup
- **check-yaml/toml** — config validation
- **check-merge-conflict** — conflict marker detection

## Code Style

- **Formatter**: ruff (100 char line length, double quotes)
- **Linter**: ruff with `E`, `F`, `I`, `UP` rules
- **Type hints**: All public APIs should have type annotations
- **Imports**: Sorted with isort via ruff

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add multi-judge ensemble evaluation
fix: correct history trimming in loop context
refactor: centralize default values into defaults.py
test: add unit tests for response parser
docs: add .env.example for environment configuration
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Make changes** with appropriate tests
3. **Run checks** locally: `make check`
4. **Open a PR** against `main` with:
   - Clear description of changes
   - Reference to any related issues
   - Confirmation that CI passes

### PR Checklist

- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] New code has tests
- [ ] No unnecessary `# type: ignore` comments

## Project Structure

```
src/nuwa/
  cli/              # CLI (Typer)
  config/           # Configuration schema (Pydantic)
  connectors/       # Agent adapters (function call, HTTP, CLI)
  core/             # Types, protocols, exceptions, defaults
  engine/           # Training loop & pipeline stages
    parallel/       # Multi-judge parallel evaluation
  guardrails/       # Overfitting, regression, consistency checks
  llm/              # LLM backend (LiteLLM), prompts, response parser
  persistence/      # Artifact store, run log, git tracker
  sandbox/          # Sandbox isolation for agent training
  sdk/              # Public API: @trainable, NuwaTrainer, train()
  web/              # FastAPI web UI with SSE streaming
tests/
  unit/             # Unit tests
  integration/      # Integration tests
```

## License

This project uses the **Business Source License 1.1 (BSL 1.1)**. Contributions are welcome, but please note:

- The Licensed Work is free for development, testing, and evaluation
- Production use requires a commercial license
- On 2028-01-01, the license automatically converts to Apache 2.0

By contributing, you agree that your contributions will be licensed under the same terms.

## Questions or Issues?

Open an issue at [github.com/KBREX684/nuwa/issues](https://github.com/KBREX684/nuwa/issues).
