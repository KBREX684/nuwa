.PHONY: install install-dev test test-unit test-integration lint format typecheck clean check

# -------------------------------------------------------------------
# Install
# -------------------------------------------------------------------

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pre-commit
	pre-commit install

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

test:
	python3 -m pytest tests/ -v --tb=short

test-unit:
	python3 -m pytest tests/unit/ -v --tb=short

test-integration:
	python3 -m pytest tests/integration/ -v --tb=short

test-cov:
	python3 -m pytest tests/ -v --tb=short --cov=nuwa --cov-report=term-missing --cov-report=html

# -------------------------------------------------------------------
# Code quality
# -------------------------------------------------------------------

lint:
	python3 -m ruff check src/ tests/

format:
	python3 -m ruff format src/ tests/
	python3 -m ruff check --fix src/ tests/

typecheck:
	python3 -m mypy src/nuwa/ --ignore-missing-imports

# -------------------------------------------------------------------
# Combo
# -------------------------------------------------------------------

check: lint typecheck test
	@echo "All checks passed."

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
