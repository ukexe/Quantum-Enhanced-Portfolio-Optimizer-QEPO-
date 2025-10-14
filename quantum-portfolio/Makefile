.PHONY: help install install-dev test test-cov lint format clean pre-commit setup-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest

test-cov: ## Run tests with coverage report
	python -m pytest --cov=src/qepo --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check if code is properly formatted
	black --check src/ tests/
	isort --check-only src/ tests/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit: ## Install pre-commit hooks
	pre-commit install

setup-dev: install-dev pre-commit ## Set up development environment
	@echo "Development environment set up successfully!"
	@echo "Run 'make test' to verify everything works."

# Data and optimization commands
data: ## Download and prepare data
	python -m qepo.cli data ingest

optimize: ## Run portfolio optimization
	python -m qepo.cli optimize

backtest: ## Run backtesting
	python -m qepo.cli backtest

report: ## Generate report (requires run-id)
	@echo "Usage: make report RUN_ID=<mlflow_run_id>"
	@if [ -z "$(RUN_ID)" ]; then echo "Error: RUN_ID not provided"; exit 1; fi
	python -m qepo.cli report --run-id $(RUN_ID)

# CI/CD helpers
ci-test: ## Run tests as they would run in CI
	python -m pytest --cov=src/qepo --cov-report=xml --cov-report=term-missing

ci-lint: ## Run linting as it would run in CI
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Release helpers
version: ## Show current version
	@python -c "import qepo; print(qepo.__version__)"

build: ## Build the package
	python -m build

# Quick development workflow
dev: format lint test ## Run full development checks (format, lint, test)

# Quantum-specific commands
quantum-test: ## Run tests that require quantum simulators
	python -m pytest -m quantum

hardware-test: ## Run tests that require IBM Quantum hardware (requires API key)
	python -m pytest -m hardware

