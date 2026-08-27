# Thin task runner for roborl. Every target echoes its commands — no magic.
# `make check` is the pre-commit gate: run it before every commit.

.PHONY: setup fmt fmt-check lint typecheck test test-all check demo clean

setup:  ## Install dev environment and pre-commit hooks
	uv sync --group dev
	uv run pre-commit install

fmt:  ## Auto-format code and fix trivially fixable lint issues
	uv run ruff format .
	uv run ruff check --fix .

fmt-check:  ## Verify formatting without changing files
	uv run ruff format --check .

lint:  ## Lint without fixing
	uv run ruff check .

typecheck:  ## Static type checking
	uv run mypy src tests

test:  ## Fast test suite (unit + smoke), the CI default
	uv run pytest -m "unit or smoke"

test-all:  ## Entire test suite including slow tests
	uv run pytest -m ""

check: fmt-check lint typecheck test  ## Full local gate: run before every commit

demo:  ## Random-agent pipeline check on CartPole (no W&B account needed)
	uv run roborl demo

clean:  ## Remove caches and local run artifacts (never touches uv.lock)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .cache runs videos wandb
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
