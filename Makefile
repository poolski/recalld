.PHONY: setup install run test lint clean

setup: install
	uv run python scripts/setup

install:
	uv sync --all-groups

run:
	uv run recalld

test:
	uv run pytest

lint:
	uv run ruff check recalld tests
	uv run ruff format --check recalld tests

fmt:
	uv run ruff format recalld tests
	uv run ruff check --fix recalld tests

clean:
	rm -rf .venv __pycache__ .pytest_cache dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
