.PHONY: lint format-check type-check test quality

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy .

test:
	uv run pytest --cov=pricing_agent --cov-report=term-missing

quality: lint format-check type-check test
