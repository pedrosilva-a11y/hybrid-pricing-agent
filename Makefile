N_USERS ?= 100
N_WEEKS ?= 24
RANDOMIZATION_RATE ?= 0.1
SEED ?= 42

.PHONY: lint format-check type-check test quality populate-db

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

type-check:
	uv run mypy .

test:
	uv run pytest --cov=pricing_agent --cov-report=term-missing

quality: lint format-check type-check test

populate-db:
	uv run python -m pricing_agent.data.main \
		--n-users $(N_USERS) \
		--n-weeks $(N_WEEKS) \
		--randomization-rate $(RANDOMIZATION_RATE) \
		--seed $(SEED)
