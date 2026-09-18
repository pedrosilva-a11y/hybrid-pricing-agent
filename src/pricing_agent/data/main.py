"""Entry point for synthetic pricing data generation."""

from pathlib import Path

from pricing_agent.data.cli import parse_args
from pricing_agent.data.config import build_pricing_data_config
from pricing_agent.data.db.database import connect_database, initialize_database
from pricing_agent.data.db.loader import load_pipeline_output
from pricing_agent.data.pipeline import run_pipeline

# Database Creation Global Variables
DATABASE_PATH = Path("data/pricing.duckdb")


def main() -> None:
    """Generate synthetic pricing data and persist it to DuckDB."""
    arguments = parse_args()
    config = build_pricing_data_config(
        n_users=arguments.n_users,
        n_weeks=arguments.n_weeks,
        randomization_rate=arguments.randomization_rate,
        seed=arguments.seed,
    )
    pipeline_output = run_pipeline(config)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    connection = connect_database(DATABASE_PATH)

    try:
        initialize_database(connection)

        load_pipeline_output(connection=connection, output=pipeline_output)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
