"""Entry point for synthetic pricing data generation."""

from pathlib import Path

from pricing_agent.data.db.database import connect_database, initialize_database
from pricing_agent.data.db.loader import load_pipeline_output
from pricing_agent.data.pipeline import run_pipeline

# Database Creation Global Variables
DATABASE_PATH = Path("data/pricing.duckdb")


def main() -> None:
    """Generate syntehtic pricing data and persist it to DuckDB."""
    pipeline_output = run_pipeline()

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
