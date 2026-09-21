"""Tests for loading synthetic pipeline output into DuckDB."""

from pathlib import Path

import duckdb
import pytest

from pricing_agent.data.config import (
    SUBSCRIPTION_TIERS,
    PricingDataConfig,
    build_pricing_data_config,
)
from pricing_agent.data.db.database import connect_database, initialize_database
from pricing_agent.data.db.loader import load_pipeline_output
from pricing_agent.data.pipeline import run_pipeline


@pytest.fixture
def config() -> PricingDataConfig:
    """Provide valid synthetic pricing data configuration."""
    return build_pricing_data_config(
        n_users=100,
        n_weeks=24,
        randomization_rate=0.1,
        seed=42,
    )


def test_load_pipeline_output_populates_all_tables(
    config: PricingDataConfig,
    tmp_path: Path,
) -> None:
    """Populate all DuckDB tables with generated pipeline data."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)

    output = run_pipeline(config)

    try:
        initialize_database(connection)
        load_pipeline_output(connection=connection, output=output)

        expected_counts = {
            "weekly_conditions": config.n_weeks,
            "users": config.n_users,
            "weekly_base_prices": config.n_weeks * len(SUBSCRIPTION_TIERS),
            "assigned_prices": config.n_users,
            "conversion_probabilities": config.n_users,
            "conversion_outcomes": config.n_users,
            "acquisition_costs": config.n_users,
            "marginal_costs": config.n_users,
            "churn_probabilities": config.n_users,
            "churn_outcomes": config.n_users,
        }

        for table, expected_count in expected_counts.items():
            result = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()

            assert result == (expected_count,)

    finally:
        connection.close()


def test_load_pipeline_output_preserves_user_values(
    config: PricingDataConfig,
    tmp_path: Path,
) -> None:
    """Persist generated user values without changing their contents."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)

    output = run_pipeline(config)

    try:
        initialize_database(connection)
        load_pipeline_output(connection=connection, output=output)

        user_row = connection.execute(
            """
            SELECT user_id, signup_week, segment, channel, tier
            FROM users
            WHERE user_id = ?
            """,
            [output["users"]["user_id"][0]],
        ).fetchone()

        assert user_row == (
            output["users"]["user_id"][0],
            output["users"]["signup_week"][0],
            output["users"]["segment"][0],
            output["users"]["channel"][0],
            output["users"]["tier"][0],
        )

    finally:
        connection.close()


def test_load_pipeline_output_rolls_back_on_failure(
    config: PricingDataConfig,
    tmp_path: Path,
) -> None:
    """Roll back all inserts when loading any table fails."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)

    output = run_pipeline(config)

    output["acquisition_costs"]["acquisition_cost"][0] = -1.0

    try:
        initialize_database(connection)

        with pytest.raises(duckdb.ConstraintException):
            load_pipeline_output(connection=connection, output=output)

        result = connection.execute("SELECT COUNT(*) FROM users").fetchone()

        assert result == (0,)

    finally:
        connection.close()
