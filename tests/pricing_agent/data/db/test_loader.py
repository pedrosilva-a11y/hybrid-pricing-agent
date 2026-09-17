"""Tests for loading synthetic pipeline output into DuckDB."""

from pathlib import Path

import duckdb
import pytest

from pricing_agent.data.db.database import connect_database, initialize_database
from pricing_agent.data.db.loader import load_pipeline_output
from pricing_agent.data.pipeline import N_USERS, N_WEEKS, run_pipeline


def test_load_pipeline_output_populates_all_tables(tmp_path: Path) -> None:
    """Populate all DuckDB tables with generated pipeline data."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)
    output = run_pipeline()

    try:
        initialize_database(connection)
        load_pipeline_output(connection=connection, output=output)

        expected_counts = {
            "weekly_demand": N_WEEKS,
            "users": N_USERS,
            "weekly_price": N_WEEKS,
            "assigned_prices": N_USERS,
            "conversion_probabilities": N_USERS,
            "conversion_outcomes": N_USERS,
            "acquisition_costs": N_USERS,
            "marginal_costs": N_USERS,
            "churn_probabilities": N_USERS,
            "churn_outcomes": N_USERS,
        }

        for table, expected_count in expected_counts.items():
            result = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()

            assert result == (expected_count,)

    finally:
        connection.close()


def test_load_pipeline_output_preserves_user_values(tmp_path: Path) -> None:
    """Persist generated values without changing their contents."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)
    output = run_pipeline()

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


def test_load_pipeline_output_rolls_back_on_failure(tmp_path: Path) -> None:
    """Roll back all inserts when loading any table fails."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)
    output = run_pipeline()

    output["acquisition_costs"]["acquisition_cost"][0] = -1.0

    try:
        initialize_database(connection)

        with pytest.raises(duckdb.ConstraintException):
            load_pipeline_output(connection=connection, output=output)

        result = connection.execute("SELECT COUNT(*) FROM users").fetchone()

        assert result == (0,)

    finally:
        connection.close()
