"""Tests for DuckDB database connection and schema initialization."""

from pathlib import Path

from pricing_agent.data.db.database import connect_database, initialize_database

EXPECTED_TABLES = {
    "weekly_demand",
    "users",
    "weekly_price",
    "assigned_prices",
    "conversion_probabilities",
    "conversion_outcomes",
    "acquisition_costs",
    "marginal_costs",
    "churn_probabilities",
    "churn_outcomes",
}


def test_connect_database_opens_working_connection(tmp_path: Path) -> None:
    """Open a working DuckDB connection."""
    database_path = tmp_path / "test.duckdb"

    connection = connect_database(database_path)

    try:
        result = connection.execute("SELECT 1").fetchone()

        assert result == (1,)

    finally:
        connection.close()


def test_initialize_database_creates_expected_tables(tmp_path: Path) -> None:
    """Create all tables defined by the database schema."""
    database_path = tmp_path / "test.duckdb"
    connection = connect_database(database_path)

    try:
        initialize_database(connection)

        rows = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
                AND table_type = 'BASE TABLE'
            """
        ).fetchall()

        table_names = {row[0] for row in rows}

        assert table_names == EXPECTED_TABLES

    finally:
        connection.close()
