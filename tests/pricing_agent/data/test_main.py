"""Integration tests for the synthetic pricing data entry point."""

import sys
from pathlib import Path

import pytest

import pricing_agent.data.main as main_module
from pricing_agent.data.db.database import connect_database


def test_main_generates_and_persists_configured_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate and persist data using command-line configuration."""
    database_path = tmp_path / "pricing.duckdb"

    monkeypatch.setattr(main_module, "DATABASE_PATH", database_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pricing-agent",
            "--n-users",
            "25",
            "--n-weeks",
            "8",
            "--randomization-rate",
            "0.2",
            "--seed",
            "123",
        ],
    )

    main_module.main()

    assert database_path.exists()

    connection = connect_database(database_path)

    try:
        users_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()

        weeks_count = connection.execute(
            "SELECT COUNT(*) FROM weekly_demand"
        ).fetchone()

        assert users_count == (25,)
        assert weeks_count == (8,)

    finally:
        connection.close()
