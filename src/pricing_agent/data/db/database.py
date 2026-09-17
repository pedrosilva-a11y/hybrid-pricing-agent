"""DuckDB database connection and schema initialization utilities."""

from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection

ENCODING = "utf-8"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect_database(database_path: Path) -> DuckDBPyConnection:
    """Open a connection to a DuckDB database.

    Args:
        database_path: Path to the DuckDB database file.

    Returns:
        Open DuckDB connection.
    """
    return duckdb.connect(str(database_path))


def initialize_database(connection: DuckDBPyConnection) -> None:
    """Create database tables defined in the schema file.

    Args:
        connection: Open DuckDB connection.
    """
    schema_sql = SCHEMA_PATH.read_text(encoding=ENCODING)
    connection.execute(schema_sql)
