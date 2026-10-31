"""Shared PostgreSQL connection handling."""

import psycopg
from psycopg.rows import dict_row

from app.config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


def get_db_connection() -> psycopg.Connection:
    """Open a connection using settings supplied by Docker Compose."""
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        sslmode="require",
        connect_timeout=5,
        row_factory=dict_row,
    )