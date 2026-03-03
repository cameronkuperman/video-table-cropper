"""Shared database helpers for the production Postgres-backed deployment."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

try:  # pragma: no cover - import depends on installed extras
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - local fallback path
    dict_row = None
    ConnectionPool = None

_db_pool: ConnectionPool | None = None


def get_database_url() -> str | None:
    value = os.environ.get("DATABASE_URL")
    return value.strip() if value else None


def database_enabled() -> bool:
    return bool(get_database_url())


def _require_pool_support() -> None:
    if ConnectionPool is None or dict_row is None:
        raise RuntimeError(
            "DATABASE_URL is configured, but psycopg_pool/psycopg are not installed. "
            "Install the web/worker Postgres dependencies first."
        )


def get_db_pool() -> ConnectionPool | None:
    global _db_pool
    database_url = get_database_url()
    if not database_url:
        return None

    _require_pool_support()
    if _db_pool is None:
        min_size = max(1, int(os.environ.get("DB_POOL_MIN_SIZE", "1") or "1"))
        max_size = max(min_size, int(os.environ.get("DB_POOL_MAX_SIZE", "5") or "5"))
        _db_pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={
                "autocommit": False,
                "row_factory": dict_row,
            },
        )
    return _db_pool


@contextmanager
def db_connection() -> Iterator[Any]:
    pool = get_db_pool()
    if pool is None:
        raise RuntimeError("DATABASE_URL is not configured")
    with pool.connection() as conn:
        yield conn


def db_healthcheck() -> dict[str, Any]:
    if not database_enabled():
        return {"enabled": False, "healthy": True, "error": None}

    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return {"enabled": True, "healthy": True, "error": None}
    except Exception as exc:  # pragma: no cover - runtime dependent
        return {"enabled": True, "healthy": False, "error": str(exc)}
