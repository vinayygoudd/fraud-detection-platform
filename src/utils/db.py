"""PostgreSQL database engine, session helpers, and SQL script execution utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

from src.utils.config import build_database_url, get_config, get_project_root

logger = logging.getLogger(__name__)

_ENGINE: Optional[Engine] = None


def get_engine(database_url: str | None = None, echo: bool | None = None) -> Engine:
    """Return a singleton SQLAlchemy engine (or create one for an explicit URL)."""
    global _ENGINE

    if database_url is not None:
        return create_engine(database_url, future=True, pool_pre_ping=True, echo=bool(echo))

    if _ENGINE is None:
        cfg = get_config()
        db_cfg = cfg["database"]
        url = build_database_url(cfg)
        _ENGINE = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
            pool_size=int(db_cfg.get("pool_size", 10)),
            max_overflow=int(db_cfg.get("max_overflow", 20)),
            echo=bool(db_cfg.get("echo", False)) if echo is None else echo,
        )
    return _ENGINE


def reset_engine() -> None:
    """Dispose and reset the cached singleton engine (useful in tests)."""
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None


@contextmanager
def get_connection(engine: Engine | None = None) -> Generator[Connection, None, None]:
    """Yield a transactional database connection."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        yield conn


def check_database_health(engine: Engine | None = None) -> tuple[bool, str]:
    """Execute a lightweight SELECT 1 probe to verify PostgreSQL readiness."""
    try:
        eng = engine or get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive runtime check
        logger.error("Database health check failed: %s", exc)
        return False, str(exc)


def load_sql_file(sql_filename: str) -> str:
    """Read a SQL script from the repository's sql/ directory."""
    sql_path = Path(sql_filename)
    if not sql_path.is_absolute():
        sql_path = get_project_root() / "sql" / sql_filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    return sql_path.read_text(encoding="utf-8")


def execute_sql_script(sql_script: str, engine: Engine | None = None) -> None:
    """Execute a multi-statement SQL script (e.g., sql/schema.sql)."""
    eng = engine or get_engine()
    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]
    with eng.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def execute_query_dict(
    sql_query: str,
    params: Dict[str, Any] | None = None,
    engine: Engine | None = None,
) -> list[Dict[str, Any]]:
    """Execute a parameterized SQL query and return rows as dictionaries."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        result = conn.execute(text(sql_query), params or {})
        return [dict(row._mapping) for row in result]
