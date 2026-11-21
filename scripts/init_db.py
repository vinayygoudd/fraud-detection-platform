"""Initialize PostgreSQL tables, constraints, and indexes from sql/schema.sql."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db import check_database_health, execute_sql_script, get_engine, load_sql_file
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging()
    logger = logging.getLogger("scripts.init_db")

    engine = get_engine()
    ok, detail = check_database_health(engine)
    if not ok:
        logger.error("Cannot connect to PostgreSQL: %s", detail)
        sys.exit(1)

    schema_sql = load_sql_file("schema.sql")
    execute_sql_script(schema_sql, engine=engine)
    logger.info("Successfully initialized PostgreSQL schema from sql/schema.sql.")


if __name__ == "__main__":
    main()
