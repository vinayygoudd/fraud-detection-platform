"""Chunked PaySim CSV ingestion into PostgreSQL `historical_transactions` table."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy import Engine, text

from src.data.audit import validate_paysim_schema
from src.utils.config import get_config
from src.utils.db import get_engine

logger = logging.getLogger(__name__)

CSV_TO_DB_COLUMN_MAP = {
    "step": "step",
    "type": "tx_type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

DB_INSERT_COLUMNS = list(CSV_TO_DB_COLUMN_MAP.values())


def prepare_paysim_chunk(chunk: pd.DataFrame, config: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Validate, rename, and type-cast a PaySim DataFrame chunk for PostgreSQL loading."""
    validation = validate_paysim_schema(chunk, config=config)
    if not validation["is_valid"]:
        raise ValueError(f"Invalid PaySim chunk: {validation['errors']}")

    renamed = chunk[list(CSV_TO_DB_COLUMN_MAP.keys())].rename(columns=CSV_TO_DB_COLUMN_MAP).copy()
    renamed["step"] = renamed["step"].astype(int)
    renamed["tx_type"] = renamed["tx_type"].astype(str).str.strip().str.upper()
    renamed["amount"] = renamed["amount"].astype(float)
    renamed["name_orig"] = renamed["name_orig"].astype(str).str.strip()
    renamed["name_dest"] = renamed["name_dest"].astype(str).str.strip()
    for col in ("old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest"):
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    renamed["is_fraud"] = renamed["is_fraud"].astype(int)
    renamed["is_flagged_fraud"] = renamed["is_flagged_fraud"].astype(int)
    return renamed[DB_INSERT_COLUMNS]


def load_paysim_to_postgres(
    csv_path: str | Path,
    engine: Engine | None = None,
    chunk_size: int | None = None,
    max_rows: int | None = None,
    truncate_first: bool = False,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Stream PaySim CSV rows into PostgreSQL `historical_transactions` using COPY or batch insert."""
    cfg = config or get_config()
    eng = engine or get_engine()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place it in data/raw/."
        )

    effective_chunk_size = chunk_size or int(cfg["data"].get("chunk_size", 100000))

    if truncate_first:
        logger.warning("Truncating historical_transactions before loading...")
        with eng.begin() as conn:
            conn.execute(text("TRUNCATE TABLE historical_transactions RESTART IDENTITY"))

    total_loaded = 0
    chunks_processed = 0
    is_postgres = eng.dialect.name == "postgresql"

    reader = pd.read_csv(path, chunksize=effective_chunk_size, nrows=max_rows)
    for chunk in reader:
        prepared = prepare_paysim_chunk(chunk, config=cfg)

        if is_postgres:
            # High-throughput PostgreSQL COPY via psycopg3 cursor
            raw_conn = eng.raw_connection()
            try:
                buffer = io.StringIO()
                prepared.to_csv(buffer, index=False, header=False, na_rep="\\N")
                buffer.seek(0)
                cols_sql = ", ".join(DB_INSERT_COLUMNS)
                with raw_conn.cursor() as cur:
                    if hasattr(cur, "copy"):
                        with cur.copy(
                            f"COPY historical_transactions ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
                        ) as copy:
                            copy.write(buffer.getvalue())
                    else:
                        cur.copy_expert(
                            f"COPY historical_transactions ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
                            buffer,
                        )
                raw_conn.commit()
            finally:
                raw_conn.close()
        else:
            prepared.to_sql(
                "historical_transactions",
                con=eng,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )

        total_loaded += len(prepared)
        chunks_processed += 1
        logger.info("Loaded chunk %d (%d total rows)", chunks_processed, total_loaded)

    return {
        "csv_path": str(path),
        "rows_loaded": total_loaded,
        "chunks_processed": chunks_processed,
        "truncated_first": truncate_first,
    }
