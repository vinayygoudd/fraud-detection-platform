"""Offline batch and online real-time PostgreSQL behavioural feature extractor."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
from sqlalchemy import Engine, text

from src.utils.config import get_config, get_feature_names
from src.utils.db import get_engine, load_sql_file

logger = logging.getLogger(__name__)


def extract_offline_features(
    engine: Engine | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Execute `sql/features.sql` in PostgreSQL and return the canonical feature DataFrame."""
    cfg = config or get_config()
    eng = engine or get_engine()
    sql_query = load_sql_file("features.sql")

    logger.info(
        "Extracting offline behavioural features from PostgreSQL (min_step=%s, max_step=%s)",
        min_step,
        max_step,
    )
    with eng.connect() as conn:
        df = pd.read_sql_query(
            text(sql_query),
            con=conn,
            params={"min_step": min_step, "max_step": max_step},
        )

    numeric_cols, cat_cols = get_feature_names(cfg)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    logger.info("Extracted %d offline feature rows.", len(df))
    return df


def extract_online_features(
    customer_id: str,
    step: int,
    amount: float,
    tx_type: str,
    engine: Engine | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute `sql/online_features.sql` in PostgreSQL for one incoming transaction.

    Strictly queries `step < :step` before the incoming event is persisted so the
    transaction never leaks into its own historical aggregates.
    """
    cfg = config or get_config()
    eng = engine or get_engine()
    sql_query = load_sql_file("online_features.sql")

    with eng.connect() as conn:
        row = (
            conn.execute(
                text(sql_query),
                {
                    "customer_id": customer_id,
                    "step": int(step),
                    "amount": float(amount),
                    "tx_type": str(tx_type).upper(),
                },
            )
            .mappings()
            .first()
        )

    if row is None:  # pragma: no cover - aggregate query always returns 1 row
        raise RuntimeError("Online feature query returned no rows.")

    numeric_cols, cat_cols = get_feature_names(cfg)
    features: Dict[str, Any] = {}
    for col in numeric_cols:
        val = row[col]
        features[col] = float(val) if val is not None else 0.0
    for col in cat_cols:
        features[col] = str(row[col])
    return features


def features_dict_to_dataframe(
    features: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert a single online feature dictionary into a 1-row DataFrame in canonical order."""
    numeric_cols, cat_cols = get_feature_names(config)
    ordered_cols = numeric_cols + cat_cols
    missing = [c for c in ordered_cols if c not in features]
    if missing:
        raise ValueError(f"Missing required features for inference: {missing}")

    record = {col: features[col] for col in ordered_cols}
    return pd.DataFrame([record], columns=ordered_cols)
