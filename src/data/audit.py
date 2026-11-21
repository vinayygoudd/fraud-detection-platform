"""Dataset auditing, schema validation, and data quality reporting for PaySim."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.config import get_config

logger = logging.getLogger(__name__)

PAYSIM_LIMITATIONS = [
    "PaySim is a synthetic mobile-money simulation, not genuine production banking data.",
    "nameOrig and nameDest are simulated account identifiers.",
    "The step column represents 1-indexed hourly simulation ticks (1..743 = ~30 days), not real wall-clock timestamps.",
    "Transaction amounts are unitless simulator values and must NOT be assumed to represent INR or USD.",
    "Simulator-specific balance mechanics may create patterns not present in real banking systems.",
    "Post-transaction balance columns (newbalanceOrig, newbalanceDest) are excluded from serving features to avoid leakage.",
    "This platform is an educational/portfolio simulation, not a production-validated banking fraud system.",
]


def compute_file_sha256(file_path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a dataset file for MLflow provenance tracking."""
    hasher = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_paysim_schema(df: pd.DataFrame, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Validate required PaySim columns, dtypes, and categorical domain values."""
    cfg = config or get_config()
    expected_cols = cfg["data"]["expected_columns"]
    valid_types = set(cfg["data"]["valid_tx_types"])

    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    errors: list[str] = []
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "type" in df.columns:
        observed_types = set(df["type"].dropna().astype(str).unique())
        invalid_types = sorted(observed_types - valid_types)
        if invalid_types:
            errors.append(f"Unexpected transaction types found: {invalid_types}")

    if "amount" in df.columns:
        neg_amounts = int((pd.to_numeric(df["amount"], errors="coerce") < 0).sum())
        if neg_amounts > 0:
            errors.append(f"Found {neg_amounts} negative transaction amounts.")

    if "step" in df.columns:
        invalid_steps = int((pd.to_numeric(df["step"], errors="coerce") < 1).sum())
        if invalid_steps > 0:
            errors.append(f"Found {invalid_steps} rows with step < 1.")

    if "isFraud" in df.columns:
        invalid_labels = set(df["isFraud"].dropna().unique()) - {0, 1}
        if invalid_labels:
            errors.append(f"Invalid isFraud values: {sorted(invalid_labels)}")

    return {
        "is_valid": len(errors) == 0,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "errors": errors,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def audit_paysim_dataframe(df: pd.DataFrame, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run comprehensive data quality, missingness, duplicate, and fraud distribution audit."""
    schema_check = validate_paysim_schema(df, config=config)
    total_rows = int(len(df))
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pct = {
        col: round(float(cnt / total_rows * 100.0), 6) if total_rows > 0 else 0.0
        for col, cnt in missing_counts.items()
    }
    exact_duplicates = int(df.duplicated().sum())

    fraud_count = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0
    flagged_count = int(df["isFlaggedFraud"].sum()) if "isFlaggedFraud" in df.columns else 0
    fraud_rate = float(fraud_count / total_rows) if total_rows > 0 else 0.0

    fraud_by_type: Dict[str, Dict[str, Any]] = {}
    if "type" in df.columns and "isFraud" in df.columns:
        grouped = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
        for tx_type, row in grouped.iterrows():
            fraud_by_type[str(tx_type)] = {
                "total_count": int(row["count"]),
                "fraud_count": int(row["sum"]),
                "fraud_rate": round(float(row["mean"]), 6),
            }

    amount_stats: Dict[str, float] = {}
    if "amount" in df.columns and total_rows > 0:
        amt = pd.to_numeric(df["amount"], errors="coerce").dropna()
        amount_stats = {
            "min": float(amt.min()),
            "p25": float(np.percentile(amt, 25)),
            "median": float(amt.median()),
            "mean": float(amt.mean()),
            "p75": float(np.percentile(amt, 75)),
            "p95": float(np.percentile(amt, 95)),
            "p99": float(np.percentile(amt, 99)),
            "max": float(amt.max()),
        }

    step_stats: Dict[str, Any] = {}
    if "step" in df.columns and total_rows > 0:
        step_stats = {
            "min_step": int(df["step"].min()),
            "max_step": int(df["step"].max()),
            "unique_steps": int(df["step"].nunique()),
        }

    account_stats: Dict[str, Any] = {}
    if "nameOrig" in df.columns and total_rows > 0:
        orig_counts = df["nameOrig"].value_counts()
        account_stats = {
            "unique_name_orig": int(df["nameOrig"].nunique()),
            "unique_name_dest": int(df["nameDest"].nunique()) if "nameDest" in df.columns else 0,
            "repeat_originating_accounts": int((orig_counts > 1).sum()),
            "max_tx_per_orig_account": int(orig_counts.max()),
        }

    return {
        "schema_validation": schema_check,
        "row_count": total_rows,
        "column_count": int(df.shape[1]),
        "missing_values": {
            "counts": missing_counts,
            "percentages": missing_pct,
            "total_missing_cells": int(sum(missing_counts.values())),
        },
        "duplicate_rows": {
            "exact_duplicate_count": exact_duplicates,
            "exact_duplicate_pct": round(
                float(exact_duplicates / total_rows * 100.0), 6
            )
            if total_rows > 0
            else 0.0,
        },
        "fraud_distribution": {
            "fraud_count": fraud_count,
            "legitimate_count": total_rows - fraud_count,
            "fraud_rate": round(fraud_rate, 6),
            "imbalance_ratio_legit_to_fraud": round(
                float((total_rows - fraud_count) / fraud_count), 2
            )
            if fraud_count > 0
            else None,
            "flagged_fraud_count": flagged_count,
            "by_transaction_type": fraud_by_type,
        },
        "amount_summary_unitless": amount_stats,
        "step_summary": step_stats,
        "account_summary": account_stats,
        "limitations": PAYSIM_LIMITATIONS,
    }


def audit_paysim_csv(
    csv_path: str | Path,
    output_report_path: str | Path | None = None,
    max_rows: int | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Audit a PaySim CSV file on disk and optionally write a JSON data quality report."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place it in data/raw/."
        )

    logger.info("Reading PaySim CSV for audit: %s", path)
    df = pd.read_csv(path, nrows=max_rows)
    report = audit_paysim_dataframe(df, config=config)
    report["source_file"] = str(path)
    report["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
    report["sha256"] = compute_file_sha256(path)
    report["sampled_rows_only"] = max_rows is not None

    if output_report_path is not None:
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved data quality audit report to %s", out_path)

    return report
