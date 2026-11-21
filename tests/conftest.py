"""Shared Pytest fixtures for database, SQL parity, ML pipeline, SHAP, and FastAPI tests."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.pool import StaticPool

from src.evaluation.decision_engine import DecisionEngine, ThresholdConfig
from src.explainability.shap_explainer import FraudShapExplainer
from src.features.preprocessing import build_preprocessor, enforce_feature_order, get_transformed_feature_names
from src.models.registry import LoadedModelBundle
from src.utils.config import get_config


def _create_sqlite_test_schema(engine: Engine) -> None:
    """Create SQLite-compatible mirror of sql/schema.sql for fast local unit tests."""
    ddl_statements = [
        """
        CREATE TABLE historical_transactions (
            tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER NOT NULL CHECK (step >= 1),
            tx_type VARCHAR(20) NOT NULL CHECK (tx_type IN ('PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN')),
            amount REAL NOT NULL CHECK (amount >= 0.0),
            name_orig VARCHAR(64) NOT NULL,
            old_balance_orig REAL,
            new_balance_orig REAL,
            name_dest VARCHAR(64) NOT NULL,
            old_balance_dest REAL,
            new_balance_dest REAL,
            is_fraud INTEGER NOT NULL CHECK (is_fraud IN (0, 1)),
            is_flagged_fraud INTEGER NOT NULL CHECK (is_flagged_fraud IN (0, 1)),
            loaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX idx_hist_tx_orig_step ON historical_transactions (name_orig, step)",
        """
        CREATE TABLE online_transactions (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id VARCHAR(128) NOT NULL UNIQUE,
            customer_id VARCHAR(64) NOT NULL,
            tx_type VARCHAR(20) NOT NULL CHECK (tx_type IN ('PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN')),
            amount REAL NOT NULL CHECK (amount > 0.0),
            step INTEGER NOT NULL CHECK (step >= 1),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
        "CREATE INDEX idx_online_tx_customer_step ON online_transactions (customer_id, step)",
        """
        CREATE TABLE prediction_logs (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id VARCHAR(128) NOT NULL UNIQUE REFERENCES online_transactions (transaction_id),
            customer_id VARCHAR(64) NOT NULL,
            step INTEGER NOT NULL,
            amount REAL NOT NULL,
            tx_type VARCHAR(20) NOT NULL,
            risk_score REAL NOT NULL,
            decision VARCHAR(16) NOT NULL,
            threshold_review REAL NOT NULL,
            threshold_block REAL NOT NULL,
            model_name VARCHAR(64) NOT NULL,
            model_version VARCHAR(64) NOT NULL,
            is_calibrated INTEGER NOT NULL DEFAULT 0,
            features_json TEXT NOT NULL,
            shap_explanations_json TEXT NOT NULL,
            latency_ms REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))


@pytest.fixture()
def db_engine() -> Engine:
    """Provide a clean test DB engine (PostgreSQL if TEST_DATABASE_URL is set, else in-memory SQLite)."""
    pg_url = os.getenv("TEST_DATABASE_URL")
    if pg_url:
        eng = create_engine(pg_url, future=True)
        schema_sql = (Path(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
        with eng.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS prediction_logs, online_transactions, historical_transactions CASCADE"))
            for stmt in [s.strip() for s in schema_sql.split(";") if s.strip()]:
                conn.execute(text(stmt))
        yield eng
        eng.dispose()
        return

    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)

    @event.listens_for(eng, "connect")
    def _register_math_funcs(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.create_function("LN", 1, lambda x: math.log(float(x)) if x is not None and float(x) > 0 else None)

    _create_sqlite_test_schema(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def synthetic_feature_df() -> pd.DataFrame:
    """Generate a small deterministic synthetic feature DataFrame for ML and SHAP unit tests."""
    rng = np.random.default_rng(42)
    n = 180
    steps = np.repeat(np.arange(1, 31), 6)
    tx_types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n)
    amounts = rng.lognormal(mean=7.5, sigma=1.4, size=n)
    is_fraud = (
        ((tx_types == "TRANSFER") | (tx_types == "CASH_OUT"))
        & (amounts > np.percentile(amounts, 72))
    ).astype(int)
    # Guarantee fraud cases exist in all chronological partitions
    is_fraud[::15] = 1

    df = pd.DataFrame(
        {
            "tx_id": np.arange(1, n + 1),
            "step": steps,
            "customer_id": [f"C{1000 + (i % 35)}" for i in range(n)],
            "is_fraud": is_fraud,
            "steps_since_last_tx": rng.choice([-1, 1, 2, 5, 12, 30], size=n),
            "tx_count_last_3_steps": rng.integers(0, 4, size=n),
            "tx_count_last_24_steps": rng.integers(0, 10, size=n),
            "tx_amount_sum_last_3_steps": rng.uniform(0, 50000, size=n),
            "tx_amount_sum_last_24_steps": rng.uniform(0, 150000, size=n),
            "hist_tx_count": rng.integers(0, 20, size=n),
            "hist_avg_amount": rng.uniform(100, 20000, size=n),
            "amount_vs_user_avg_ratio": rng.uniform(0.1, 8.0, size=n),
            "hist_max_amount": rng.uniform(500, 60000, size=n),
            "amount_vs_user_max_ratio": rng.uniform(0.05, 3.0, size=n),
            "high_risk_type_count_24_steps": rng.integers(0, 5, size=n),
            "step_hour_of_day": (steps - 1) % 24,
            "is_first_tx": rng.integers(0, 2, size=n),
            "amount": amounts,
            "log_amount": np.log1p(amounts),
            "tx_type": tx_types,
        }
    )
    return df


@pytest.fixture()
def trained_model_bundle(synthetic_feature_df: pd.DataFrame) -> LoadedModelBundle:
    """Train a lightweight RandomForest + Preprocessor + SHAP bundle on synthetic data."""
    from sklearn.ensemble import RandomForestClassifier

    cfg = get_config()
    X_raw = enforce_feature_order(synthetic_feature_df, cfg)
    y = synthetic_feature_df["is_fraud"].to_numpy()

    preprocessor = build_preprocessor(cfg, scale_numeric=True)
    X_trans = preprocessor.fit_transform(X_raw)
    transformed_names = get_transformed_feature_names(preprocessor)

    clf = RandomForestClassifier(
        n_estimators=25,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_trans, y)

    thresholds = ThresholdConfig(review_threshold=0.30, block_threshold=0.70)
    explainer = FraudShapExplainer(
        estimator=clf,
        model_type="random_forest",
        feature_names=transformed_names,
        background_matrix=X_trans[:40],
    )

    return LoadedModelBundle(
        model_name="random_forest",
        model_version="test-v1",
        estimator=clf,
        preprocessor=preprocessor,
        calibrator=None,
        is_calibrated=False,
        feature_names_raw=list(X_raw.columns),
        feature_names_transformed=transformed_names,
        thresholds=thresholds,
        decision_engine=DecisionEngine(thresholds),
        shap_explainer=explainer,
        metadata={
            "selected_model": "random_forest",
            "model_version": "test-v1",
            "is_calibrated": False,
            "calibration_method": None,
            "score_semantics": "uncalibrated_risk_score",
            "selection_criterion": "highest_validation_pr_auc",
            "thresholds": {"review_threshold": 0.30, "block_threshold": 0.70},
        },
    )
