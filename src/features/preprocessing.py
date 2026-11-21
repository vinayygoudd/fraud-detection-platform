"""Scikit-learn preprocessing pipeline preserving deterministic feature order."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import get_config, get_feature_names


def build_preprocessor(
    config: Dict[str, Any] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer for the 15 numeric and 1 categorical features.

    Must be fitted ONLY on the chronological training split to prevent data leakage.
    """
    cfg = config or get_config()
    numeric_cols, cat_cols = get_feature_names(cfg)
    valid_tx_types = list(cfg["data"]["valid_tx_types"])

    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[valid_tx_types],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return exact output feature names after ColumnTransformer transformation."""
    return [str(name) for name in preprocessor.get_feature_names_out()]


def enforce_feature_order(
    df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Select and order raw input feature columns identically across train and serve."""
    numeric_cols, cat_cols = get_feature_names(config)
    ordered = numeric_cols + cat_cols
    missing = [col for col in ordered if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")

    out = df[ordered].copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
    for col in cat_cols:
        out[col] = out[col].astype(str)
    return out
