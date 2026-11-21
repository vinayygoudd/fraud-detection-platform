"""Tests for ML preprocessing, feature order, model artifact loading, and SHAP explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocessing import enforce_feature_order
from src.models.registry import LoadedModelBundle, load_model_bundle, save_model_artifacts
from src.utils.config import get_config


def test_preprocessing_and_feature_order_consistency(
    synthetic_feature_df: pd.DataFrame,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    """Verify feature order enforcement and transformed matrix shape."""
    cfg = get_config()
    shuffled_df = synthetic_feature_df[list(reversed(synthetic_feature_df.columns))].copy()
    ordered = enforce_feature_order(shuffled_df, cfg)
    assert list(ordered.columns) == trained_model_bundle.feature_names_raw

    X_trans = trained_model_bundle.preprocessor.transform(ordered.iloc[:5])
    assert X_trans.shape == (5, len(trained_model_bundle.feature_names_transformed))
    probs = trained_model_bundle.estimator.predict_proba(X_trans)[:, 1]
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_shap_output_shape_names_and_local_consistency(
    synthetic_feature_df: pd.DataFrame,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    """Verify SHAP output dimensions, feature names, and risk-increasing/decreasing direction labels."""
    cfg = get_config()
    X_raw = enforce_feature_order(synthetic_feature_df.iloc[:4], cfg)
    X_trans = trained_model_bundle.preprocessor.transform(X_raw)

    assert trained_model_bundle.shap_explainer is not None
    shap_matrix = trained_model_bundle.shap_explainer.compute_shap_values(X_trans)
    assert shap_matrix.shape == X_trans.shape
    assert np.all(np.isfinite(shap_matrix))

    local = trained_model_bundle.shap_explainer.explain_local(X_trans[:1], top_k=5)
    assert len(local["top_features"]) == 5

    valid_names = set(trained_model_bundle.feature_names_transformed)
    for item in local["top_features"]:
        assert item["feature_name"] in valid_names
        if item["shap_contribution"] > 0:
            assert item["direction"] == "increases_risk"
        elif item["shap_contribution"] < 0:
            assert item["direction"] == "decreases_risk"
        else:
            assert item["direction"] == "neutral"

    for item in local["top_risk_increasing"]:
        assert item["shap_contribution"] > 0
    for item in local["top_risk_decreasing"]:
        assert item["shap_contribution"] < 0


def test_model_artifact_save_and_reload(
    tmp_path: Path,
    synthetic_feature_df: pd.DataFrame,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    """Verify saved artifacts can be reloaded reproducibly and produce identical scores."""
    cfg = get_config()
    X_raw = enforce_feature_order(synthetic_feature_df.iloc[:2], cfg)
    X_trans = trained_model_bundle.preprocessor.transform(X_raw)
    original_score = trained_model_bundle.predict_risk_score(X_trans[:1])

    metadata = {
        **trained_model_bundle.metadata,
        "feature_schema": {
            "raw_features": trained_model_bundle.feature_names_raw,
            "transformed_features": trained_model_bundle.feature_names_transformed,
        },
    }
    save_model_artifacts(
        models_dir=tmp_path,
        selected_model_name=trained_model_bundle.model_name,
        model_version=trained_model_bundle.model_version,
        trained_estimators={trained_model_bundle.model_name: trained_model_bundle.estimator},
        preprocessor=trained_model_bundle.preprocessor,
        calibrator=None,
        background_matrix=X_trans,
        metadata=metadata,
    )

    reloaded = load_model_bundle(models_dir=tmp_path, init_shap=True)
    reloaded_score = reloaded.predict_risk_score(X_trans[:1])
    assert reloaded.model_name == trained_model_bundle.model_name
    assert reloaded.model_version == trained_model_bundle.model_version
    assert reloaded_score == original_score
