"""Global and local SHAP explainability for preprocessed fraud risk models.

NOTE: SHAP values measure additive feature attribution relative to the model's
background expectation. They do NOT prove causation or prove that a transaction
is fraudulent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _extract_positive_class_shap(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output shapes across binary Tree/Linear explainers to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        # Older SHAP versions return [class_0_array, class_1_array]
        arr = np.asarray(shap_values[1], dtype=float)
    else:
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim == 3:
            # Newer SHAP versions return (n_samples, n_features, 2)
            arr = arr[:, :, 1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class FraudShapExplainer:
    """Wrapper providing consistent global and local SHAP explanations after preprocessing."""

    def __init__(
        self,
        estimator: Any,
        model_type: str,
        feature_names: List[str],
        background_matrix: np.ndarray,
    ) -> None:
        import shap

        self.estimator = estimator
        self.model_type = model_type
        self.feature_names = list(feature_names)
        self.background_matrix = np.asarray(background_matrix, dtype=float)

        if model_type in ("xgboost", "random_forest"):
            self._explainer = shap.TreeExplainer(estimator)
        elif model_type == "logistic_regression":
            self._explainer = shap.LinearExplainer(estimator, self.background_matrix)
        else:
            self._explainer = shap.Explainer(estimator, self.background_matrix)

    def compute_shap_values(self, X_transformed: np.ndarray) -> np.ndarray:
        """Compute positive-class SHAP values of shape (n_samples, n_features)."""
        X_arr = np.asarray(X_transformed, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        raw = self._explainer.shap_values(X_arr)
        values = _extract_positive_class_shap(raw)
        if values.shape != X_arr.shape:
            raise ValueError(
                f"SHAP output shape {values.shape} does not match input shape {X_arr.shape}."
            )
        return values

    def explain_local(
        self,
        X_row_transformed: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Return structured local explanation for one preprocessed transaction row."""
        row = np.asarray(X_row_transformed, dtype=float)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.shape[0] != 1:
            raise ValueError("explain_local expects a single transaction row.")

        shap_vec = self.compute_shap_values(row)[0]
        obs_vec = row[0]

        items: List[Dict[str, Any]] = []
        for feat_name, obs_val, sv in zip(self.feature_names, obs_vec, shap_vec):
            direction = (
                "increases_risk"
                if sv > 1e-12
                else ("decreases_risk" if sv < -1e-12 else "neutral")
            )
            items.append(
                {
                    "feature_name": str(feat_name),
                    "observed_value": round(float(obs_val), 6),
                    "shap_contribution": round(float(sv), 6),
                    "direction": direction,
                }
            )

        sorted_abs = sorted(items, key=lambda x: abs(x["shap_contribution"]), reverse=True)
        risk_increasing = [x for x in sorted_abs if x["direction"] == "increases_risk"][:top_k]
        risk_decreasing = [x for x in sorted_abs if x["direction"] == "decreases_risk"][:top_k]

        return {
            "top_features": sorted_abs[:top_k],
            "top_risk_increasing": risk_increasing,
            "top_risk_decreasing": risk_decreasing,
            "disclaimer": (
                "SHAP attributions reflect statistical contribution to the model's risk score "
                "and do not prove causation or confirm fraud."
            ),
        }

    def save_global_plots(
        self,
        X_sample_transformed: np.ndarray,
        summary_plot_path: str | Path,
        bar_plot_path: str | Path,
    ) -> tuple[Path, Path]:
        """Generate and save global SHAP beeswarm summary plot and mean-|SHAP| bar plot."""
        import shap

        X_arr = np.asarray(X_sample_transformed, dtype=float)
        shap_vals = self.compute_shap_values(X_arr)

        summary_path = Path(summary_plot_path)
        bar_path = Path(bar_plot_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        bar_path.parent.mkdir(parents=True, exist_ok=True)

        # Beeswarm summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            features=X_arr,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            features=X_arr,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Saved global SHAP plots to %s and %s", summary_path, bar_path)
        return summary_path, bar_path
