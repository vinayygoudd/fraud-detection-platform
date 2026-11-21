"""Probability calibration diagnostics and Validation-only post-hoc calibrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute Brier score and Expected Calibration Error (ECE) across probability bins."""
    y_t = np.asarray(y_true, dtype=float)
    y_s = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return {"brier_score": 0.0, "ece": 0.0, "n_bins": n_bins, "bins": []}

    brier = float(np.mean((y_s - y_t) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_summary = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_s >= lo) & (y_s <= hi)
        else:
            mask = (y_s >= lo) & (y_s < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_pred = float(np.mean(y_s[mask]))
        avg_true = float(np.mean(y_t[mask]))
        ece += (count / n) * abs(avg_pred - avg_true)
        bins_summary.append(
            {
                "bin_lower": round(float(lo), 4),
                "bin_upper": round(float(hi), 4),
                "count": count,
                "mean_predicted_score": round(avg_pred, 6),
                "observed_fraud_rate": round(avg_true, 6),
            }
        )

    return {
        "brier_score": round(brier, 6),
        "ece": round(float(ece), 6),
        "n_bins": n_bins,
        "bins": bins_summary,
    }


class ScoreCalibrator:
    """Post-hoc 1D score calibrator fitted ONLY on Validation predictions.

    Supports isotonic regression or Platt scaling (sigmoid / logistic regression)
    so class-weighted models can be calibrated without refitting the base estimator
    or touching the Test set.
    """

    def __init__(self, method: Literal["isotonic", "sigmoid"] = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self._calibrator: Any = None
        self.is_fitted: bool = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "ScoreCalibrator":
        """Fit calibrator on Validation scores and labels."""
        scores = np.clip(np.asarray(raw_scores, dtype=float), 1e-7, 1.0 - 1e-7)
        y = np.asarray(y_true, dtype=int)

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            cal.fit(scores, y)
            self._calibrator = cal
        else:
            from sklearn.linear_model import LogisticRegression

            logits = np.log(scores / (1.0 - scores)).reshape(-1, 1)
            cal = LogisticRegression(C=1.0, solver="lbfgs")
            cal.fit(logits, y)
            self._calibrator = cal

        self.is_fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw model risk scores to calibrated scores in [0, 1]."""
        if not self.is_fitted or self._calibrator is None:
            raise RuntimeError("ScoreCalibrator must be fitted before calling transform().")
        scores = np.clip(np.asarray(raw_scores, dtype=float), 1e-7, 1.0 - 1e-7)
        if self.method == "isotonic":
            return np.clip(self._calibrator.predict(scores), 0.0, 1.0)
        logits = np.log(scores / (1.0 - scores)).reshape(-1, 1)
        return np.clip(self._calibrator.predict_proba(logits)[:, 1], 0.0, 1.0)


def save_calibration_plot(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray | None,
    output_path: str | Path,
    model_name: str = "Selected Model",
    n_bins: int = 10,
) -> Path:
    """Save reliability diagram comparing uncalibrated vs. calibrated scores."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_diag = compute_calibration_diagnostics(y_true, raw_scores, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    if raw_diag["bins"]:
        rx = [b["mean_predicted_score"] for b in raw_diag["bins"]]
        ry = [b["observed_fraud_rate"] for b in raw_diag["bins"]]
        ax.plot(
            rx,
            ry,
            marker="o",
            lw=2,
            label=f"{model_name} Raw (Brier={raw_diag['brier_score']:.4f}, ECE={raw_diag['ece']:.4f})",
        )

    if calibrated_scores is not None:
        cal_diag = compute_calibration_diagnostics(y_true, calibrated_scores, n_bins=n_bins)
        if cal_diag["bins"]:
            cx = [b["mean_predicted_score"] for b in cal_diag["bins"]]
            cy = [b["observed_fraud_rate"] for b in cal_diag["bins"]]
            ax.plot(
                cx,
                cy,
                marker="s",
                lw=2,
                label=f"{model_name} Calibrated (Brier={cal_diag['brier_score']:.4f}, ECE={cal_diag['ece']:.4f})",
            )

    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Observed Fraud Fraction")
    ax.set_title(f"Calibration Reliability Diagram — {model_name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
