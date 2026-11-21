"""Classification evaluation metrics and diagnostic plots for imbalanced fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _average_precision_fallback(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure-NumPy Average Precision (PR-AUC) fallback if sklearn is unavailable."""
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precisions = tp / ranks
    return float(np.sum(precisions * y_sorted) / n_pos)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute imbalanced fraud detection metrics with PR-AUC as the primary metric."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    y_pred = (y_s >= threshold).astype(int)

    tp = int(np.sum((y_t == 1) & (y_pred == 1)))
    fp = int(np.sum((y_t == 0) & (y_pred == 1)))
    tn = int(np.sum((y_t == 0) & (y_pred == 0)))
    fn = int(np.sum((y_t == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    brier = float(np.mean((y_s - y_t) ** 2)) if len(y_t) > 0 else 0.0

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        pr_auc = (
            float(average_precision_score(y_t, y_s))
            if len(np.unique(y_t)) > 1
            else 0.0
        )
        roc_auc = (
            float(roc_auc_score(y_t, y_s))
            if len(np.unique(y_t)) > 1
            else 0.0
        )
    except ImportError:
        pr_auc = _average_precision_fallback(y_t, y_s)
        roc_auc = 0.0

    return {
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "brier_score": round(brier, 6),
        "threshold_used": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {"total": int(len(y_t)), "fraud": int(y_t.sum()), "legit": int((y_t == 0).sum())},
    }


def save_pr_and_roc_curves(
    y_true: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    pr_output_path: str | Path,
    roc_output_path: str | Path,
) -> tuple[Path, Path]:
    """Generate and save Precision-Recall and ROC comparison curves."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    y_t = np.asarray(y_true, dtype=int)
    pr_path = Path(pr_output_path)
    roc_path = Path(roc_output_path)
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = float(y_t.mean()) if len(y_t) > 0 else 0.0
    for name, scores in model_scores.items():
        prec, rec, _ = precision_recall_curve(y_t, scores)
        ap = average_precision_score(y_t, scores)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {ap:.4f})")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"No-skill baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Primary Metric: PR-AUC)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in model_scores.items():
        fpr_arr, tpr_arr, _ = roc_curve(y_t, scores)
        auc_val = roc_auc_score(y_t, scores)
        ax.plot(fpr_arr, tpr_arr, lw=2, label=f"{name} (ROC-AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    return pr_path, roc_path
