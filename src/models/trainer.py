"""Chronological ML training pipeline with Logistic Regression, Random Forest, XGBoost, and MLflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.calibration import (
    ScoreCalibrator,
    compute_calibration_diagnostics,
    save_calibration_plot,
)
from src.evaluation.cost_matrix import CostAssumptions, compute_decision_costs
from src.evaluation.decision_engine import (
    DecisionEngine,
    ThresholdConfig,
    optimize_thresholds_on_validation,
    run_cost_sensitivity_analysis,
)
from src.evaluation.metrics import compute_classification_metrics, save_pr_and_roc_curves
from src.explainability.shap_explainer import FraudShapExplainer
from src.features.preprocessing import (
    build_preprocessor,
    enforce_feature_order,
    get_transformed_feature_names,
)
from src.models.registry import save_model_artifacts
from src.utils.config import get_config, get_cost_matrix_config, get_project_root

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    step_col: str = "step",
    target_col: str = "is_fraud",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split transactions strictly chronologically by `step` so future steps never leak into past splits."""
    if df.empty:
        raise ValueError("Cannot split an empty feature DataFrame.")

    df_sorted = df.sort_values([step_col], kind="mergesort").reset_index(drop=True)
    steps = df_sorted[step_col].to_numpy()
    n = len(df_sorted)

    train_cut_idx = int(n * train_frac)
    val_cut_idx = int(n * (train_frac + val_frac))
    train_max_step = int(steps[min(train_cut_idx, n - 1)])
    val_max_step = int(steps[min(val_cut_idx, n - 1)])

    # Ensure strictly disjoint, non-empty Train / Validation / Test step boundaries
    max_step_overall = int(steps[-1])
    if val_max_step <= train_max_step or val_max_step >= max_step_overall:
        unique_steps = np.sort(df_sorted[step_col].unique())
        if len(unique_steps) < 3:
            raise ValueError("Dataset must contain at least 3 distinct steps for chronological splitting.")
        if val_max_step >= max_step_overall:
            val_max_step = int(unique_steps[-2])
        if train_max_step >= val_max_step:
            t_idx = max(0, int(len(unique_steps) * train_frac) - 1)
            t_idx = min(t_idx, len(unique_steps) - 3)
            v_idx = max(t_idx + 1, int(len(unique_steps) * (train_frac + val_frac)) - 1)
            v_idx = min(v_idx, len(unique_steps) - 2)
            train_max_step = int(unique_steps[t_idx])
            val_max_step = int(unique_steps[v_idx])

    train_df = df_sorted[df_sorted[step_col] <= train_max_step].copy()
    val_df = df_sorted[(df_sorted[step_col] > train_max_step) & (df_sorted[step_col] <= val_max_step)].copy()
    test_df = df_sorted[df_sorted[step_col] > val_max_step].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Chronological split produced an empty partition: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
        )

    split_meta = {
        "strategy": "chronological_by_step",
        "train_step_range": [int(train_df[step_col].min()), int(train_df[step_col].max())],
        "val_step_range": [int(val_df[step_col].min()), int(val_df[step_col].max())],
        "test_step_range": [int(test_df[step_col].min()), int(test_df[step_col].max())],
        "row_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "fraud_counts": {
            "train": int(train_df[target_col].sum()),
            "val": int(val_df[target_col].sum()),
            "test": int(test_df[target_col].sum()),
        },
        "fraud_rates": {
            "train": round(float(train_df[target_col].mean()), 6),
            "val": round(float(val_df[target_col].mean()), 6),
            "test": round(float(test_df[target_col].mean()), 6),
        },
    }
    return train_df, val_df, test_df, split_meta


def _build_candidate_models(
    y_train: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Instantiate Logistic Regression, Random Forest, and XGBoost with imbalance handling."""
    seed = int(cfg["project"].get("random_seed", 42))
    m_cfg = cfg["models"]

    n_pos = max(int(np.sum(y_train == 1)), 1)
    n_neg = max(int(np.sum(y_train == 0)), 1)
    scale_pos_weight = float(n_neg / n_pos)

    lr_cfg = m_cfg["logistic_regression"]
    rf_cfg = m_cfg["random_forest"]
    xgb_cfg = m_cfg["xgboost"]

    return {
        "logistic_regression": LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            solver=str(lr_cfg.get("solver", "lbfgs")),
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth", 12),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 10)),
            class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 300)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            scale_pos_weight=scale_pos_weight,
            eval_metric=str(xgb_cfg.get("eval_metric", "aucpr")),
            tree_method=str(xgb_cfg.get("tree_method", "hist")),
            n_jobs=int(xgb_cfg.get("n_jobs", -1)),
            random_state=seed,
        ),
    }


def train_and_evaluate(
    features_df: pd.DataFrame,
    config: Dict[str, Any] | None = None,
    cost_config: Dict[str, Any] | None = None,
    dataset_metadata: Dict[str, Any] | None = None,
    enable_mlflow: bool = True,
) -> Dict[str, Any]:
    """Run end-to-end chronological training, Validation selection, calibration, threshold tuning, SHAP, and Test eval."""
    cfg = config or get_config()
    cost_cfg = cost_config or get_cost_matrix_config()
    root = get_project_root()

    models_dir = root / cfg["paths"]["models_dir"]
    plots_dir = root / cfg["paths"]["plots_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    for d in (models_dir, plots_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_col = cfg["features"]["target_column"]
    step_col = cfg["features"]["step_column"]
    split_cfg = cfg["splitting"]

    train_df, val_df, test_df, split_meta = chronological_split(
        features_df,
        step_col=step_col,
        target_col=target_col,
        train_frac=float(split_cfg.get("train_fraction", 0.70)),
        val_frac=float(split_cfg.get("val_fraction", 0.15)),
    )
    logger.info("Chronological split completed: %s", split_meta["row_counts"])
    logger.info("Fraud counts per split: %s", split_meta["fraud_counts"])

    X_train_raw = enforce_feature_order(train_df, cfg)
    y_train = train_df[target_col].astype(int).to_numpy()

    X_val_raw = enforce_feature_order(val_df, cfg)
    y_val = val_df[target_col].astype(int).to_numpy()
    amounts_val = val_df["amount"].astype(float).to_numpy()

    X_test_raw = enforce_feature_order(test_df, cfg)
    y_test = test_df[target_col].astype(int).to_numpy()
    amounts_test = test_df["amount"].astype(float).to_numpy()

    # Fit preprocessor ONLY on training data
    preprocessor = build_preprocessor(cfg, scale_numeric=True)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    transformed_names = get_transformed_feature_names(preprocessor)

    candidates = _build_candidate_models(y_train, cfg)
    trained_estimators: Dict[str, Any] = {}
    val_scores_map: Dict[str, np.ndarray] = {}
    val_metrics_map: Dict[str, Dict[str, Any]] = {}

    for name, est in candidates.items():
        logger.info("Training candidate model: %s", name)
        est.fit(X_train, y_train)
        trained_estimators[name] = est
        scores_val = est.predict_proba(X_val)[:, 1]
        val_scores_map[name] = scores_val
        val_metrics_map[name] = compute_classification_metrics(y_val, scores_val, threshold=0.5)
        logger.info(
            "Validation %s -> PR-AUC=%.4f, ROC-AUC=%.4f, Brier=%.4f",
            name,
            val_metrics_map[name]["pr_auc"],
            val_metrics_map[name]["roc_auc"],
            val_metrics_map[name]["brier_score"],
        )

    # Select winning model strictly by Validation primary metric (PR-AUC)
    primary_metric = str(cfg["models"].get("primary_selection_metric", "pr_auc"))
    selected_model_name = max(
        val_metrics_map.keys(),
        key=lambda m: float(val_metrics_map[m][primary_metric]),
    )
    selected_estimator = trained_estimators[selected_model_name]
    logger.info(
        "Selected model based on Validation %s: %s (%.4f)",
        primary_metric,
        selected_model_name,
        val_metrics_map[selected_model_name][primary_metric],
    )

    # Optional post-hoc calibration fitted ONLY on Validation predictions
    cal_cfg = cfg["models"].get("calibration", {})
    cal_enabled = bool(cal_cfg.get("enabled", True))
    cal_method = str(cal_cfg.get("method", "isotonic"))
    cal_bins = int(cal_cfg.get("n_bins", 10))
    raw_val_selected = val_scores_map[selected_model_name]

    calibrator: ScoreCalibrator | None = None
    is_calibrated = False
    val_scores_for_decision = raw_val_selected

    if cal_enabled and len(np.unique(y_val)) > 1:
        calibrator = ScoreCalibrator(method=cal_method).fit(raw_val_selected, y_val)
        cal_val_scores = calibrator.transform(raw_val_selected)
        raw_diag = compute_calibration_diagnostics(y_val, raw_val_selected, n_bins=cal_bins)
        cal_diag = compute_calibration_diagnostics(y_val, cal_val_scores, n_bins=cal_bins)
        if cal_diag["brier_score"] <= raw_diag["brier_score"]:
            is_calibrated = True
            val_scores_for_decision = cal_val_scores
            logger.info(
                "Validation %s calibration improved Brier score: %.5f -> %.5f",
                cal_method,
                raw_diag["brier_score"],
                cal_diag["brier_score"],
            )
        else:
            calibrator = None
            logger.info("Calibration did not improve Brier score; retaining raw risk_score.")

    # Optimize (t1, t2) thresholds ONLY on Validation data
    cost_assumptions = CostAssumptions.from_config(cost_cfg)
    threshold_opt = optimize_thresholds_on_validation(
        y_val=y_val,
        scores_val=val_scores_for_decision,
        amounts_val=amounts_val,
        assumptions=cost_assumptions,
        cost_cfg=cost_cfg,
    )
    sensitivity = run_cost_sensitivity_analysis(
        y_val=y_val,
        scores_val=val_scores_for_decision,
        amounts_val=amounts_val,
        base_assumptions=cost_assumptions,
        cost_cfg=cost_cfg,
    )

    opt_t1 = threshold_opt["optimal_thresholds"]["review_threshold"]
    opt_t2 = threshold_opt["optimal_thresholds"]["block_threshold"]
    decision_engine = DecisionEngine(ThresholdConfig(review_threshold=opt_t1, block_threshold=opt_t2))

    # =========================================================================
    # FINAL UNTOUCHED TEST EVALUATION (executed once after all choices are fixed)
    # =========================================================================
    raw_test_scores = selected_estimator.predict_proba(X_test)[:, 1]
    final_test_scores = (
        calibrator.transform(raw_test_scores)
        if (is_calibrated and calibrator is not None)
        else raw_test_scores
    )
    test_metrics = compute_classification_metrics(y_test, final_test_scores, threshold=opt_t2)
    test_calibration_raw = compute_calibration_diagnostics(y_test, raw_test_scores, n_bins=cal_bins)
    test_calibration_final = compute_calibration_diagnostics(y_test, final_test_scores, n_bins=cal_bins)
    test_decisions = decision_engine.decide_batch(final_test_scores)
    test_cost_summary = compute_decision_costs(y_test, test_decisions, amounts_test, cost_assumptions)

    # Save diagnostic plots
    pr_plot, roc_plot = save_pr_and_roc_curves(
        y_true=y_val,
        model_scores=val_scores_map,
        pr_output_path=plots_dir / "validation_pr_curves.png",
        roc_output_path=plots_dir / "validation_roc_curves.png",
    )
    cal_plot = save_calibration_plot(
        y_true=y_test,
        raw_scores=raw_test_scores,
        calibrated_scores=final_test_scores if is_calibrated else None,
        output_path=plots_dir / "test_calibration_plot.png",
        model_name=selected_model_name,
        n_bins=cal_bins,
    )

    # Global SHAP plots and background sample for API local explanations
    rng = np.random.default_rng(int(cfg["project"].get("random_seed", 42)))
    bg_size = min(int(cfg["explainability"].get("background_sample_size", 200)), len(X_train))
    bg_idx = rng.choice(len(X_train), size=bg_size, replace=False)
    background_matrix = X_train[bg_idx]

    shap_explainer = FraudShapExplainer(
        estimator=selected_estimator,
        model_type=selected_model_name,
        feature_names=transformed_names,
        background_matrix=background_matrix,
    )
    shap_sample_n = min(int(cfg["explainability"].get("summary_plot_sample_size", 1000)), len(X_test))
    shap_idx = rng.choice(len(X_test), size=shap_sample_n, replace=False)
    shap_summary_plot, shap_bar_plot = shap_explainer.save_global_plots(
        X_sample_transformed=X_test[shap_idx],
        summary_plot_path=plots_dir / "shap_summary.png",
        bar_plot_path=plots_dir / "shap_importance_bar.png",
    )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "selected_model": selected_model_name,
        "model_version": model_version,
        "selection_criterion": f"highest_validation_{primary_metric}",
        "is_calibrated": is_calibrated,
        "calibration_method": cal_method if is_calibrated else None,
        "score_semantics": (
            "calibrated_probability_estimate" if is_calibrated else "uncalibrated_risk_score"
        ),
        "thresholds": {"review_threshold": opt_t1, "block_threshold": opt_t2},
        "feature_schema": {
            "raw_features": list(X_train_raw.columns),
            "transformed_features": transformed_names,
        },
        "split_metadata": split_meta,
        "dataset_metadata": dataset_metadata or {},
        "validation_metrics_all_models": val_metrics_map,
        "validation_threshold_optimization": threshold_opt,
        "cost_sensitivity_analysis": sensitivity,
        "test_metrics_selected_model": test_metrics,
        "test_calibration_raw": test_calibration_raw,
        "test_calibration_final": test_calibration_final,
        "test_cost_summary": test_cost_summary,
    }

    saved_artifacts = save_model_artifacts(
        models_dir=models_dir,
        selected_model_name=selected_model_name,
        model_version=model_version,
        trained_estimators=trained_estimators,
        preprocessor=preprocessor,
        calibrator=calibrator,
        background_matrix=background_matrix,
        metadata=metadata,
    )

    eval_report_path = reports_dir / "evaluation_report.json"
    eval_report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Optional MLflow tracking
    if enable_mlflow:
        _log_to_mlflow(
            cfg=cfg,
            metadata=metadata,
            selected_estimator=selected_estimator,
            plot_paths=[pr_plot, roc_plot, cal_plot, shap_summary_plot, shap_bar_plot],
            report_path=eval_report_path,
        )

    return {
        "selected_model": selected_model_name,
        "model_version": model_version,
        "thresholds": metadata["thresholds"],
        "validation_metrics": val_metrics_map,
        "test_metrics": test_metrics,
        "test_cost_summary": test_cost_summary,
        "saved_artifacts": saved_artifacts,
        "evaluation_report": str(eval_report_path),
    }


def _log_to_mlflow(
    cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    selected_estimator: Any,
    plot_paths: list[Path],
    report_path: Path,
) -> None:
    """Log parameters, metrics, plots, and model artifacts to MLflow."""
    try:
        import mlflow
        import mlflow.sklearn

        tracking_uri = cfg["paths"].get("mlflow_tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"{metadata['selected_model']}_{metadata['model_version']}"):
            mlflow.log_params(
                {
                    "selected_model": metadata["selected_model"],
                    "model_version": metadata["model_version"],
                    "is_calibrated": metadata["is_calibrated"],
                    "calibration_method": str(metadata["calibration_method"]),
                    "review_threshold_t1": metadata["thresholds"]["review_threshold"],
                    "block_threshold_t2": metadata["thresholds"]["block_threshold"],
                    "train_rows": metadata["split_metadata"]["row_counts"]["train"],
                    "val_rows": metadata["split_metadata"]["row_counts"]["val"],
                    "test_rows": metadata["split_metadata"]["row_counts"]["test"],
                    "dataset_sha256": str(metadata["dataset_metadata"].get("sha256", "unknown")),
                }
            )
            for model_name, m in metadata["validation_metrics_all_models"].items():
                mlflow.log_metrics(
                    {
                        f"val_{model_name}_pr_auc": m["pr_auc"],
                        f"val_{model_name}_roc_auc": m["roc_auc"],
                        f"val_{model_name}_f1": m["f1"],
                        f"val_{model_name}_brier": m["brier_score"],
                    }
                )
            tm = metadata["test_metrics_selected_model"]
            mlflow.log_metrics(
                {
                    "test_pr_auc": tm["pr_auc"],
                    "test_roc_auc": tm["roc_auc"],
                    "test_precision": tm["precision"],
                    "test_recall": tm["recall"],
                    "test_f1": tm["f1"],
                    "test_fpr": tm["fpr"],
                    "test_fnr": tm["fnr"],
                    "test_brier": tm["brier_score"],
                    "test_total_cost": metadata["test_cost_summary"]["total_cost"],
                }
            )
            for p in plot_paths:
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="plots")
            if report_path.exists():
                mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.sklearn.log_model(selected_estimator, artifact_path="selected_model")
    except Exception as exc:  # pragma: no cover - MLflow optional runtime guard
        logger.warning("MLflow logging failed (training artifacts still saved locally): %s", exc)
