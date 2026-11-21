"""Model artifact persistence, metadata management, and reproducible serving bundle loader."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.evaluation.decision_engine import DecisionEngine, ThresholdConfig
from src.explainability.shap_explainer import FraudShapExplainer
from src.utils.config import get_config, get_project_root

logger = logging.getLogger(__name__)


@dataclass
class LoadedModelBundle:
    """Complete serving bundle loaded by the FastAPI application."""

    model_name: str
    model_version: str
    estimator: Any
    preprocessor: Any
    calibrator: Any
    is_calibrated: bool
    feature_names_raw: List[str]
    feature_names_transformed: List[str]
    thresholds: ThresholdConfig
    decision_engine: DecisionEngine
    shap_explainer: Optional[FraudShapExplainer]
    metadata: Dict[str, Any]

    def predict_risk_score(self, X_transformed: np.ndarray) -> float:
        """Return scalar risk score in [0, 1], calibrated if calibrator is attached."""
        raw_prob = float(self.estimator.predict_proba(X_transformed)[0, 1])
        if self.calibrator is not None and self.is_calibrated:
            return float(self.calibrator.transform(np.array([raw_prob]))[0])
        return float(np.clip(raw_prob, 0.0, 1.0))


def resolve_models_dir(artifacts_dir: str | Path | None = None) -> Path:
    """Resolve the directory where model artifacts and metadata are stored."""
    if artifacts_dir is not None:
        base = Path(artifacts_dir)
    else:
        cfg = get_config()
        base = Path(cfg["paths"]["models_dir"])
    if not base.is_absolute():
        base = get_project_root() / base
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_model_artifacts(
    models_dir: str | Path,
    selected_model_name: str,
    model_version: str,
    trained_estimators: Dict[str, Any],
    preprocessor: Any,
    calibrator: Any,
    background_matrix: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    """Persist individual candidate models, selected model, preprocessor, calibrator, and metadata."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {}
    for name, est in trained_estimators.items():
        path = out_dir / f"{name}.joblib"
        joblib.dump(est, path)
        saved_paths[f"model_{name}"] = str(path)

    selected_path = out_dir / "selected_model.joblib"
    joblib.dump(trained_estimators[selected_model_name], selected_path)
    saved_paths["selected_model"] = str(selected_path)

    preprocessor_path = out_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    saved_paths["preprocessor"] = str(preprocessor_path)

    if calibrator is not None:
        cal_path = out_dir / "calibrator.joblib"
        joblib.dump(calibrator, cal_path)
        saved_paths["calibrator"] = str(cal_path)

    bg_path = out_dir / "shap_background.npy"
    np.save(bg_path, np.asarray(background_matrix, dtype=np.float64))
    saved_paths["shap_background"] = str(bg_path)

    full_meta = {
        **metadata,
        "selected_model": selected_model_name,
        "model_version": model_version,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_files": {k: Path(v).name for k, v in saved_paths.items()},
    }
    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
    saved_paths["metadata"] = str(meta_path)

    logger.info(
        "Saved selected model (%s v%s) and artifacts to %s",
        selected_model_name,
        model_version,
        out_dir,
    )
    return saved_paths


def load_model_bundle(
    models_dir: str | Path | None = None,
    init_shap: bool = True,
) -> LoadedModelBundle:
    """Load the persisted selected model, preprocessor, calibrator, thresholds, and SHAP explainer."""
    dir_path = resolve_models_dir(models_dir)
    meta_path = dir_path / "model_metadata.json"
    model_path = dir_path / "selected_model.joblib"
    preproc_path = dir_path / "preprocessor.joblib"
    bg_path = dir_path / "shap_background.npy"
    cal_path = dir_path / "calibrator.joblib"

    for req in (meta_path, model_path, preproc_path):
        if not req.exists():
            raise FileNotFoundError(
                f"Required model artifact missing: {req}. Run `python scripts/train_pipeline.py` first."
            )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    estimator = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    is_calibrated = bool(metadata.get("is_calibrated", False)) and calibrator is not None

    t_cfg = metadata["thresholds"]
    thresholds = ThresholdConfig(
        review_threshold=float(t_cfg["review_threshold"]),
        block_threshold=float(t_cfg["block_threshold"]),
    )
    decision_engine = DecisionEngine(thresholds)

    feature_names_raw = list(metadata["feature_schema"]["raw_features"])
    feature_names_transformed = list(metadata["feature_schema"]["transformed_features"])

    shap_explainer: Optional[FraudShapExplainer] = None
    if init_shap and bg_path.exists():
        bg = np.load(bg_path)
        shap_explainer = FraudShapExplainer(
            estimator=estimator,
            model_type=str(metadata["selected_model"]),
            feature_names=feature_names_transformed,
            background_matrix=bg,
        )

    return LoadedModelBundle(
        model_name=str(metadata["selected_model"]),
        model_version=str(metadata["model_version"]),
        estimator=estimator,
        preprocessor=preprocessor,
        calibrator=calibrator,
        is_calibrated=is_calibrated,
        feature_names_raw=feature_names_raw,
        feature_names_transformed=feature_names_transformed,
        thresholds=thresholds,
        decision_engine=decision_engine,
        shap_explainer=shap_explainer,
        metadata=metadata,
    )
