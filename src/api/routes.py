"""FastAPI route handlers for /predict, /health, and /model-info."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import Engine, text

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ShapFeatureContribution,
    ThresholdInfo,
)
from src.features.extractor import extract_online_features, features_dict_to_dataframe
from src.models.registry import LoadedModelBundle
from src.utils.config import get_config
from src.utils.db import check_database_health

logger = logging.getLogger(__name__)
router = APIRouter()

SHAP_DISCLAIMER = (
    "SHAP attributions indicate statistical contribution to the model's risk_score "
    "and do not prove causation or confirm that a transaction is fraudulent."
)


def _get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database engine is not initialized.",
        )
    return engine


def _get_model_bundle(request: Request) -> LoadedModelBundle:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Run `python scripts/train_pipeline.py` first.",
        )
    return bundle


def _lookup_existing_prediction(engine: Engine, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an existing prediction_logs record by unique transaction_id for idempotency."""
    query = text(
        """
        SELECT
            transaction_id, customer_id, step, amount, tx_type,
            risk_score, decision, threshold_review, threshold_block,
            model_name, model_version, is_calibrated,
            features_json, shap_explanations_json, latency_ms
        FROM prediction_logs
        WHERE transaction_id = :transaction_id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"transaction_id": transaction_id}).mappings().first()
    return dict(row) if row is not None else None


def _persist_event_and_prediction(
    engine: Engine,
    transaction_id: str,
    payload: PredictRequest,
    risk_score: float,
    decision: str,
    bundle: LoadedModelBundle,
    features: Dict[str, Any],
    shap_payload: Dict[str, Any],
    latency_ms: float,
) -> None:
    """Persist online event and prediction log atomically AFTER feature extraction."""
    insert_tx = text(
        """
        INSERT INTO online_transactions (transaction_id, customer_id, tx_type, amount, step)
        VALUES (:transaction_id, :customer_id, :tx_type, :amount, :step)
        """
    )
    insert_log = text(
        """
        INSERT INTO prediction_logs (
            transaction_id, customer_id, step, amount, tx_type,
            risk_score, decision, threshold_review, threshold_block,
            model_name, model_version, is_calibrated,
            features_json, shap_explanations_json, latency_ms
        )
        VALUES (
            :transaction_id, :customer_id, :step, :amount, :tx_type,
            :risk_score, :decision, :threshold_review, :threshold_block,
            :model_name, :model_version, :is_calibrated,
            :features_json, :shap_explanations_json, :latency_ms
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            insert_tx,
            {
                "transaction_id": transaction_id,
                "customer_id": payload.customer_id,
                "tx_type": payload.tx_type,
                "amount": float(payload.amount),
                "step": int(payload.step),
            },
        )
        conn.execute(
            insert_log,
            {
                "transaction_id": transaction_id,
                "customer_id": payload.customer_id,
                "step": int(payload.step),
                "amount": float(payload.amount),
                "tx_type": payload.tx_type,
                "risk_score": float(risk_score),
                "decision": decision,
                "threshold_review": float(bundle.thresholds.review_threshold),
                "threshold_block": float(bundle.thresholds.block_threshold),
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
                "is_calibrated": bool(bundle.is_calibrated),
                "features_json": json.dumps(features),
                "shap_explanations_json": json.dumps(shap_payload),
                "latency_ms": float(latency_ms),
            },
        )


@router.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check(request: Request) -> HealthResponse:
    """Check PostgreSQL connectivity and ML model readiness."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        db_ok, db_detail = False, "Database engine not initialized"
    else:
        db_ok, db_detail = check_database_health(engine)

    bundle = getattr(request.app.state, "model_bundle", None)
    model_ok = bundle is not None
    model_detail = (
        f"{bundle.model_name} (v{bundle.model_version})"
        if model_ok
        else "Model artifacts not loaded"
    )

    overall = "healthy" if (db_ok and model_ok) else "degraded"
    return HealthResponse(
        status=overall,
        database_ready=db_ok,
        database_detail=db_detail,
        model_ready=model_ok,
        model_detail=model_detail,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(request: Request) -> ModelInfoResponse:
    """Return selected model version, feature schema, thresholds, calibration status, and metadata."""
    bundle = _get_model_bundle(request)
    meta = bundle.metadata
    return ModelInfoResponse(
        selected_model=bundle.model_name,
        model_version=bundle.model_version,
        is_calibrated=bundle.is_calibrated,
        calibration_method=meta.get("calibration_method"),
        score_semantics=meta.get("score_semantics", "uncalibrated_risk_score"),
        thresholds=ThresholdInfo(
            review_threshold=bundle.thresholds.review_threshold,
            block_threshold=bundle.thresholds.block_threshold,
        ),
        raw_features=bundle.feature_names_raw,
        transformed_features=bundle.feature_names_transformed,
        training_metadata={
            "selection_criterion": meta.get("selection_criterion"),
            "split_metadata": meta.get("split_metadata", {}),
            "test_metrics_selected_model": meta.get("test_metrics_selected_model", {}),
            "saved_at_utc": meta.get("saved_at_utc"),
        },
    )


@router.post("/predict", response_model=PredictResponse, tags=["Scoring"])
def predict_transaction(payload: PredictRequest, request: Request) -> PredictResponse:
    """Evaluate a transaction using PostgreSQL historical features, ML scoring, SHAP, and cost-sensitive decisioning.

    Duplicate policy:
      - If `transaction_id` already exists with matching payload fields, returns the stored
        prediction with `idempotent_replay=True` (HTTP 200) without double-counting.
      - If `transaction_id` already exists with DIFFERENT payload fields, returns HTTP 409 Conflict.
    """
    start_ts = time.perf_counter()
    bundle = _get_model_bundle(request)
    engine = _get_engine(request)
    cfg = get_config()

    customer_id = payload.customer_id.strip()
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="customer_id must not be blank.",
        )

    tx_id = (payload.transaction_id.strip() if payload.transaction_id else f"tx_{uuid.uuid4().hex}")

    # Step 1: Check duplicate / idempotency status
    try:
        existing = _lookup_existing_prediction(engine, tx_id)
    except Exception as exc:
        logger.exception("Database error while checking transaction_id=%s", tx_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error during idempotency check: {exc}",
        ) from exc

    if existing is not None:
        same_payload = (
            existing["customer_id"] == customer_id
            and int(existing["step"]) == int(payload.step)
            and abs(float(existing["amount"]) - float(payload.amount)) < 1e-6
            and str(existing["tx_type"]) == payload.tx_type
        )
        if not same_payload:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"transaction_id '{tx_id}' already exists with different transaction parameters."
                ),
            )

        stored_shap = existing["shap_explanations_json"]
        if isinstance(stored_shap, str):
            stored_shap = json.loads(stored_shap)
        stored_features = existing["features_json"]
        if isinstance(stored_features, str):
            stored_features = json.loads(stored_features)

        return PredictResponse(
            transaction_id=tx_id,
            customer_id=existing["customer_id"],
            step=int(existing["step"]),
            amount=float(existing["amount"]),
            tx_type=existing["tx_type"],
            risk_score=float(existing["risk_score"]),
            score_type=(
                "calibrated_probability_estimate"
                if existing["is_calibrated"]
                else "uncalibrated_risk_score"
            ),
            decision=existing["decision"],
            thresholds=ThresholdInfo(
                review_threshold=float(existing["threshold_review"]),
                block_threshold=float(existing["threshold_block"]),
            ),
            top_explanations=[
                ShapFeatureContribution(**item) for item in stored_shap.get("top_features", [])
            ],
            top_risk_increasing=[
                ShapFeatureContribution(**item)
                for item in stored_shap.get("top_risk_increasing", [])
            ],
            top_risk_decreasing=[
                ShapFeatureContribution(**item)
                for item in stored_shap.get("top_risk_decreasing", [])
            ],
            is_cold_start_account=bool(int(stored_features.get("is_first_tx", 0)) == 1),
            idempotent_replay=True,
            model_name=str(existing["model_name"]),
            model_version=str(existing["model_version"]),
            latency_ms=round(float(existing.get("latency_ms") or 0.0), 2),
            disclaimer=SHAP_DISCLAIMER,
        )

    # Step 2: Query PostgreSQL for strictly earlier historical features (step < payload.step)
    try:
        features = extract_online_features(
            customer_id=customer_id,
            step=payload.step,
            amount=payload.amount,
            tx_type=payload.tx_type,
            engine=engine,
            config=cfg,
        )
    except Exception as exc:
        logger.exception("Database error during online feature extraction for %s", customer_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error during historical feature extraction: {exc}",
        ) from exc

    # Step 3: Preprocess & score
    X_raw_df = features_dict_to_dataframe(features, config=cfg)
    X_transformed = bundle.preprocessor.transform(X_raw_df)
    risk_score = round(float(bundle.predict_risk_score(X_transformed)), 6)

    # Step 4: Local SHAP explanation (top 3-5 features)
    top_k = int(cfg["explainability"].get("top_k_features", 5))
    if bundle.shap_explainer is not None:
        shap_payload = bundle.shap_explainer.explain_local(X_transformed, top_k=top_k)
    else:
        shap_payload = {
            "top_features": [],
            "top_risk_increasing": [],
            "top_risk_decreasing": [],
            "disclaimer": SHAP_DISCLAIMER,
        }

    # Step 5: Three-way cost-sensitive decision
    decision = bundle.decision_engine.decide_one(risk_score)
    latency_ms = round((time.perf_counter() - start_ts) * 1000.0, 2)

    # Step 6: Persist event and prediction log AFTER feature extraction
    try:
        _persist_event_and_prediction(
            engine=engine,
            transaction_id=tx_id,
            payload=payload,
            risk_score=risk_score,
            decision=decision,
            bundle=bundle,
            features=features,
            shap_payload=shap_payload,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.exception("Database error persisting prediction for transaction_id=%s", tx_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error persisting prediction: {exc}",
        ) from exc

    return PredictResponse(
        transaction_id=tx_id,
        customer_id=customer_id,
        step=int(payload.step),
        amount=float(payload.amount),
        tx_type=payload.tx_type,
        risk_score=risk_score,
        score_type=(
            "calibrated_probability_estimate"
            if bundle.is_calibrated
            else "uncalibrated_risk_score"
        ),
        decision=decision,
        thresholds=ThresholdInfo(
            review_threshold=bundle.thresholds.review_threshold,
            block_threshold=bundle.thresholds.block_threshold,
        ),
        top_explanations=[
            ShapFeatureContribution(**item) for item in shap_payload.get("top_features", [])
        ],
        top_risk_increasing=[
            ShapFeatureContribution(**item) for item in shap_payload.get("top_risk_increasing", [])
        ],
        top_risk_decreasing=[
            ShapFeatureContribution(**item) for item in shap_payload.get("top_risk_decreasing", [])
        ],
        is_cold_start_account=bool(int(features.get("is_first_tx", 0)) == 1),
        idempotent_replay=False,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        latency_ms=latency_ms,
        disclaimer=SHAP_DISCLAIMER,
    )
