"""Pydantic request and response schemas for the Fraud Detection FastAPI service."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ValidTxType = Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
DecisionType = Literal["APPROVE", "REVIEW", "BLOCK"]


class PredictRequest(BaseModel):
    """Incoming transaction scoring payload.

    Historical behavioural features are NEVER supplied by the caller; they are
    computed server-side from PostgreSQL using strictly earlier steps.
    """

    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Originating account identifier (e.g., PaySim nameOrig such as 'C1231006815').",
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="Positive transaction amount (unitless PaySim scale).",
    )
    tx_type: ValidTxType = Field(
        ...,
        description="Transaction type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, or CASH_IN.",
    )
    step: int = Field(
        ...,
        ge=1,
        le=100000,
        description="Hourly simulation step index (>= 1).",
    )
    transaction_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional client-supplied idempotency key for deduplicating retries.",
    )


class ShapFeatureContribution(BaseModel):
    """Single feature attribution item in a local SHAP explanation."""

    feature_name: str
    observed_value: float
    shap_contribution: float
    direction: Literal["increases_risk", "decreases_risk", "neutral"]


class ThresholdInfo(BaseModel):
    """Active three-way decision thresholds (0 < review_threshold < block_threshold < 1)."""

    review_threshold: float
    block_threshold: float


class PredictResponse(BaseModel):
    """Structured real-time fraud risk scoring response."""

    transaction_id: str
    customer_id: str
    step: int
    amount: float
    tx_type: ValidTxType
    risk_score: float = Field(..., ge=0.0, le=1.0)
    score_type: str
    decision: DecisionType
    thresholds: ThresholdInfo
    top_explanations: List[ShapFeatureContribution]
    top_risk_increasing: List[ShapFeatureContribution]
    top_risk_decreasing: List[ShapFeatureContribution]
    is_cold_start_account: bool
    idempotent_replay: bool = False
    model_name: str
    model_version: str
    latency_ms: float
    disclaimer: str


class HealthResponse(BaseModel):
    """Service readiness status for database and ML model artifacts."""

    status: Literal["healthy", "degraded"]
    database_ready: bool
    database_detail: str
    model_ready: bool
    model_detail: str


class ModelInfoResponse(BaseModel):
    """Metadata describing the currently loaded model, schema, thresholds, and calibration."""

    selected_model: str
    model_version: str
    is_calibrated: bool
    calibration_method: Optional[str] = None
    score_semantics: str
    thresholds: ThresholdInfo
    raw_features: List[str]
    transformed_features: List[str]
    training_metadata: Dict[str, Any]
