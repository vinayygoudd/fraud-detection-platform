"""Integration tests for FastAPI /health, /model-info, and /predict endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from src.api.main import create_app
from src.models.registry import LoadedModelBundle


def test_health_and_model_info_endpoints(
    db_engine: Engine,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    app = create_app(db_engine=db_engine, model_bundle=trained_model_bundle, load_defaults=False)
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    assert h_data["status"] == "healthy"
    assert h_data["database_ready"] is True
    assert h_data["model_ready"] is True

    info_resp = client.get("/model-info")
    assert info_resp.status_code == 200
    i_data = info_resp.json()
    assert i_data["selected_model"] == "random_forest"
    assert i_data["model_version"] == "test-v1"
    assert i_data["thresholds"]["review_threshold"] == 0.30
    assert i_data["thresholds"]["block_threshold"] == 0.70
    assert len(i_data["raw_features"]) == 16


def test_predict_valid_cold_start_and_idempotency(
    db_engine: Engine,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    app = create_app(db_engine=db_engine, model_bundle=trained_model_bundle, load_defaults=False)
    client = TestClient(app)

    payload = {
        "transaction_id": "tx-idemp-001",
        "customer_id": "C_NEW_ACCOUNT",
        "amount": 1250.0,
        "tx_type": "TRANSFER",
        "step": 100,
    }
    resp1 = client.post("/predict", json=payload)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["transaction_id"] == "tx-idemp-001"
    assert 0.0 <= body1["risk_score"] <= 1.0
    assert body1["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert body1["is_cold_start_account"] is True
    assert body1["idempotent_replay"] is False
    assert 3 <= len(body1["top_explanations"]) <= 5

    # Exact retry must replay stored prediction without double-counting
    resp2 = client.post("/predict", json=payload)
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["idempotent_replay"] is True
    assert body2["risk_score"] == body1["risk_score"]
    assert body2["decision"] == body1["decision"]

    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM online_transactions")).scalar()
    assert count == 1

    # Conflicting payload with same transaction_id must return HTTP 409
    conflict_payload = {**payload, "amount": 9999.0}
    resp_conflict = client.post("/predict", json=conflict_payload)
    assert resp_conflict.status_code == 409


def test_predict_input_validation_errors(
    db_engine: Engine,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    app = create_app(db_engine=db_engine, model_bundle=trained_model_bundle, load_defaults=False)
    client = TestClient(app)

    # Negative amount
    r_amt = client.post(
        "/predict",
        json={"customer_id": "C1", "amount": -10.0, "tx_type": "PAYMENT", "step": 10},
    )
    assert r_amt.status_code == 422

    # Invalid step (< 1)
    r_step = client.post(
        "/predict",
        json={"customer_id": "C1", "amount": 10.0, "tx_type": "PAYMENT", "step": 0},
    )
    assert r_step.status_code == 422

    # Unsupported tx_type
    r_type = client.post(
        "/predict",
        json={"customer_id": "C1", "amount": 10.0, "tx_type": "CRYPTO_WIRE", "step": 10},
    )
    assert r_type.status_code == 422


def test_predict_model_unavailable_and_database_error(
    db_engine: Engine,
    trained_model_bundle: LoadedModelBundle,
) -> None:
    # Model unavailable -> 503
    app_no_model = create_app(db_engine=db_engine, model_bundle=None, load_defaults=False)
    client_no_model = TestClient(app_no_model)
    assert client_no_model.get("/health").json()["model_ready"] is False
    resp_no_model = client_no_model.post(
        "/predict",
        json={"customer_id": "C1", "amount": 100.0, "tx_type": "PAYMENT", "step": 10},
    )
    assert resp_no_model.status_code == 503

    # Database unavailable -> 503
    app_no_db = create_app(db_engine=None, model_bundle=trained_model_bundle, load_defaults=False)
    client_no_db = TestClient(app_no_db)
    assert client_no_db.get("/health").json()["database_ready"] is False
    resp_no_db = client_no_db.post(
        "/predict",
        json={"customer_id": "C1", "amount": 100.0, "tx_type": "PAYMENT", "step": 10},
    )
    assert resp_no_db.status_code == 503
