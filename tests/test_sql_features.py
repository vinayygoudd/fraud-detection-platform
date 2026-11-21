"""Tests for PostgreSQL/SQL schema, temporal leakage protection, window boundaries, and offline/online parity."""

from __future__ import annotations

import math

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from src.features.extractor import extract_offline_features, extract_online_features
from src.utils.config import get_feature_names


def _seed_deterministic_history(engine: Engine) -> None:
    """Insert deterministic transactions testing LAG, 3-step, 24-step, same-step, and future-step rules."""
    rows = [
        # Account C_HIST:
        # step 490 (outside 24-step window of 520, inside lifetime)
        (490, "PAYMENT", 100.0, "C_HIST", 500.0, 400.0, "M1", 0.0, 0.0, 0, 0),
        # step 496 (exact lower boundary of 24-step window for step 520: 520 - 24 = 496)
        (496, "TRANSFER", 200.0, "C_HIST", 400.0, 200.0, "C2", 0.0, 200.0, 0, 0),
        # step 517 (exact lower boundary of 3-step window for step 520: 520 - 3 = 517)
        (517, "CASH_OUT", 300.0, "C_HIST", 200.0, 0.0, "C3", 0.0, 300.0, 0, 0),
        # step 519 (strictly previous step for step 520 -> LAG = 1)
        (519, "PAYMENT", 400.0, "C_HIST", 1000.0, 600.0, "M4", 0.0, 0.0, 0, 0),
        # TWO same-step transactions at step 520 (MUST NOT leak into each other!)
        (520, "TRANSFER", 500.0, "C_HIST", 600.0, 100.0, "C5", 0.0, 500.0, 1, 0),
        (520, "CASH_OUT", 1000.0, "C_HIST", 100.0, 0.0, "C6", 0.0, 1000.0, 1, 0),
        # Future transaction at step 525 (MUST NOT leak into step 520!)
        (525, "TRANSFER", 99999.0, "C_HIST", 99999.0, 0.0, "C7", 0.0, 99999.0, 1, 0),
        # Cold-start account C_COLD at step 520 (no prior history)
        (520, "PAYMENT", 50.0, "C_COLD", 500.0, 450.0, "M8", 0.0, 0.0, 0, 0),
    ]
    insert_sql = text(
        """
        INSERT INTO historical_transactions (
            step, tx_type, amount, name_orig, old_balance_orig, new_balance_orig,
            name_dest, old_balance_dest, new_balance_dest, is_fraud, is_flagged_fraud
        ) VALUES (
            :step, :tx_type, :amount, :name_orig, :old_balance_orig, :new_balance_orig,
            :name_dest, :old_balance_dest, :new_balance_dest, :is_fraud, :is_flagged_fraud
        )
        """
    )
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                insert_sql,
                {
                    "step": r[0],
                    "tx_type": r[1],
                    "amount": r[2],
                    "name_orig": r[3],
                    "old_balance_orig": r[4],
                    "new_balance_orig": r[5],
                    "name_dest": r[6],
                    "old_balance_dest": r[7],
                    "new_balance_dest": r[8],
                    "is_fraud": r[9],
                    "is_flagged_fraud": r[10],
                },
            )


def test_schema_constraints_and_duplicate_transaction_id(db_engine: Engine) -> None:
    """Verify CHECK and UNIQUE constraints on online_transactions."""
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO online_transactions (transaction_id, customer_id, tx_type, amount, step) "
                "VALUES ('dup-1', 'C1', 'PAYMENT', 25.0, 10)"
            )
        )

    # Duplicate transaction_id must fail UNIQUE constraint
    with pytest.raises(IntegrityError):
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO online_transactions (transaction_id, customer_id, tx_type, amount, step) "
                    "VALUES ('dup-1', 'C1', 'PAYMENT', 25.0, 10)"
                )
            )

    # Non-positive amount must fail CHECK constraint
    with pytest.raises(IntegrityError):
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO online_transactions (transaction_id, customer_id, tx_type, amount, step) "
                    "VALUES ('neg-1', 'C1', 'PAYMENT', -5.0, 10)"
                )
            )


def test_offline_sql_temporal_leakage_and_window_boundaries(db_engine: Engine) -> None:
    """Verify LAG, 3-step, 24-step, lifetime aggregates, same-step exclusion, and future exclusion."""
    _seed_deterministic_history(db_engine)
    df = extract_offline_features(engine=db_engine)

    # Inspect BOTH transactions for C_HIST at step 520
    step520_rows = df[(df["customer_id"] == "C_HIST") & (df["step"] == 520)].sort_values("amount")
    assert len(step520_rows) == 2

    # Both same-step rows MUST share identical strictly-earlier historical aggregates
    for _, row in step520_rows.iterrows():
        assert row["steps_since_last_tx"] == 1  # 520 - 519 (NOT 0 from same-step sibling!)
        assert row["tx_count_last_3_steps"] == 2  # steps 517, 519
        assert row["tx_amount_sum_last_3_steps"] == pytest.approx(700.0)  # 300 + 400
        assert row["tx_count_last_24_steps"] == 3  # steps 496, 517, 519 (490 excluded)
        assert row["tx_amount_sum_last_24_steps"] == pytest.approx(900.0)  # 200 + 300 + 400
        assert row["hist_tx_count"] == 4  # steps 490, 496, 517, 519 (step 520 & 525 excluded!)
        assert row["hist_avg_amount"] == pytest.approx(250.0)  # (100+200+300+400)/4
        assert row["hist_max_amount"] == pytest.approx(400.0)  # step 525 (99999) MUST be excluded
        assert row["high_risk_type_count_24_steps"] == 2  # TRANSFER@496 + CASH_OUT@517
        assert row["step_hour_of_day"] == (520 - 1) % 24
        assert row["is_first_tx"] == 0

    first_row = step520_rows.iloc[0]  # amount = 500
    assert first_row["amount_vs_user_avg_ratio"] == pytest.approx(500.0 / 250.0)
    assert first_row["amount_vs_user_max_ratio"] == pytest.approx(500.0 / 400.0)
    assert first_row["log_amount"] == pytest.approx(math.log1p(500.0))

    # Inspect cold-start account C_COLD
    cold_row = df[df["customer_id"] == "C_COLD"].iloc[0]
    assert cold_row["steps_since_last_tx"] == -1
    assert cold_row["hist_tx_count"] == 0
    assert cold_row["hist_avg_amount"] == pytest.approx(0.0)
    assert cold_row["amount_vs_user_avg_ratio"] == pytest.approx(0.0)
    assert cold_row["amount_vs_user_max_ratio"] == pytest.approx(0.0)
    assert cold_row["is_first_tx"] == 1


def test_offline_and_online_feature_parity(db_engine: Engine) -> None:
    """Assert bit-for-bit parity between sql/features.sql (offline) and sql/online_features.sql (online)."""
    _seed_deterministic_history(db_engine)
    offline_df = extract_offline_features(engine=db_engine)

    num_cols, cat_cols = get_feature_names()
    all_cols = num_cols + cat_cols

    # Check parity for C_HIST at step 520 (amount 500, TRANSFER) and C_COLD at step 520 (amount 50, PAYMENT)
    test_cases = [
        ("C_HIST", 520, 500.0, "TRANSFER"),
        ("C_HIST", 520, 1000.0, "CASH_OUT"),
        ("C_COLD", 520, 50.0, "PAYMENT"),
    ]
    for customer_id, step, amount, tx_type in test_cases:
        offline_match = offline_df[
            (offline_df["customer_id"] == customer_id)
            & (offline_df["step"] == step)
            & (offline_df["amount"] == amount)
            & (offline_df["tx_type"] == tx_type)
        ]
        assert len(offline_match) == 1
        offline_row = offline_match.iloc[0]

        online_feat = extract_online_features(
            customer_id=customer_id,
            step=step,
            amount=amount,
            tx_type=tx_type,
            engine=db_engine,
        )

        for col in all_cols:
            if col in num_cols:
                assert float(online_feat[col]) == pytest.approx(float(offline_row[col]), rel=1e-6), (
                    f"Parity mismatch on numeric feature '{col}' for {customer_id}@step={step}"
                )
            else:
                assert str(online_feat[col]) == str(offline_row[col]), (
                    f"Parity mismatch on categorical feature '{col}'"
                )
