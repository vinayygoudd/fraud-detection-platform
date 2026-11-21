-- ============================================================================
-- Real-Time Fraud Detection & Risk Scoring Platform
-- PostgreSQL Database Schema, Constraints, and Indexes
-- ============================================================================
-- Design principles:
-- 1. Keep original PaySim historical training data (`historical_transactions`)
--    strictly separated from simulated real-time API events (`online_transactions`)
--    so live scoring cannot contaminate the historical training dataset.
-- 2. Enforce data integrity via CHECK and UNIQUE constraints.
-- 3. Support fast per-account temporal lookups with composite B-tree indexes.
-- ============================================================================

CREATE TABLE IF NOT EXISTS historical_transactions (
    tx_id            BIGSERIAL PRIMARY KEY,
    step             INTEGER NOT NULL CHECK (step >= 1),
    tx_type          VARCHAR(20) NOT NULL CHECK (
                         tx_type IN ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')
                     ),
    amount           DOUBLE PRECISION NOT NULL CHECK (amount >= 0.0),
    name_orig        VARCHAR(64) NOT NULL,
    old_balance_orig DOUBLE PRECISION,
    new_balance_orig DOUBLE PRECISION,
    name_dest        VARCHAR(64) NOT NULL,
    old_balance_dest DOUBLE PRECISION,
    new_balance_dest DOUBLE PRECISION,
    is_fraud         SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud SMALLINT NOT NULL CHECK (is_flagged_fraud IN (0, 1)),
    loaded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Composite index accelerating per-account temporal window scans (name_orig, step)
CREATE INDEX IF NOT EXISTS idx_hist_tx_orig_step
    ON historical_transactions (name_orig, step);

-- Index accelerating chronological Train / Validation / Test split queries
CREATE INDEX IF NOT EXISTS idx_hist_tx_step
    ON historical_transactions (step);

-- Index supporting fraud prevalence queries
CREATE INDEX IF NOT EXISTS idx_hist_tx_fraud
    ON historical_transactions (is_fraud);


CREATE TABLE IF NOT EXISTS online_transactions (
    event_id       BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(128) NOT NULL UNIQUE,
    customer_id    VARCHAR(64) NOT NULL,
    tx_type        VARCHAR(20) NOT NULL CHECK (
                       tx_type IN ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')
                   ),
    amount         DOUBLE PRECISION NOT NULL CHECK (amount > 0.0),
    step           INTEGER NOT NULL CHECK (step >= 1),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Composite index accelerating online per-customer historical lookups
CREATE INDEX IF NOT EXISTS idx_online_tx_customer_step
    ON online_transactions (customer_id, step);


CREATE TABLE IF NOT EXISTS prediction_logs (
    prediction_id         BIGSERIAL PRIMARY KEY,
    transaction_id        VARCHAR(128) NOT NULL UNIQUE
                          REFERENCES online_transactions (transaction_id)
                          ON DELETE CASCADE,
    customer_id           VARCHAR(64) NOT NULL,
    step                  INTEGER NOT NULL CHECK (step >= 1),
    amount                DOUBLE PRECISION NOT NULL CHECK (amount > 0.0),
    tx_type               VARCHAR(20) NOT NULL CHECK (
                              tx_type IN ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')
                          ),
    risk_score            DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    decision              VARCHAR(16) NOT NULL CHECK (decision IN ('APPROVE', 'REVIEW', 'BLOCK')),
    threshold_review      DOUBLE PRECISION NOT NULL CHECK (threshold_review > 0.0 AND threshold_review < 1.0),
    threshold_block       DOUBLE PRECISION NOT NULL CHECK (
                              threshold_block > threshold_review AND threshold_block < 1.0
                          ),
    model_name            VARCHAR(64) NOT NULL,
    model_version         VARCHAR(64) NOT NULL,
    is_calibrated         BOOLEAN NOT NULL DEFAULT FALSE,
    features_json         JSONB NOT NULL,
    shap_explanations_json JSONB NOT NULL,
    latency_ms            DOUBLE PRECISION,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_logs_customer_created
    ON prediction_logs (customer_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pred_logs_decision_created
    ON prediction_logs (decision, created_at DESC);
