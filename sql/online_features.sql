-- ============================================================================
-- Online Real-Time Behavioural Feature Extraction (PostgreSQL)
-- ============================================================================
-- Evaluates a single incoming transaction (customer_id, step, amount, tx_type)
-- against strictly earlier transactions (`step < current_step`) across both:
--   - `historical_transactions` (immutable PaySim history)
--   - `online_transactions` (previously persisted API events)
--
-- TEMPORAL LEAKAGE PROTECTION:
--   - Strictly filters `step < current_step`, excluding same-step and future transactions.
--   - Must execute BEFORE persisting the incoming event into `online_transactions`.
--
-- PARITY GUARANTEE:
--   Every feature definition matches `sql/features.sql` identically, including
--   cold-start defaults (`steps_since_last_tx = -1`, `is_first_tx = 1`, ratios = 0.0).
-- ============================================================================

WITH eligible_history AS (
    SELECT step, amount, tx_type
    FROM historical_transactions
    WHERE name_orig = :customer_id
      AND step < CAST(:step AS INTEGER)
    UNION ALL
    SELECT step, amount, tx_type
    FROM online_transactions
    WHERE customer_id = :customer_id
      AND step < CAST(:step AS INTEGER)
),
agg AS (
    SELECT
        MAX(step) AS prev_step,
        COALESCE(SUM(CASE WHEN step >= (CAST(:step AS INTEGER) - 3) THEN 1 ELSE 0 END), 0) AS tx_count_last_3_steps,
        COALESCE(SUM(CASE WHEN step >= (CAST(:step AS INTEGER) - 24) THEN 1 ELSE 0 END), 0) AS tx_count_last_24_steps,
        COALESCE(SUM(CASE WHEN step >= (CAST(:step AS INTEGER) - 3) THEN amount ELSE 0.0 END), 0.0) AS tx_amount_sum_last_3_steps,
        COALESCE(SUM(CASE WHEN step >= (CAST(:step AS INTEGER) - 24) THEN amount ELSE 0.0 END), 0.0) AS tx_amount_sum_last_24_steps,
        COUNT(*) AS hist_tx_count,
        COALESCE(SUM(amount), 0.0) AS hist_amount_sum,
        COALESCE(MAX(amount), 0.0) AS hist_max_amount,
        COALESCE(
            SUM(
                CASE
                    WHEN step >= (CAST(:step AS INTEGER) - 24) AND tx_type IN ('TRANSFER', 'CASH_OUT') THEN 1
                    ELSE 0
                END
            ),
            0
        ) AS high_risk_type_count_24_steps
    FROM eligible_history
)
SELECT
    CASE
        WHEN agg.prev_step IS NULL THEN -1
        ELSE (CAST(:step AS INTEGER) - agg.prev_step)
    END AS steps_since_last_tx,
    CAST(agg.tx_count_last_3_steps AS INTEGER) AS tx_count_last_3_steps,
    CAST(agg.tx_count_last_24_steps AS INTEGER) AS tx_count_last_24_steps,
    CAST(agg.tx_amount_sum_last_3_steps AS DOUBLE PRECISION) AS tx_amount_sum_last_3_steps,
    CAST(agg.tx_amount_sum_last_24_steps AS DOUBLE PRECISION) AS tx_amount_sum_last_24_steps,
    CAST(agg.hist_tx_count AS INTEGER) AS hist_tx_count,
    CAST(
        CASE
            WHEN agg.hist_tx_count > 0 THEN agg.hist_amount_sum / agg.hist_tx_count
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS hist_avg_amount,
    CAST(
        CASE
            WHEN agg.hist_tx_count > 0 AND (agg.hist_amount_sum / agg.hist_tx_count) > 0.0
                THEN CAST(:amount AS DOUBLE PRECISION) / (agg.hist_amount_sum / agg.hist_tx_count)
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS amount_vs_user_avg_ratio,
    CAST(agg.hist_max_amount AS DOUBLE PRECISION) AS hist_max_amount,
    CAST(
        CASE
            WHEN agg.hist_max_amount > 0.0
                THEN CAST(:amount AS DOUBLE PRECISION) / agg.hist_max_amount
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS amount_vs_user_max_ratio,
    CAST(agg.high_risk_type_count_24_steps AS INTEGER) AS high_risk_type_count_24_steps,
    CAST((CAST(:step AS INTEGER) - 1) % 24 AS INTEGER) AS step_hour_of_day,
    CASE
        WHEN agg.hist_tx_count = 0 THEN 1
        ELSE 0
    END AS is_first_tx,
    CAST(:amount AS DOUBLE PRECISION) AS amount,
    CAST(LN(1.0 + CAST(:amount AS DOUBLE PRECISION)) AS DOUBLE PRECISION) AS log_amount,
    CAST(:tx_type AS VARCHAR(20)) AS tx_type
FROM agg;
