-- ============================================================================
-- Offline Batch Behavioural Feature Engineering (PostgreSQL)
-- ============================================================================
-- TEMPORAL LEAKAGE PROTECTION:
--   PaySim provides hourly `step` integers without reliable within-step
--   transaction ordering. To guarantee zero same-step or future-step leakage:
--   1. `step_agg` first aggregates each account's activity at the (name_orig, step)
--      grain so every row in `step_history` has a strictly unique (name_orig, step).
--   2. `LAG(step)` over `step_agg` is therefore guaranteed to return a strictly
--      earlier step (`prev_step < current_step`), never a same-step sibling.
--   3. All rolling and lifetime windows use `RANGE BETWEEN ... PRECEDING AND 1 PRECEDING`,
--      which strictly excludes the current step and all future steps:
--        - Last 3 steps  : [step - 3, step - 1]
--        - Last 24 steps : [step - 24, step - 1]
--        - Full history  : (-infinity, step - 1]
--   4. Post-transaction balance columns (`new_balance_orig`, `new_balance_dest`)
--      and target/flag columns (`is_fraud`, `is_flagged_fraud`) are never used as
--      serving features.
--
-- PARITY GUARANTEE:
--   Mathematical definitions here are identical to `sql/online_features.sql`.
-- ============================================================================

WITH step_agg AS (
    SELECT
        name_orig,
        step,
        COUNT(*) AS step_tx_count,
        SUM(amount) AS step_amount_sum,
        MAX(amount) AS step_amount_max,
        SUM(
            CASE
                WHEN tx_type IN ('TRANSFER', 'CASH_OUT') THEN 1
                ELSE 0
            END
        ) AS step_high_risk_count
    FROM historical_transactions
    GROUP BY name_orig, step
),
step_history AS (
    SELECT
        name_orig,
        step,
        LAG(step) OVER (
            PARTITION BY name_orig
            ORDER BY step
        ) AS prev_step,
        COALESCE(
            SUM(step_tx_count) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS tx_count_last_3_steps,
        COALESCE(
            SUM(step_tx_count) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS tx_count_last_24_steps,
        COALESCE(
            SUM(step_amount_sum) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING
            ),
            0.0
        ) AS tx_amount_sum_last_3_steps,
        COALESCE(
            SUM(step_amount_sum) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            ),
            0.0
        ) AS tx_amount_sum_last_24_steps,
        COALESCE(
            SUM(step_tx_count) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS hist_tx_count,
        COALESCE(
            SUM(step_amount_sum) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0.0
        ) AS hist_amount_sum,
        COALESCE(
            MAX(step_amount_max) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            0.0
        ) AS hist_max_amount,
        COALESCE(
            SUM(step_high_risk_count) OVER (
                PARTITION BY name_orig
                ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            ),
            0
        ) AS high_risk_type_count_24_steps
    FROM step_agg
)
SELECT
    t.tx_id,
    t.step,
    t.name_orig AS customer_id,
    t.is_fraud,
    CASE
        WHEN sh.prev_step IS NULL THEN -1
        ELSE (t.step - sh.prev_step)
    END AS steps_since_last_tx,
    CAST(sh.tx_count_last_3_steps AS INTEGER) AS tx_count_last_3_steps,
    CAST(sh.tx_count_last_24_steps AS INTEGER) AS tx_count_last_24_steps,
    CAST(sh.tx_amount_sum_last_3_steps AS DOUBLE PRECISION) AS tx_amount_sum_last_3_steps,
    CAST(sh.tx_amount_sum_last_24_steps AS DOUBLE PRECISION) AS tx_amount_sum_last_24_steps,
    CAST(sh.hist_tx_count AS INTEGER) AS hist_tx_count,
    CAST(
        CASE
            WHEN sh.hist_tx_count > 0 THEN sh.hist_amount_sum / sh.hist_tx_count
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS hist_avg_amount,
    CAST(
        CASE
            WHEN sh.hist_tx_count > 0 AND (sh.hist_amount_sum / sh.hist_tx_count) > 0.0
                THEN t.amount / (sh.hist_amount_sum / sh.hist_tx_count)
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS amount_vs_user_avg_ratio,
    CAST(sh.hist_max_amount AS DOUBLE PRECISION) AS hist_max_amount,
    CAST(
        CASE
            WHEN sh.hist_max_amount > 0.0 THEN t.amount / sh.hist_max_amount
            ELSE 0.0
        END AS DOUBLE PRECISION
    ) AS amount_vs_user_max_ratio,
    CAST(sh.high_risk_type_count_24_steps AS INTEGER) AS high_risk_type_count_24_steps,
    CAST((t.step - 1) % 24 AS INTEGER) AS step_hour_of_day,
    CASE
        WHEN sh.hist_tx_count = 0 THEN 1
        ELSE 0
    END AS is_first_tx,
    CAST(t.amount AS DOUBLE PRECISION) AS amount,
    CAST(LN(1.0 + t.amount) AS DOUBLE PRECISION) AS log_amount,
    t.tx_type
FROM historical_transactions t
INNER JOIN step_history sh
    ON t.name_orig = sh.name_orig
   AND t.step = sh.step
WHERE (CAST(:min_step AS INTEGER) IS NULL OR t.step >= CAST(:min_step AS INTEGER))
  AND (CAST(:max_step AS INTEGER) IS NULL OR t.step <= CAST(:max_step AS INTEGER))
ORDER BY t.step ASC, t.tx_id ASC;
