"""Unit tests for three-way decision engine boundaries, validation, and financial cost calculations."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.cost_matrix import (
    CostAssumptions,
    compute_baseline_approve_all_cost,
    compute_decision_costs,
)
from src.evaluation.decision_engine import (
    DecisionEngine,
    ThresholdConfig,
    optimize_thresholds_on_validation,
)


def test_decision_engine_exact_boundaries() -> None:
    """Verify exact boundary rules: < t1 APPROVE, t1 <= score < t2 REVIEW, >= t2 BLOCK."""
    engine = DecisionEngine(ThresholdConfig(review_threshold=0.25, block_threshold=0.75))

    assert engine.decide_one(0.0) == "APPROVE"
    assert engine.decide_one(0.249999) == "APPROVE"
    assert engine.decide_one(0.25) == "REVIEW"
    assert engine.decide_one(0.50) == "REVIEW"
    assert engine.decide_one(0.749999) == "REVIEW"
    assert engine.decide_one(0.75) == "BLOCK"
    assert engine.decide_one(1.0) == "BLOCK"

    batch = engine.decide_batch(np.array([0.10, 0.25, 0.74, 0.75, 0.99]))
    assert list(batch) == ["APPROVE", "REVIEW", "REVIEW", "BLOCK", "BLOCK"]


@pytest.mark.parametrize(
    "t1,t2",
    [
        (0.0, 0.5),
        (-0.1, 0.5),
        (0.5, 0.5),
        (0.7, 0.3),
        (0.3, 1.0),
        (0.3, 1.2),
    ],
)
def test_invalid_threshold_configurations_raise(t1: float, t2: float) -> None:
    """Verify 0 < t1 < t2 < 1 invariant is strictly enforced."""
    with pytest.raises(ValueError):
        ThresholdConfig(review_threshold=t1, block_threshold=t2)


def test_financial_cost_calculation_and_validation_optimization() -> None:
    """Verify deterministic financial cost math and Validation threshold optimization."""
    assumptions = CostAssumptions(
        false_negative_fixed=25.0,
        false_negative_amount_fraction=1.0,
        false_positive_block_fixed=35.0,
        false_positive_block_amount_fraction=0.02,
        manual_review_fixed=10.0,
        review_fraud_slippage_fraction=0.15,
        review_legit_friction_fixed=3.0,
        true_positive_block_fixed=1.0,
        true_negative_approve_fixed=0.0,
    )

    y_true = np.array([1, 0, 1, 0, 1, 0])
    decisions = np.array(["APPROVE", "BLOCK", "REVIEW", "REVIEW", "BLOCK", "APPROVE"], dtype=object)
    amounts = np.array([1000.0, 500.0, 2000.0, 100.0, 3000.0, 200.0])

    res = compute_decision_costs(y_true, decisions, amounts, assumptions)
    expected_fn = 25.0 + 1.0 * 1000.0           # 1025.0
    expected_fp = 35.0 + 0.02 * 500.0           # 45.0
    expected_rev_fraud = 10.0 + 0.15 * 2000.0   # 310.0
    expected_rev_legit = 10.0 + 3.0             # 13.0
    expected_tp = 1.0                           # 1.0
    expected_tn = 0.0                           # 0.0
    expected_total = expected_fn + expected_fp + expected_rev_fraud + expected_rev_legit + expected_tp + expected_tn

    assert res["total_cost"] == pytest.approx(expected_total)
    assert res["counts"] == {"APPROVE": 2, "REVIEW": 2, "BLOCK": 2}

    # Verify optimizer beats approve-all baseline on separable validation scores
    y_val = np.array([0] * 90 + [1] * 10)
    scores_val = np.concatenate([np.linspace(0.01, 0.25, 90), np.linspace(0.70, 0.98, 10)])
    amounts_val = np.full(100, 5000.0)

    baseline = compute_baseline_approve_all_cost(y_val, amounts_val, assumptions)
    opt = optimize_thresholds_on_validation(y_val, scores_val, amounts_val, assumptions=assumptions)
    assert opt["validation_cost_summary"]["total_cost"] < baseline["total_cost"]
    assert 0.0 < opt["optimal_thresholds"]["review_threshold"] < opt["optimal_thresholds"]["block_threshold"] < 1.0
