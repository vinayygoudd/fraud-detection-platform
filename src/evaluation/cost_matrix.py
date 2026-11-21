"""Configurable financial cost matrix for three-way fraud decisioning.

All cost figures are illustrative cost units (PaySim amounts are unitless synthetic
simulator values and must NOT be assumed to represent INR or USD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.utils.config import get_cost_matrix_config


@dataclass(frozen=True)
class CostAssumptions:
    """Illustrative financial cost parameters for APPROVE / REVIEW / BLOCK outcomes."""

    false_negative_fixed: float = 25.0
    false_negative_amount_fraction: float = 1.0
    false_positive_block_fixed: float = 35.0
    false_positive_block_amount_fraction: float = 0.02
    manual_review_fixed: float = 10.0
    review_fraud_slippage_fraction: float = 0.15
    review_legit_friction_fixed: float = 3.0
    true_positive_block_fixed: float = 1.0
    true_negative_approve_fixed: float = 0.0
    currency_unit: str = "illustrative_cost_units"

    @classmethod
    def from_config(cls, cost_cfg: Dict[str, Any] | None = None) -> "CostAssumptions":
        cfg = cost_cfg or get_cost_matrix_config()
        c = cfg.get("costs", cfg)
        return cls(
            false_negative_fixed=float(c.get("false_negative_fixed", 25.0)),
            false_negative_amount_fraction=float(c.get("false_negative_amount_fraction", 1.0)),
            false_positive_block_fixed=float(c.get("false_positive_block_fixed", 35.0)),
            false_positive_block_amount_fraction=float(
                c.get("false_positive_block_amount_fraction", 0.02)
            ),
            manual_review_fixed=float(c.get("manual_review_fixed", 10.0)),
            review_fraud_slippage_fraction=float(c.get("review_fraud_slippage_fraction", 0.15)),
            review_legit_friction_fixed=float(c.get("review_legit_friction_fixed", 3.0)),
            true_positive_block_fixed=float(c.get("true_positive_block_fixed", 1.0)),
            true_negative_approve_fixed=float(c.get("true_negative_approve_fixed", 0.0)),
            currency_unit=str(cfg.get("currency_unit", "illustrative_cost_units")),
        )


def compute_decision_costs(
    y_true: np.ndarray,
    decisions: np.ndarray,
    amounts: np.ndarray,
    assumptions: CostAssumptions,
) -> Dict[str, Any]:
    """Evaluate total and per-outcome financial cost for APPROVE / REVIEW / BLOCK decisions."""
    y_t = np.asarray(y_true, dtype=int)
    dec = np.asarray(decisions, dtype=object)
    amt = np.asarray(amounts, dtype=float)

    if not (len(y_t) == len(dec) == len(amt)):
        raise ValueError("y_true, decisions, and amounts must have equal length.")

    n = len(y_t)
    if n == 0:
        return {
            "total_cost": 0.0,
            "avg_cost_per_tx": 0.0,
            "currency_unit": assumptions.currency_unit,
            "counts": {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0},
            "review_rate": 0.0,
            "block_rate": 0.0,
            "cost_breakdown": {},
        }

    is_fraud = y_t == 1
    is_legit = y_t == 0
    is_approve = dec == "APPROVE"
    is_review = dec == "REVIEW"
    is_block = dec == "BLOCK"

    # Fraud approved (False Negative)
    fn_mask = is_fraud & is_approve
    fn_cost = float(
        np.sum(
            assumptions.false_negative_fixed
            + assumptions.false_negative_amount_fraction * amt[fn_mask]
        )
    )

    # Legitimate blocked (False Positive)
    fp_mask = is_legit & is_block
    fp_cost = float(
        np.sum(
            assumptions.false_positive_block_fixed
            + assumptions.false_positive_block_amount_fraction * amt[fp_mask]
        )
    )

    # Fraud sent to review
    rev_fraud_mask = is_fraud & is_review
    rev_fraud_cost = float(
        np.sum(
            assumptions.manual_review_fixed
            + assumptions.review_fraud_slippage_fraction * amt[rev_fraud_mask]
        )
    )

    # Legitimate sent to review
    rev_legit_mask = is_legit & is_review
    rev_legit_cost = float(
        np.sum(
            (assumptions.manual_review_fixed + assumptions.review_legit_friction_fixed)
            * np.ones(np.sum(rev_legit_mask))
        )
    )

    # Fraud blocked (True Positive)
    tp_mask = is_fraud & is_block
    tp_cost = float(np.sum(tp_mask) * assumptions.true_positive_block_fixed)

    # Legitimate approved (True Negative)
    tn_mask = is_legit & is_approve
    tn_cost = float(np.sum(tn_mask) * assumptions.true_negative_approve_fixed)

    total_cost = fn_cost + fp_cost + rev_fraud_cost + rev_legit_cost + tp_cost + tn_cost

    return {
        "total_cost": round(total_cost, 4),
        "avg_cost_per_tx": round(total_cost / n, 6),
        "currency_unit": assumptions.currency_unit,
        "counts": {
            "APPROVE": int(np.sum(is_approve)),
            "REVIEW": int(np.sum(is_review)),
            "BLOCK": int(np.sum(is_block)),
        },
        "review_rate": round(float(np.mean(is_review)), 6),
        "block_rate": round(float(np.mean(is_block)), 6),
        "cost_breakdown": {
            "false_negative_approve_fraud": round(fn_cost, 4),
            "false_positive_block_legit": round(fp_cost, 4),
            "review_fraud": round(rev_fraud_cost, 4),
            "review_legit": round(rev_legit_cost, 4),
            "true_positive_block_fraud": round(tp_cost, 4),
            "true_negative_approve_legit": round(tn_cost, 4),
        },
    }


def compute_baseline_approve_all_cost(
    y_true: np.ndarray,
    amounts: np.ndarray,
    assumptions: CostAssumptions,
) -> Dict[str, Any]:
    """Compute baseline cost if every transaction is blindly approved (no fraud model)."""
    decisions = np.full(len(y_true), "APPROVE", dtype=object)
    return compute_decision_costs(y_true, decisions, amounts, assumptions)
