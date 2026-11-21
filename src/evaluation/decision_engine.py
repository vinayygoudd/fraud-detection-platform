"""Three-way cost-sensitive fraud decision engine (APPROVE / REVIEW / BLOCK)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.cost_matrix import (
    CostAssumptions,
    compute_baseline_approve_all_cost,
    compute_decision_costs,
)
from src.utils.config import get_cost_matrix_config


@dataclass(frozen=True)
class ThresholdConfig:
    """Validated two-threshold configuration enforcing 0 < t1 < t2 < 1."""

    review_threshold: float  # t1
    block_threshold: float   # t2

    def __post_init__(self) -> None:
        if not (0.0 < self.review_threshold < self.block_threshold < 1.0):
            raise ValueError(
                f"Invalid thresholds: review_threshold={self.review_threshold}, "
                f"block_threshold={self.block_threshold}. Must strictly satisfy 0 < t1 < t2 < 1."
            )


class DecisionEngine:
    """Maps a scalar risk_score in [0, 1] to APPROVE, REVIEW, or BLOCK.

    Decision rules:
      - risk_score < t1              -> APPROVE
      - t1 <= risk_score < t2        -> REVIEW
      - risk_score >= t2             -> BLOCK
    """

    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds

    def decide_one(self, risk_score: float) -> str:
        """Return operational decision for a single transaction risk score."""
        score = float(risk_score)
        if np.isnan(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"risk_score must be in [0.0, 1.0], got {risk_score}")
        if score < self.thresholds.review_threshold:
            return "APPROVE"
        if score < self.thresholds.block_threshold:
            return "REVIEW"
        return "BLOCK"

    def decide_batch(self, risk_scores: np.ndarray) -> np.ndarray:
        """Vectorized operational decisions for an array of risk scores."""
        scores = np.asarray(risk_scores, dtype=float)
        if np.any(np.isnan(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("All risk_scores must be finite values in [0.0, 1.0].")
        decisions = np.full(scores.shape, "APPROVE", dtype=object)
        decisions[(scores >= self.thresholds.review_threshold) & (scores < self.thresholds.block_threshold)] = "REVIEW"
        decisions[scores >= self.thresholds.block_threshold] = "BLOCK"
        return decisions


def optimize_thresholds_on_validation(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    amounts_val: np.ndarray,
    assumptions: CostAssumptions | None = None,
    cost_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Grid-search (t1, t2) on Validation data to minimize expected financial cost.

    Enforces 0 < t1 < t2 < 1 and optional `max_review_rate` capacity constraint.
    NEVER call this function on the Test split.
    """
    cfg = cost_cfg or get_cost_matrix_config()
    cost_params = assumptions or CostAssumptions.from_config(cfg)
    de_cfg = cfg.get("decision_engine", {})
    search_cfg = de_cfg.get("search", {})

    t1_min = float(search_cfg.get("t1_min", 0.05))
    t1_max = float(search_cfg.get("t1_max", 0.60))
    t1_step = float(search_cfg.get("t1_step", 0.05))
    t2_min = float(search_cfg.get("t2_min", 0.20))
    t2_max = float(search_cfg.get("t2_max", 0.95))
    t2_step = float(search_cfg.get("t2_step", 0.05))
    min_gap = float(search_cfg.get("min_gap", 0.05))
    max_review_rate: Optional[float] = de_cfg.get("max_review_rate", 0.10)

    t1_grid = np.round(np.arange(t1_min, t1_max + 1e-9, t1_step), 4)
    t2_grid = np.round(np.arange(t2_min, t2_max + 1e-9, t2_step), 4)

    baseline = compute_baseline_approve_all_cost(y_val, amounts_val, cost_params)
    best_result: Optional[Dict[str, Any]] = None
    candidates_evaluated = 0

    for t1 in t1_grid:
        for t2 in t2_grid:
            if not (0.0 < t1 < t2 < 1.0) or (t2 - t1) < min_gap - 1e-9:
                continue
            candidates_evaluated += 1
            engine = DecisionEngine(ThresholdConfig(review_threshold=float(t1), block_threshold=float(t2)))
            decisions = engine.decide_batch(scores_val)
            eval_res = compute_decision_costs(y_val, decisions, amounts_val, cost_params)

            if max_review_rate is not None and eval_res["review_rate"] > max_review_rate:
                continue

            if best_result is None or eval_res["total_cost"] < best_result["total_cost"]:
                best_result = {
                    "review_threshold": float(t1),
                    "block_threshold": float(t2),
                    **eval_res,
                }

    # Fallback if capacity constraint excluded every candidate
    if best_result is None:
        default_t1 = float(de_cfg.get("default_review_threshold", 0.25))
        default_t2 = float(de_cfg.get("default_block_threshold", 0.75))
        engine = DecisionEngine(ThresholdConfig(review_threshold=default_t1, block_threshold=default_t2))
        eval_res = compute_decision_costs(y_val, engine.decide_batch(scores_val), amounts_val, cost_params)
        best_result = {
            "review_threshold": default_t1,
            "block_threshold": default_t2,
            "fallback_used": True,
            **eval_res,
        }

    savings = baseline["total_cost"] - best_result["total_cost"]
    savings_pct = (savings / baseline["total_cost"] * 100.0) if baseline["total_cost"] > 0 else 0.0

    return {
        "optimal_thresholds": {
            "review_threshold": best_result["review_threshold"],
            "block_threshold": best_result["block_threshold"],
        },
        "validation_cost_summary": best_result,
        "baseline_approve_all_cost": baseline,
        "cost_savings_vs_approve_all": round(float(savings), 4),
        "cost_savings_pct": round(float(savings_pct), 2),
        "max_review_rate_constraint": max_review_rate,
        "candidates_evaluated": candidates_evaluated,
    }


def run_cost_sensitivity_analysis(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    amounts_val: np.ndarray,
    base_assumptions: CostAssumptions | None = None,
    cost_cfg: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Run sensitivity analysis across false-negative, false-positive, and review cost multipliers."""
    cfg = cost_cfg or get_cost_matrix_config()
    base = base_assumptions or CostAssumptions.from_config(cfg)
    sens_cfg = cfg.get("sensitivity_analysis", {})

    results: List[Dict[str, Any]] = []
    for mult in sens_cfg.get("false_negative_multipliers", [0.5, 1.0, 2.0]):
        mod = replace(
            base,
            false_negative_fixed=base.false_negative_fixed * float(mult),
            false_negative_amount_fraction=base.false_negative_amount_fraction * float(mult),
        )
        opt = optimize_thresholds_on_validation(y_val, scores_val, amounts_val, assumptions=mod, cost_cfg=cfg)
        results.append(
            {
                "parameter": "false_negative_cost",
                "multiplier": float(mult),
                "review_threshold": opt["optimal_thresholds"]["review_threshold"],
                "block_threshold": opt["optimal_thresholds"]["block_threshold"],
                "total_cost": opt["validation_cost_summary"]["total_cost"],
                "review_rate": opt["validation_cost_summary"]["review_rate"],
            }
        )

    for mult in sens_cfg.get("false_positive_multipliers", [0.5, 1.0, 2.0]):
        mod = replace(
            base,
            false_positive_block_fixed=base.false_positive_block_fixed * float(mult),
            false_positive_block_amount_fraction=base.false_positive_block_amount_fraction * float(mult),
        )
        opt = optimize_thresholds_on_validation(y_val, scores_val, amounts_val, assumptions=mod, cost_cfg=cfg)
        results.append(
            {
                "parameter": "false_positive_block_cost",
                "multiplier": float(mult),
                "review_threshold": opt["optimal_thresholds"]["review_threshold"],
                "block_threshold": opt["optimal_thresholds"]["block_threshold"],
                "total_cost": opt["validation_cost_summary"]["total_cost"],
                "review_rate": opt["validation_cost_summary"]["review_rate"],
            }
        )

    return results
