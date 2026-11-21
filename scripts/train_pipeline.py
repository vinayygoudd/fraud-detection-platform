"""End-to-end training pipeline: PostgreSQL SQL feature extraction -> ML training -> Evaluation -> SHAP -> MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import extract_offline_features
from src.models.trainer import train_and_evaluate
from src.utils.config import get_config, get_cost_matrix_config
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end fraud detection training pipeline from PostgreSQL."
    )
    parser.add_argument("--min-step", type=int, default=None, help="Optional minimum step filter.")
    parser.add_argument("--max-step", type=int, default=None, help="Optional maximum step filter.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment logging for this run.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger("scripts.train_pipeline")
    args = parse_args()

    cfg = get_config()
    cost_cfg = get_cost_matrix_config()
    engine = get_engine()

    features_df = extract_offline_features(
        engine=engine,
        min_step=args.min_step,
        max_step=args.max_step,
        config=cfg,
    )
    if features_df.empty:
        logger.error(
            "No rows returned from historical_transactions. "
            "Run `python scripts/init_db.py` and `python scripts/load_paysim.py` first."
        )
        sys.exit(1)

    audit_report_path = PROJECT_ROOT / cfg["paths"]["reports_dir"] / "paysim_data_quality_report.json"
    dataset_meta = {}
    if audit_report_path.exists():
        audit_data = json.loads(audit_report_path.read_text(encoding="utf-8"))
        dataset_meta = {
            "source_file": audit_data.get("source_file"),
            "sha256": audit_data.get("sha256"),
            "total_csv_rows": audit_data.get("row_count"),
        }

    result = train_and_evaluate(
        features_df=features_df,
        config=cfg,
        cost_config=cost_cfg,
        dataset_metadata=dataset_meta,
        enable_mlflow=not args.no_mlflow,
    )
    logger.info("Training pipeline finished successfully.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
