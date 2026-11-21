"""Audit and load the PaySim CSV dataset into PostgreSQL historical_transactions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import audit_paysim_csv
from src.data.ingestion import load_paysim_to_postgres
from src.utils.config import get_config
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and load PaySim CSV into PostgreSQL.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to PaySim CSV (defaults to configs/config.yaml paths.raw_paysim_file).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick local smoke tests.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for streaming CSV ingestion.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate historical_transactions before loading.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip full CSV data quality audit report before loading.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger("scripts.load_paysim")
    args = parse_args()
    cfg = get_config()

    csv_path = Path(args.csv_path or cfg["paths"]["raw_paysim_file"])
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not args.skip_audit:
        report_path = PROJECT_ROOT / cfg["paths"]["reports_dir"] / "paysim_data_quality_report.json"
        audit_report = audit_paysim_csv(
            csv_path=csv_path,
            output_report_path=report_path,
            max_rows=args.max_rows,
            config=cfg,
        )
        logger.info(
            "Audit complete: rows=%d, fraud=%d (%.4f%%), valid_schema=%s",
            audit_report["row_count"],
            audit_report["fraud_distribution"]["fraud_count"],
            audit_report["fraud_distribution"]["fraud_rate"] * 100.0,
            audit_report["schema_validation"]["is_valid"],
        )
        if not audit_report["schema_validation"]["is_valid"]:
            logger.error("Schema validation failed: %s", audit_report["schema_validation"]["errors"])
            sys.exit(1)

    engine = get_engine()
    summary = load_paysim_to_postgres(
        csv_path=csv_path,
        engine=engine,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        truncate_first=args.truncate,
        config=cfg,
    )
    logger.info("PostgreSQL ingestion complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
