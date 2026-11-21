"""Structured logging configuration for CLI scripts and FastAPI service."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure root logging with consistent timestamped formatting."""
    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger("fraud_detection")
