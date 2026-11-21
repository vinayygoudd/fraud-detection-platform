"""Central configuration loader combining YAML files and environment variables."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _resolve_env_placeholders(value: Any) -> Any:
    """Recursively expand ${VAR:-default} placeholders in YAML values."""
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):
        def _replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_val)

        return ENV_VAR_PATTERN.sub(_replacer, value)
    return value


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and resolve environment variable placeholders."""
    load_dotenv()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    return _resolve_env_placeholders(raw_cfg)


@lru_cache(maxsize=4)
def get_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load and cache the main platform configuration."""
    resolved_path = config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
    return load_yaml_config(resolved_path)


@lru_cache(maxsize=4)
def get_cost_matrix_config(cost_matrix_path: str | None = None) -> Dict[str, Any]:
    """Load and cache the financial cost matrix configuration."""
    resolved_path = cost_matrix_path or os.getenv(
        "COST_MATRIX_PATH", "configs/cost_matrix.yaml"
    )
    return load_yaml_config(resolved_path)


def build_database_url(config: Dict[str, Any] | None = None) -> str:
    """Construct a SQLAlchemy PostgreSQL connection URL from config / environment."""
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        return direct_url

    cfg = config or get_config()
    db_cfg = cfg["database"]
    host = os.getenv("POSTGRES_HOST", db_cfg.get("host", "localhost"))
    port = os.getenv("POSTGRES_PORT", str(db_cfg.get("port", 5432)))
    name = os.getenv("POSTGRES_DB", db_cfg.get("name", "fraud_detection"))
    user = os.getenv("POSTGRES_USER", db_cfg.get("user", "fraud_user"))
    password = os.getenv("POSTGRES_PASSWORD", db_cfg.get("password", ""))
    return f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{name}"


def get_feature_names(config: Dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Return canonical (numeric_features, categorical_features) in exact model order."""
    cfg = config or get_config()
    feat_cfg = cfg["features"]
    return list(feat_cfg["numeric_features"]), list(feat_cfg["categorical_features"])
