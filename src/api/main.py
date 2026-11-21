"""FastAPI application factory and startup lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI

from src.api.routes import router
from src.models.registry import LoadedModelBundle, load_model_bundle
from src.utils.config import get_config
from src.utils.db import get_engine
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app(
    db_engine: Any = None,
    model_bundle: Optional[LoadedModelBundle] = None,
    load_defaults: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    setup_logging()
    cfg = get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if app.state.db_engine is None and load_defaults:
            try:
                app.state.db_engine = get_engine()
            except Exception as exc:
                logger.warning("Could not initialize DB engine at startup: %s", exc)

        if app.state.model_bundle is None and load_defaults:
            try:
                app.state.model_bundle = load_model_bundle()
                logger.info(
                    "Loaded model bundle: %s v%s",
                    app.state.model_bundle.model_name,
                    app.state.model_bundle.model_version,
                )
            except FileNotFoundError as exc:
                logger.warning("Model artifacts not found at startup: %s", exc)
            except Exception as exc:
                logger.exception("Unexpected error loading model bundle: %s", exc)
        yield

    app = FastAPI(
        title=cfg["project"]["name"],
        version=cfg["project"]["version"],
        description=(
            "Simulated digital payment fraud detection and cost-sensitive risk scoring platform "
            "backed by PostgreSQL behavioural features, classical ML, and SHAP explainability."
        ),
        lifespan=lifespan,
    )
    app.state.db_engine = db_engine
    app.state.model_bundle = model_bundle
    app.include_router(router)
    return app


app = create_app()
