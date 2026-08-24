"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import TraceMiddleware
from app.api.routes import health, investigations
from app.config.settings import Settings, get_settings
from app.observability.logging import configure_logging, get_logger
from app.services.container import Container

logger = get_logger(__name__)

DESCRIPTION = """
Agentic decision-intelligence platform for CPG/Retail revenue, pricing and
promotion management.

Claude plans, selects tools, interprets evidence and re-plans. Deterministic
ML, statistical and optimisation models produce every number. The LLM never
computes a business figure - it explains one, always with the model version and
data version it came from.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so tests can construct an app
    with overridden settings, and so importing the module has no side effects.
    """
    cfg = settings or get_settings()
    configure_logging(cfg)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # The container is built once at startup and attached to app state.
        # Components inside it stay lazy, so a missing dependency surfaces in
        # /health rather than preventing the service from starting at all.
        container = Container(cfg)
        application.state.container = container
        logger.info(
            "app.startup",
            environment=cfg.app.environment.value,
            version=cfg.app.version,
        )
        try:
            yield
        finally:
            logger.info("app.shutdown")

    application = FastAPI(
        title="Agentic Revenue Intelligence",
        description=DESCRIPTION.strip(),
        version=cfg.app.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(TraceMiddleware)
    register_exception_handlers(application)

    application.include_router(health.router)
    application.include_router(investigations.router)

    return application
