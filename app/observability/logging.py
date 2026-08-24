"""Structured logging.

JSON by default so logs are queryable once they land in a Databricks table
(Stage 2); human-readable console output when ``OBSERVABILITY__JSON_LOGS=false``,
which is what you want while developing.

Every record is automatically enriched with the active trace context, so a tool
call logged deep in the stack is joinable to the API request that caused it
without any explicit plumbing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.config.settings import Settings, get_settings
from app.observability.context import current_context

_configured = False


def _add_trace_context(
    _logger: Any, _method: str, event_dict: EventDict
) -> EventDict:
    """Merge contextvar trace ids into every record."""
    for key, value in current_context().items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: safe to call from both the API lifespan and a CLI entrypoint.
    """
    global _configured
    if _configured:
        return

    cfg = (settings or get_settings()).observability
    level = getattr(logging, cfg.log_level, logging.INFO)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_trace_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if cfg.json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # The stdlib factory (rather than PrintLoggerFactory) so that records
        # carry a logger name for ``add_logger_name``, and so that library logs
        # emitted through stdlib logging land in the same stream and format.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    # uvicorn installs its own handlers; let them propagate to root so
    # everything ends up in one stream with one format.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Configures logging on first use."""
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def reset_logging() -> None:
    """Allow reconfiguration. Intended for tests only."""
    global _configured
    _configured = False
