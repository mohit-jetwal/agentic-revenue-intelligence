"""HTTP middleware: trace propagation, request logging, metrics."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.context import new_trace_id, trace_context
from app.observability.logging import get_logger
from app.observability.metrics import METRICS

logger = get_logger(__name__)

TRACE_HEADER = "X-Trace-Id"


class TraceMiddleware(BaseHTTPMiddleware):
    """Bind a trace id to every request and echo it back.

    Accepts an inbound ``X-Trace-Id`` so a caller can correlate across services,
    generating one when absent. The id is bound to a contextvar for the duration
    of the request, which is how a log line emitted inside a tool call ends up
    joinable to the HTTP request that triggered it.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get(TRACE_HEADER) or new_trace_id()
        started = time.perf_counter()

        with trace_context(trace_id):
            request.state.trace_id = trace_id
            METRICS.increment("requests_total")
            try:
                response = await call_next(request)
            except Exception:
                METRICS.increment("requests_failed")
                logger.exception(
                    "http.unhandled_error",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                raise

            duration_ms = int((time.perf_counter() - started) * 1000)
            if response.status_code >= 500:
                METRICS.increment("requests_failed")

            response.headers[TRACE_HEADER] = trace_id
            # /health and /metrics are polled frequently; logging them at INFO
            # drowns the signal from real traffic.
            level = logger.debug if request.url.path in ("/health", "/metrics") else logger.info
            level(
                "http.request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
