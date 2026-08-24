"""Exception handlers producing a uniform error body.

Every non-2xx response uses :class:`~app.schemas.api.ErrorResponse` and carries
the request's trace id, so a user reporting "it failed" hands over an identifier
that leads straight to the logs for that request.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.logging import get_logger
from app.schemas.api import ErrorResponse
from app.services.container import ConfigurationError

logger = get_logger(__name__)


class NotImplementedYetError(Exception):
    """Raised by endpoints whose implementation belongs to a later step.

    Carries the step name so a 501 tells the caller *when* the endpoint arrives
    rather than merely that it is missing.
    """

    def __init__(self, feature: str, step: str) -> None:
        super().__init__(f"{feature} is implemented in {step}")
        self.feature = feature
        self.step = step


def _trace_id(request: Request) -> str | None:
    value = getattr(request.state, "trace_id", None)
    return value if isinstance(value, str) else None


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers to the application."""

    @app.exception_handler(NotImplementedYetError)
    async def _not_implemented(request: Request, exc: NotImplementedYetError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content=ErrorResponse(
                error="not_implemented",
                detail=str(exc),
                trace_id=_trace_id(request),
                implemented_in=exc.step,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ConfigurationError)
    async def _configuration_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("api.configuration_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="configuration_error",
                detail=str(exc),
                trace_id=_trace_id(request),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(
                error="validation_error",
                detail=str(exc.errors()),
                trace_id=_trace_id(request),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                detail=str(exc.detail),
                trace_id=_trace_id(request),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback, return a generic message: internal details must not
        # reach the client, but the trace id lets an operator find them.
        logger.exception("api.unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred.",
                trace_id=_trace_id(request),
            ).model_dump(mode="json"),
        )
