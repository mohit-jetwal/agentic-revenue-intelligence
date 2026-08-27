"""Typed failures for the forecasting capability.

Before this module the code raised bare ``ValueError``, ``KeyError`` and
``RuntimeError``. That works, but it pushes the classification problem onto the
caller: a service trying to decide whether a failure is *recoverable* - whether
a different request would succeed - had to inspect message strings, and a
supervisor agent in Step 16 would have to do the same.

**The inheritance is the load-bearing part, not the names.**
:class:`InsufficientHistoryError`, :class:`UnknownSeriesError` and
:class:`HorizonUnavailableError` all subclass
:class:`ml.base.InsufficientDataError`, and :class:`ModelUnavailableError`
subclasses :class:`ml.base.ModelNotFittedError`. That means
``app/services/forecast_service.py`` keeps mapping them to the right error codes
without a single change, while callers who want the finer distinction can now ask
for it. Introducing a parallel hierarchy would have been cleaner on paper and
would have silently broken that mapping.

Each class carries the *recoverability* judgement as a class attribute, because
that is a property of the failure kind rather than of the call site - and
deciding it once here is what stops two call sites from disagreeing about
whether the same failure is worth retrying.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ml.base import InsufficientDataError, ModelNotFittedError


class ForecastingError(Exception):
    """Base for every forecasting failure.

    ``recoverable`` means: could a *different request* succeed against the same
    system? A missing model is not recoverable - no reformulation helps until
    someone trains one. A horizon reaching past the planning calendar is, because
    a shorter horizon or an earlier as-of would work.
    """

    #: Stable identifier surfaced to callers and agents.
    code: str = "forecast_failed"
    #: Whether re-planning could succeed. See the class docstring.
    recoverable: bool = True

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, for a structured error response."""
        return {
            "error_code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "detail": self.detail,
        }


class InsufficientHistoryError(ForecastingError, InsufficientDataError):
    """A series is too short to build features from.

    Distinct from :class:`HorizonUnavailableError`, which is about the *future*
    end of the window. This one is about the past: the 364-day seasonal lag is
    undefined for a product launched three months ago, and a model asked to
    predict from mostly-NaN history would return a number with no support behind
    it.
    """

    code = "insufficient_data"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        available_days: int | None = None,
        required_days: int | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message, available_days=available_days, required_days=required_days, **detail
        )


class UnknownSeriesError(ForecastingError, InsufficientDataError):
    """The requested product/store is not in the trained series set.

    Raised rather than returning an empty forecast, deliberately. Zero units
    reads as "no demand expected", which is a completely different claim from
    "this series is not in the model" - and a planner acting on the first would
    be making an inventory decision on a fact nobody established.
    """

    code = "insufficient_data"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(message, product_ids=product_ids, store_ids=store_ids, **detail)


class HorizonUnavailableError(ForecastingError, InsufficientDataError):
    """The horizon reaches past the known-in-advance planning data.

    Carries ``latest_valid_as_of`` so the caller is told what *would* have
    worked. A recoverable error without that boundary leaves an agent able to
    conclude only "it failed", which is not enough to re-plan on.
    """

    code = "insufficient_data"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        horizon_days: int | None = None,
        known_until: date | None = None,
        latest_valid_as_of: date | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message,
            horizon_days=horizon_days,
            known_until=str(known_until) if known_until else None,
            latest_valid_as_of=str(latest_valid_as_of) if latest_valid_as_of else None,
            **detail,
        )


class ModelUnavailableError(ForecastingError, ModelNotFittedError):
    """A model is loaded but unusable - no estimator, no config, nothing to save.

    Not recoverable: no reformulation of the request produces a forecast until
    someone trains a model. Saying so lets a supervisor stop retrying rather
    than burning turns on requests that cannot succeed.

    Note what this is *not* used for. ``FittedForecastModel.load_from`` still
    raises a plain ``FileNotFoundError`` when the artifact is absent, because
    that is literally what happened and the service already maps it to
    ``model_not_found``. Replacing it here would route a missing file through
    ``model_not_fitted`` instead - a less accurate code for a worse reason.
    """

    code = "model_not_fitted"
    recoverable = False


class FeatureGenerationError(ForecastingError):
    """The feature panel or future scaffold could not be built.

    Usually an upstream data problem - a missing table, a column that changed
    shape - rather than anything about the request. Recoverable in the narrow
    sense that fixing the data fixes it, which is why the detail carries the
    stage that failed.
    """

    code = "forecast_failed"
    recoverable = True

    def __init__(self, message: str, *, stage: str | None = None, **detail: Any) -> None:
        super().__init__(message, stage=stage, **detail)


class InvalidForecastRequestError(ForecastingError):
    """The request is malformed - an unsupported horizon, a reversed date range."""

    code = "invalid_input"
    recoverable = True


__all__ = [
    "FeatureGenerationError",
    "ForecastingError",
    "HorizonUnavailableError",
    "InsufficientHistoryError",
    "InvalidForecastRequestError",
    "ModelUnavailableError",
    "UnknownSeriesError",
]
