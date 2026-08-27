"""Request/response schemas for the forecasting service (brief sections 18-19).

Shaped so the Step 13 tool wrapper can lift these into a ``ToolResult`` without
translation - the provenance fields mirror it exactly. Deliberately *not* a
``ToolResult`` itself: the service is usable from a notebook, a script or the
API, and none of those should have to know what a tool envelope is.

The one field worth pausing on is ``confidence``. Section 18 says not to
fabricate it, so it is **derived from measured interval coverage**, not asserted.
See :attr:`ForecastResponse.confidence` for what it does and does not mean.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.domain import ForecastHorizon


class ForecastRequest(BaseModel):
    """What to forecast."""

    model_config = ConfigDict(frozen=True)

    horizon: ForecastHorizon = ForecastHorizon.D30
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None
    #: Forecast origin. Defaults to the latest date that supports the full
    #: horizon, which is not the latest date in the data - see the service.
    as_of_date: date | None = None
    include_points: bool = True
    include_series: bool = False
    max_points: int = Field(default=400, gt=0, le=10_000)


class ForecastPointRecord(BaseModel):
    """One forecast day."""

    date: date
    predicted_units: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class ForecastSeriesRecord(BaseModel):
    """One series' horizon total."""

    model_config = ConfigDict(protected_namespaces=())

    product_id: str
    store_id: str
    total_predicted_units: float
    total_lower_bound: float | None = None
    total_upper_bound: float | None = None
    total_predicted_revenue: float | None = None


class ForecastAccuracy(BaseModel):
    """Measured historical accuracy, per horizon bucket.

    Part of the response rather than buried in MLflow because a forecast without
    its error record is not actionable: a caller cannot tell whether a 3%
    movement is signal or noise without knowing the model's normal error at that
    horizon.
    """

    test_wmape: float | None = None
    bucket_wmape: dict[str, float] = Field(default_factory=dict)
    bucket_coverage: dict[str, float] = Field(default_factory=dict)
    interval_nominal: float | None = None
    fva_vs_seasonal_naive_pp: dict[str, float] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
    """A forecast, with everything needed to judge it."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "success"
    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str

    horizon_days: int
    as_of_date: date
    total_predicted_units: float
    total_lower_bound: float | None = None
    total_upper_bound: float | None = None
    total_predicted_revenue: float | None = None

    points: list[ForecastPointRecord] = Field(default_factory=list)
    series: list[ForecastSeriesRecord] = Field(default_factory=list)
    series_count: int = 0

    #: Measured interval coverage at this horizon, in [0, 1] - **not** a
    #: subjective confidence and not a probability that the forecast is right.
    #:
    #: It answers exactly one question: historically, how often did the
    #: prediction interval contain the actual? A value near the nominal 0.9 means
    #: the interval is trustworthy; a value well below it means the interval is
    #: too narrow and is reported as such rather than quietly widened.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    accuracy: ForecastAccuracy = Field(default_factory=ForecastAccuracy)
    fallback_used: bool = False
    fallback_reason: str | None = None
    fallback_rows: int = 0
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_ms: int = 0

    def summary(self) -> str:
        interval = (
            f" [{self.total_lower_bound:,.0f}-{self.total_upper_bound:,.0f}]"
            if self.total_lower_bound is not None
            else ""
        )
        return (
            f"{self.total_predicted_units:,.0f} units{interval} over "
            f"{self.horizon_days} days from {self.as_of_date} "
            f"({self.model_name} {self.model_version})"
        )


class ForecastErrorResponse(BaseModel):
    """An expected failure, returned rather than raised."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "error"
    error_code: str
    message: str
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int = 0
