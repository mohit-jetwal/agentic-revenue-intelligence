"""The forecasting tool (brief sections 20, 39).

The contract a future Claude agent calls. LangGraph is **not** implemented here
and none is needed: ``AnalyticalTool`` from Step 1 already defines the envelope,
so this class writes only ``_execute`` and four class attributes. ``run()`` is
``@final`` in the base and handles validation, timing, tracing, metrics and
error wrapping - which is why a tool cannot accidentally return a malformed
result.

What the agent should be able to do with the output, and what it should not:

* It should be able to state a number **with its uncertainty**, because
  ``confidence`` here is measured interval coverage rather than a mood.
* It should be able to re-plan around a refusal, because a recoverable failure
  says what would have worked - "the latest as-of supporting 90 days is X".
* It should **not** need to know that LightGBM exists, where the parquet lives,
  how features are built, or what MLflow is. None of that appears in the schema.
* It should not claim a promotion *caused* anything. The assumptions list says
  so explicitly, because a forecast is predictive and attribution belongs to the
  uplift model.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import get_logger
from app.schemas.domain import ForecastHorizon
from app.schemas.forecast import ForecastErrorResponse, ForecastRequest
from app.schemas.tool_contract import ModelProvenance, ToolErrorCode
from app.services.forecast_service import ForecastingService
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput

logger = get_logger(__name__)


class ForecastInput(BaseModel):
    """What the agent asks for.

    Horizon is an int here rather than the ``ForecastHorizon`` enum because an
    LLM writes JSON, and ``30`` is what it will naturally produce. The mapping to
    the supported set happens below, with an error message that lists the
    options rather than silently rounding.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str | None = Field(
        default=None, description="Product to forecast. Omit to forecast every trained product."
    )
    store_id: str | None = Field(
        default=None, description="Store to forecast. Omit to forecast across stores."
    )
    region: str | None = Field(default=None, description="Restrict to one region.")
    forecast_horizon: int = Field(
        default=28,
        description=(
            "Days ahead to forecast. One of 7, 14, 28, 30 or 90. "
            "28 is four whole weeks and is the usual retail planning horizon."
        ),
    )
    as_of_date: date | None = Field(
        default=None,
        description=(
            "Forecast origin. Defaults to the most recent date that supports the "
            "full horizon, which may be earlier than the last date in the data."
        ),
    )
    include_daily: bool = Field(
        default=True, description="Include the day-by-day path as well as the total."
    )


class ForecastOutput(BaseModel):
    """What the agent gets back."""

    model_config = ConfigDict(protected_namespaces=())

    horizon_days: int
    as_of_date: date
    total_predicted_units: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    total_predicted_revenue: float | None = None
    series_count: int = 0
    forecast: list[dict[str, object]] = Field(default_factory=list)
    accuracy: dict[str, object] = Field(default_factory=dict)
    fallback_used: bool = False
    fallback_reason: str | None = None


class ForecastingTool(AnalyticalTool[ForecastInput, ForecastOutput]):
    """Forecast demand over 7, 14, 30 or 90 days."""

    name = "forecast_demand"
    description = (
        "Forecast future demand for a product, store, or region over a 7, 14, 28, "
        "30 or 90 day horizon (28 days is four whole weeks, the usual retail "
        "planning window). Returns total predicted units with a calibrated "
        "prediction interval, an optional day-by-day path, and the model's "
        "measured historical accuracy at that horizon. This is a *predictive* "
        "forecast given planned prices and promotions - it does not estimate what "
        "a promotion caused."
    )
    input_schema = ForecastInput
    output_schema = ForecastOutput
    permission = "run_model"
    timeout_seconds = 120.0

    def __init__(self, service: ForecastingService) -> None:
        self._service = service

    def _execute(self, payload: ForecastInput) -> ToolOutput[ForecastOutput]:
        horizon = self._resolve_horizon(payload.forecast_horizon)

        response = self._service.forecast(
            ForecastRequest(
                horizon=horizon,
                product_ids=[payload.product_id] if payload.product_id else None,
                store_ids=[payload.store_id] if payload.store_id else None,
                region=payload.region,
                as_of_date=payload.as_of_date,
                include_points=payload.include_daily,
                include_series=False,
            )
        )

        # `isinstance` rather than a status-string check: it narrows the union
        # for the type checker as well as at runtime, so every attribute access
        # below is verified rather than assumed.
        if isinstance(response, ForecastErrorResponse):
            # Translated into the tool's own error vocabulary, so a supervisor
            # sees one taxonomy rather than each tool's private strings.
            raise ToolExecutionError(
                response.message,
                code=_ERROR_CODES.get(response.error_code, ToolErrorCode.INTERNAL_ERROR),
                recoverable=response.recoverable,
                detail=response.detail,
            )

        output = ForecastOutput(
            horizon_days=response.horizon_days,
            as_of_date=response.as_of_date,
            total_predicted_units=round(response.total_predicted_units, 1),
            lower_bound=(
                round(response.total_lower_bound, 1)
                if response.total_lower_bound is not None
                else None
            ),
            upper_bound=(
                round(response.total_upper_bound, 1)
                if response.total_upper_bound is not None
                else None
            ),
            total_predicted_revenue=(
                round(response.total_predicted_revenue, 2)
                if response.total_predicted_revenue is not None
                else None
            ),
            series_count=response.series_count,
            forecast=[
                {
                    "date": str(point.date),
                    "predicted_units": round(point.predicted_units, 1),
                    "lower_bound": (
                        round(point.lower_bound, 1) if point.lower_bound is not None else None
                    ),
                    "upper_bound": (
                        round(point.upper_bound, 1) if point.upper_bound is not None else None
                    ),
                }
                for point in response.points
            ],
            accuracy={
                "test_wmape": response.accuracy.test_wmape,
                "wmape_by_horizon": response.accuracy.bucket_wmape,
                "interval_nominal_coverage": response.accuracy.interval_nominal,
            },
            fallback_used=response.fallback_used,
            fallback_reason=response.fallback_reason,
        )

        return ToolOutput(
            payload=output,
            provenance=ModelProvenance(
                model_name=response.model_name,
                model_version=response.model_version,
                dataset_version=response.dataset_version,
            ),
            confidence=response.confidence,
            assumptions=list(response.assumptions),
            warnings=list(response.warnings),
        )

    @staticmethod
    def _resolve_horizon(days: int) -> ForecastHorizon:
        """Map a plain integer onto a supported horizon.

        Refuses rather than snapping to the nearest. The model was trained and
        calibrated for these horizons; quietly serving 45 days as 30 would return
        a number with an interval that does not describe it.
        """
        for horizon in ForecastHorizon:
            if horizon.days == days:
                return horizon
        supported = ", ".join(str(h.days) for h in ForecastHorizon)
        raise ToolExecutionError(
            f"forecast_horizon must be one of {supported}; got {days}",
            code=ToolErrorCode.INVALID_INPUT,
            recoverable=True,
            detail={"supported_horizons": [h.days for h in ForecastHorizon]},
        )


#: Service error codes to the tool taxonomy.
_ERROR_CODES = {
    "model_not_found": ToolErrorCode.MODEL_NOT_FOUND,
    "model_not_fitted": ToolErrorCode.MODEL_NOT_FOUND,
    "insufficient_data": ToolErrorCode.INSUFFICIENT_DATA,
    "forecast_failed": ToolErrorCode.INTERNAL_ERROR,
    "invalid_input": ToolErrorCode.INVALID_INPUT,
}
