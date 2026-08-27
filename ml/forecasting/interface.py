"""Demand forecasting model interface (Stage 1 Step 5)."""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from app.schemas.domain import ForecastHorizon
from ml.base import AnalyticalModel


class ForecastPoint(BaseModel):
    date: date
    predicted_units: float = Field(ge=0)
    lower_bound: float | None = None
    upper_bound: float | None = None


class ForecastResult(BaseModel):
    """A demand forecast with the accuracy record needed to judge it.

    ``backtest_metrics`` is part of the result rather than buried in MLflow
    because a forecast without its historical error is not actionable - the
    agent needs to know whether +/-3% is signal or noise before recommending
    anything on the strength of it.
    """

    product_id: str | None = None
    store_id: str | None = None
    region: str | None = None
    horizon: ForecastHorizon
    points: list[ForecastPoint] = Field(default_factory=list)

    total_predicted_units: float = Field(ge=0)
    total_predicted_revenue: float | None = None

    #: WMAPE is the headline: it weights by volume, so a large error on a
    #: high-volume SKU is not hidden by small errors on a long tail of slow ones.
    backtest_metrics: dict[str, float] = Field(default_factory=dict)
    model_used: str | None = None


class ForecastingModel(AnalyticalModel[ForecastResult]):
    """Demand forecaster supporting 7/14/28/30/90-day horizons."""

    name = "demand_forecast"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        horizon: ForecastHorizon,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        as_of: date | None = None,
    ) -> ForecastResult:
        """Forecast demand over the requested horizon."""

    @abstractmethod
    def backtest(
        self,
        *,
        horizon: ForecastHorizon,
        n_splits: int = 3,
        product_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Expanding-window temporal validation.

        Expanding-window, never random k-fold: shuffling time series rows lets
        the model see the future while predicting the past, which produces
        excellent scores and worthless forecasts.
        """
