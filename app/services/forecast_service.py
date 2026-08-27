"""Forecasting service (brief section 19).

The seam between the model and everything that consumes it - the API, a
notebook, and from Step 13 the tool layer. Responsibilities: validate the
request, load the model once, produce the forecast, attach the accuracy record,
and return a structured result.

**Expected failures come back as values, not exceptions.** By Step 16 a
supervisor agent has to re-plan around them, and it can only do that with a
failure it can read: a code, a human-readable message, and a ``recoverable``
flag telling it whether a different request could succeed.

The most common failure here is one the data makes unavoidable: the calendar,
promotion schedule and price plan stop at the end of the generated dataset, so a
90-day horizon is only fully informed from an as-of at least 90 days before that.
Rather than assuming "no promotions planned" - which biases those days low and
would be indistinguishable from a real forecast - the request is refused with the
latest as-of that would work.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from app.schemas.domain import ForecastHorizon
from app.schemas.forecast import (
    ForecastAccuracy,
    ForecastErrorResponse,
    ForecastPointRecord,
    ForecastRequest,
    ForecastResponse,
    ForecastSeriesRecord,
)
from data.repositories.base import DataAccessError, DataRepository
from features.contracts.specs import FEATURE_VERSION
from ml.base import InsufficientDataError, ModelNotFittedError
from ml.forecasting.model import FittedForecastModel
from ml.forecasting.predict import latest_supported_as_of

logger = get_logger(__name__)


class ForecastingService:
    """Serves demand forecasts from a trained model."""

    def __init__(
        self,
        repository: DataRepository,
        *,
        model: FittedForecastModel | None = None,
        model_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()
        self._model_dir = model_dir or self._default_model_dir()
        self._model = model

    def _default_model_dir(self) -> Path:
        """Prefer the full model, fall back to a sampled one.

        Both are legitimate artifacts; the sampled one is what a developer has
        after a quick run. Preferring the full model means a machine with both
        serves the better one, and the response's provenance says which.
        """
        root = self._settings.project_root / "data" / "local" / "models"
        full = root / "forecasting"
        if (full / "model.joblib").is_file():
            return full
        return root / "forecasting_sampled"

    @property
    def model(self) -> FittedForecastModel:
        """The trained model, loaded on first use.

        Lazy so constructing the service - which the DI container does at
        startup - never requires a trained model to exist. A missing model
        should surface when someone asks for a forecast, with a message saying
        how to train one, not as a container failure at boot.
        """
        if self._model is None:
            self._model = FittedForecastModel.load_from(self._model_dir, self._repository)
            logger.info("forecast_service.model_loaded", directory=str(self._model_dir))
        return self._model

    @property
    def is_available(self) -> bool:
        if self._model is not None:
            return True
        return (self._model_dir / "model.joblib").is_file()

    # -- forecasting --------------------------------------------------------

    def forecast(
        self, request: ForecastRequest
    ) -> ForecastResponse | ForecastErrorResponse:
        """Produce a forecast, or a structured reason why not."""
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            model = self.model
        except FileNotFoundError as exc:
            return ForecastErrorResponse(
                error_code="model_not_found",
                message=str(exc),
                # No different request produces a forecast until someone trains
                # a model, so re-planning cannot help.
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )

        try:
            detail = model.predict_detail(
                horizon=request.horizon,
                product_ids=request.product_ids,
                store_ids=request.store_ids,
                region=request.region,
                as_of=request.as_of_date,
            )
        except InsufficientDataError as exc:
            return ForecastErrorResponse(
                error_code="insufficient_data",
                message=str(exc),
                # A shorter horizon or an earlier as-of would succeed.
                recoverable=True,
                detail=self._insufficient_detail(request),
                execution_time_ms=elapsed_ms(),
            )
        except ModelNotFittedError as exc:
            return ForecastErrorResponse(
                error_code="model_not_fitted",
                message=str(exc),
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )
        except (DataAccessError, ValueError) as exc:
            logger.warning("forecast_service.failed", error=str(exc))
            return ForecastErrorResponse(
                error_code="forecast_failed",
                message=str(exc),
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        totals = model.series_totals(
            horizon=request.horizon,
            product_ids=request.product_ids,
            store_ids=request.store_ids,
            region=request.region,
            as_of=request.as_of_date,
        )

        accuracy = self._accuracy(model, request.horizon)
        response = ForecastResponse(
            model_name=model.name,
            model_version=model.version,
            dataset_version=self._dataset_version(),
            feature_version=FEATURE_VERSION,
            horizon_days=request.horizon.days,
            as_of_date=detail.as_of,
            total_predicted_units=detail.total_units,
            total_lower_bound=float(totals["total_lower_bound"].sum())
            if not totals.empty and totals["total_lower_bound"].notna().all()
            else None,
            total_upper_bound=float(totals["total_upper_bound"].sum())
            if not totals.empty and totals["total_upper_bound"].notna().all()
            else None,
            total_predicted_revenue=detail.total_revenue(),
            points=self._points(detail, request) if request.include_points else [],
            series=self._series(totals) if request.include_series else [],
            series_count=len(totals),
            confidence=self._confidence(accuracy, request.horizon),
            accuracy=accuracy,
            fallback_used=detail.fallback_rows > 0,
            fallback_reason=self._fallback_reason(detail),
            fallback_rows=detail.fallback_rows,
            assumptions=self._assumptions(model, detail),
            warnings=self._warnings(model, detail, accuracy),
            execution_time_ms=elapsed_ms(),
        )

        logger.info(
            "forecast_service.forecast",
            horizon=request.horizon.value,
            series=response.series_count,
            units=round(response.total_predicted_units, 1),
            duration_ms=response.execution_time_ms,
        )
        return response

    # -- helpers ------------------------------------------------------------

    def _insufficient_detail(self, request: ForecastRequest) -> dict[str, Any]:
        """Tell the caller what *would* work.

        An agent re-planning around this needs the boundary, not just a refusal.
        """
        try:
            view = self._repository.as_of(date.today())
            latest = latest_supported_as_of(view, request.horizon.days)
            return {
                "horizon_days": request.horizon.days,
                "latest_valid_as_of": str(latest),
            }
        except Exception:  # noqa: BLE001 - detail is best-effort
            return {"horizon_days": request.horizon.days}

    def _points(self, detail: Any, request: ForecastRequest) -> list[ForecastPointRecord]:
        frame = detail.frame
        has_interval = "lower_bound" in frame.columns and "upper_bound" in frame.columns

        aggregation: dict[str, tuple[str, str]] = {"predicted": ("predicted_units", "sum")}
        if has_interval:
            aggregation["lower"] = ("lower_bound", "sum")
            aggregation["upper"] = ("upper_bound", "sum")

        daily = (
            frame.groupby("target_date", observed=True)
            .agg(**aggregation)
            .reset_index()
            .sort_values("target_date")
            .head(request.max_points)
        )
        return [
            ForecastPointRecord(
                date=row["target_date"].date()
                if hasattr(row["target_date"], "date")
                else row["target_date"],
                predicted_units=float(row["predicted"]),
                lower_bound=float(row["lower"]) if has_interval else None,
                upper_bound=float(row["upper"]) if has_interval else None,
            )
            for _, row in daily.iterrows()
        ]

    def _series(self, totals: Any) -> list[ForecastSeriesRecord]:
        if totals.empty:
            return []
        return [
            ForecastSeriesRecord(
                product_id=str(row["product_id"]),
                store_id=str(row["store_id"]),
                total_predicted_units=float(row["total_predicted_units"]),
                total_lower_bound=_optional_float(row.get("total_lower_bound")),
                total_upper_bound=_optional_float(row.get("total_upper_bound")),
                total_predicted_revenue=_optional_float(row.get("total_predicted_revenue")),
            )
            for _, row in totals.iterrows()
        ]

    def _accuracy(
        self, model: FittedForecastModel, horizon: ForecastHorizon
    ) -> ForecastAccuracy:
        metrics = dict(getattr(model, "_metrics", {}) or {})
        bucket_wmape = {
            key.replace("bucket_", "").replace("_wmape", ""): value
            for key, value in metrics.items()
            if key.startswith("bucket_") and key.endswith("_wmape")
        }
        calibration = model.calibration
        return ForecastAccuracy(
            test_wmape=metrics.get("test_wmape"),
            bucket_wmape=bucket_wmape,
            interval_nominal=calibration.nominal_coverage if calibration else None,
        )

    def _confidence(
        self, accuracy: ForecastAccuracy, horizon: ForecastHorizon
    ) -> float | None:
        """Derive confidence from measured coverage, never assert it.

        Returns the interval's **nominal** coverage only when a calibration
        exists, because that is the one number with a defensible meaning: the
        share of past actuals that fell inside intervals built this way. When no
        calibration exists this returns ``None`` rather than a plausible-looking
        default - an absent number is honest, and a fabricated 0.89 is exactly
        what section 18 forbids.
        """
        return accuracy.interval_nominal

    def _fallback_reason(self, detail: Any) -> str | None:
        if not detail.fallback_rows:
            return None
        parts = [f"{count} rows via {reason}" for reason, count in detail.fallback_reasons.items()]
        return "; ".join(parts) if parts else "primary model produced no value"

    def _assumptions(self, model: FittedForecastModel, detail: Any) -> list[str]:
        """Assumptions derived from what the model actually did."""
        assumptions = [
            "Forecast is expected demand, not expected shipments: rows where "
            "inventory censored sales were excluded from training, and no "
            "inventory feature is used.",
            "Planned promotions and planned prices for the forecast dates are "
            "taken from the promotion and pricing calendars, which are known in "
            "advance. If those plans change, the forecast changes with them.",
            "Competitor prices are carried from the forecast origin - they are "
            "observed data and cannot be known for a future date.",
        ]
        if model.calibration is not None:
            assumptions.append(
                f"Prediction intervals are split-conformal at "
                f"{model.calibration.nominal_coverage:.0%} nominal, calibrated "
                f"separately per horizon bucket. Coverage was measured on "
                f"held-out data rather than assumed."
            )
        else:
            assumptions.append(
                "No calibrated interval is available, so no bounds are reported."
            )
        assumptions.append(
            "A forecast is predictive, not causal. It says what is likely given "
            "the plan; it does not say what a promotion caused. Incremental "
            "effect is the uplift model's question, not this one."
        )
        return assumptions

    def _warnings(
        self, model: FittedForecastModel, detail: Any, accuracy: ForecastAccuracy
    ) -> list[str]:
        warnings: list[str] = []

        if detail.fallback_rows:
            share = detail.fallback_rows / max(len(detail.frame), 1)
            warnings.append(
                f"{detail.fallback_rows:,} of {len(detail.frame):,} rows "
                f"({share:.0%}) used a fallback rather than the model."
            )

        long_buckets = {
            bucket: value
            for bucket, value in accuracy.bucket_wmape.items()
            if bucket.startswith(("h29", "h57")) and value > 0.5
        }
        if long_buckets:
            formatted = ", ".join(f"{k} {v:.0%}" for k, v in long_buckets.items())
            warnings.append(
                f"Long-horizon accuracy is weak ({formatted}). Treat the far end "
                f"of this forecast as directional and prefer the aggregate over "
                f"individual days."
            )

        if accuracy.test_wmape is not None and accuracy.test_wmape > 0.5:
            warnings.append(
                f"Overall WMAPE is {accuracy.test_wmape:.0%}. Demand here is "
                f"noisy - the irreducible floor measured in Step 4 is ~35% - so "
                f"aggregate before acting and treat single rows as indicative."
            )

        return warnings

    def _dataset_version(self) -> str:
        try:
            return self._repository.dataset_version()
        except Exception:  # noqa: BLE001 - provenance must not break a forecast
            return "unknown"

    def health_check(self) -> tuple[bool, str]:
        if not self.is_available:
            return False, (
                f"no trained forecaster at {self._model_dir} "
                f"(run scripts/train_forecast.py)"
            )
        try:
            model = self.model
        except FileNotFoundError as exc:
            return False, str(exc)
        return True, f"forecast {model.estimator_name} {model.version}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result
