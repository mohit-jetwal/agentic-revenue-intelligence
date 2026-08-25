"""Baseline sales service (brief sections 19, 33).

The seam between the model and everything that consumes it. Section 19 is
specific about the split: data loading does not belong in the model class, and
the model does not belong in the caller.

Responsibilities here: load the trained model once, hold it, translate a
business request into a prediction, and return a structured response carrying
the provenance and the assumptions.

**On assumptions.** They are generated from what the model actually did - which
promotion approach it used, whether the interval was calibrated, how many rows
fell back to cold start - rather than written as a fixed list. A hand-written
assumption survives a change to the model and quietly becomes untrue, which is
worse than no assumption at all because it reads as verified.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from app.schemas.baseline import (
    BaselineErrorResponse,
    BaselineMetricsSummary,
    BaselineRecord,
    BaselineRequest,
    BaselineResponse,
)
from data.repositories.base import DataAccessError, DataRepository
from ml.base import InsufficientDataError, ModelNotFittedError
from ml.baseline.model import FittedBaselineModel
from ml.baseline.training import PromotionApproach

logger = get_logger(__name__)


class BaselineSalesService:
    """Serves baseline estimates from a trained model.

    Constructed with a repository and, optionally, an already-loaded model.
    Dependency injection rather than internal construction, so a test can pass a
    model fitted on ten rows and never touch the filesystem.
    """

    def __init__(
        self,
        repository: DataRepository,
        *,
        model: FittedBaselineModel | None = None,
        model_dir: Path | None = None,
        settings: Settings | None = None,
        metrics: BaselineMetricsSummary | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()
        self._model_dir = model_dir or (
            self._settings.project_root / "data" / "local" / "models" / "baseline"
        )
        self._model = model
        self._metrics = metrics or BaselineMetricsSummary()

    @property
    def model(self) -> FittedBaselineModel:
        """The trained model, loaded on first use.

        Lazy so constructing the service - which the DI container does at
        startup - never requires a trained model to exist. A missing model
        should surface when someone asks for a baseline, with a message saying
        how to train one, not as a container failure at boot.
        """
        if self._model is None:
            self._model = FittedBaselineModel.load_from(self._model_dir, self._repository)
            logger.info("baseline_service.model_loaded", directory=str(self._model_dir))
        return self._model

    @property
    def is_available(self) -> bool:
        """Whether a trained model exists, without raising if it does not."""
        if self._model is not None:
            return True
        return (self._model_dir / "model.joblib").is_file()

    # -- prediction ---------------------------------------------------------

    def predict(self, request: BaselineRequest) -> BaselineResponse | BaselineErrorResponse:
        """Estimate baseline sales for a slice.

        Never raises for an expected failure: a missing model, an empty slice or
        a bad date range come back as a structured error with a code, so the
        Step 13 tool wrapper can turn them into a ``ToolResult`` and the Step 16
        supervisor can re-plan around them.
        """
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        if request.start_date > request.end_date:
            return BaselineErrorResponse(
                error_code="invalid_input",
                message=f"start_date {request.start_date} is after end_date {request.end_date}",
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        try:
            model = self.model
        except FileNotFoundError as exc:
            return BaselineErrorResponse(
                error_code="model_not_found",
                message=str(exc),
                # Not recoverable by re-planning: no different request produces
                # a baseline until someone trains one.
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )

        try:
            detail = model.predict_detail(
                start_date=request.start_date,
                end_date=request.end_date,
                product_ids=request.product_ids,
                store_ids=request.store_ids,
                region=request.region,
                as_of_date=request.as_of_date,
            )
        except InsufficientDataError as exc:
            return BaselineErrorResponse(
                error_code="insufficient_data",
                message=str(exc),
                recoverable=True,
                detail={
                    "start_date": str(request.start_date),
                    "end_date": str(request.end_date),
                    "products": len(request.product_ids) if request.product_ids else None,
                },
                execution_time_ms=elapsed_ms(),
            )
        except ModelNotFittedError as exc:
            return BaselineErrorResponse(
                error_code="model_not_fitted",
                message=str(exc),
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )
        except (DataAccessError, ValueError) as exc:
            logger.warning("baseline_service.failed", error=str(exc))
            return BaselineErrorResponse(
                error_code="prediction_failed",
                message=str(exc),
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        aggregate = detail.aggregate(
            start_date=request.start_date,
            end_date=request.end_date,
            product_id=(
                request.product_ids[0]
                if request.product_ids and len(request.product_ids) == 1
                else None
            ),
            store_id=(
                request.store_ids[0]
                if request.store_ids and len(request.store_ids) == 1
                else None
            ),
            region=request.region,
        )

        records: list[BaselineRecord] = []
        if request.include_records:
            frame = detail.frame.head(request.max_records)
            records = [
                BaselineRecord(
                    date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                    product_id=str(row["product_id"]),
                    store_id=str(row["store_id"]),
                    actual_units=float(row["actual_units"]),
                    baseline_units=float(row["baseline_units"]),
                    sales_gap=float(row["sales_gap"]),
                    sales_gap_pct=_optional_float(row.get("sales_gap_pct")),
                    baseline_lower=_optional_float(row.get("baseline_lower")),
                    baseline_upper=_optional_float(row.get("baseline_upper")),
                    is_significant=_optional_bool(row.get("is_significant")),
                    promotion_flag=_optional_bool(row.get("promotion_flag")),
                    stockout_flag=_optional_bool(row.get("stockout_flag")),
                    fallback_used=bool(row.get("fallback_used", False)),
                )
                for row in frame.to_dict(orient="records")
            ]

        response = BaselineResponse(
            model_name=detail.model_name,
            model_version=detail.model_version,
            dataset_version=detail.dataset_version,
            feature_version=detail.feature_version,
            result=aggregate,
            records=records,
            record_count=len(detail.frame),
            fallback_rows=detail.fallback_rows,
            metrics=self._metrics,
            assumptions=self._assumptions(model, detail),
            warnings=self._warnings(model, detail, request),
            execution_time_ms=elapsed_ms(),
        )

        logger.info(
            "baseline_service.predicted",
            rows=response.record_count,
            baseline_units=round(aggregate.baseline_units, 1),
            actual_units=round(aggregate.actual_units, 1),
            duration_ms=response.execution_time_ms,
        )
        return response

    # -- narrative ----------------------------------------------------------

    def _assumptions(self, model: FittedBaselineModel, detail: Any) -> list[str]:
        """Assumptions derived from what the model actually did."""
        assumptions = [
            "Baseline is expected demand under normal trading conditions - no "
            "promotion running and stock available.",
            "Estimated from historical demand, seasonality, price, competitor "
            "position and store/product characteristics.",
        ]

        if model.approach is PromotionApproach.EXCLUDE:
            assumptions.append(
                "Trained on non-promotional rows only, so the prediction is a "
                "genuine no-promotion counterfactual. Because promotions are "
                "scheduled toward seasonal peaks, this may understate baseline "
                "during peaks - and therefore overstate uplift."
            )
        else:
            assumptions.append(
                "Trained on all rows with promotion features as controls, then "
                "predicted with those features zeroed. Avoids selection bias, but "
                "relies on the model extrapolating to a no-promotion state for "
                "rows that were usually promoted."
            )

        assumptions.append(
            "Stockout rows were excluded from training, so the baseline estimates "
            "demand rather than inventory-constrained sales."
        )

        if model.calibration is not None:
            assumptions.append(
                f"Prediction intervals are split-conformal at "
                f"{model.calibration.nominal_coverage:.0%} nominal coverage, "
                f"calibrated on a held-out fold. Coverage was measured on test "
                f"data rather than assumed."
            )
        else:
            assumptions.append(
                "No prediction interval is available for this model, so "
                "`is_significant` is not populated."
            )

        assumptions.append(
            "A gap between actual and baseline is not automatically causal uplift. "
            "Attributing it to a promotion requires the causal assumptions the "
            "uplift model tests, not merely a difference."
        )
        return assumptions

    def _warnings(
        self, model: FittedBaselineModel, detail: Any, request: BaselineRequest
    ) -> list[str]:
        """Caveats the caller must surface, generated from the actual result."""
        warnings: list[str] = []

        if detail.fallback_rows:
            share = detail.fallback_rows / max(len(detail.frame), 1)
            warnings.append(
                f"{detail.fallback_rows:,} of {len(detail.frame):,} rows "
                f"({share:.0%}) used the cold-start fallback - a category x channel "
                f"mean rather than the model - because the series lacks sufficient "
                f"history."
            )

        if "stockout_flag" in detail.frame.columns:
            stockouts = int(detail.frame["stockout_flag"].astype(bool).sum())
            if stockouts:
                warnings.append(
                    f"{stockouts:,} rows in this window were stockouts, so observed "
                    f"sales there understate demand. The baseline estimates demand; "
                    f"the difference is lost sales, not a demand decline."
                )

        if "promotion_flag" in detail.frame.columns:
            promoted = int(detail.frame["promotion_flag"].astype(bool).sum())
            if promoted:
                warnings.append(
                    f"{promoted:,} rows were promotional. Actual exceeding baseline "
                    f"there is expected and is the uplift signal, not model error."
                )

        if self._metrics.test_wmape is not None and self._metrics.test_wmape > 0.25:
            warnings.append(
                f"The model's test WMAPE is {self._metrics.test_wmape:.0%}, which is "
                f"high. Treat individual row estimates as indicative and prefer "
                f"aggregates."
            )

        if self._metrics.backtest_stable is False:
            warnings.append(
                "Backtest accuracy varies materially between quarters, so a single "
                "headline accuracy figure understates the uncertainty."
            )

        return warnings

    def health_check(self) -> tuple[bool, str]:
        """Probe for ``GET /health``."""
        if not self.is_available:
            return False, f"no trained baseline model at {self._model_dir} (run Step 4 training)"
        try:
            model = self.model
        except FileNotFoundError as exc:
            return False, str(exc)
        return True, (
            f"baseline {model.estimator.name if model.estimator else '?'} "
            f"{model.version} ({model.approach.value})"
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN check without importing numpy


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
