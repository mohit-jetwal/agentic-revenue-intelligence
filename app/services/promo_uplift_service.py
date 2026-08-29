"""Promo uplift service (brief section 29).

The seam between the causal machinery and everything that consumes it. Same
contract as :mod:`app.services.forecast_service`: validate, load, estimate,
attach provenance, return a structured result **or** a structured refusal.

Two things differ, and both come from the subject matter.

**A persisted analysis is served, not a live estimate.** Uplift asks about
promotions that already ran, and the answer does not change between requests. A
full run - control construction, cross-fitted nuisance models, placebo,
sensitivity - takes minutes, so it happens once in ``scripts/estimate_uplift.py``
and the service reads the result. An ad-hoc slice re-runs on demand and says so
in its warnings.

**A refusal here is a stronger statement than a forecasting refusal.** A forecast
that cannot be produced is an inconvenience. An uplift estimate that cannot be
*identified* means the data does not answer the question, and returning a number
anyway would be worse than returning nothing. Hence
``validation_status: failed``, which carries the estimate *and* the reason it
must not be called causal - because withholding it entirely does not stop anyone
computing a naive number instead.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from app.schemas.promo_uplift import (
    MethodComparisonRecord,
    UpliftDiagnostics,
    UpliftErrorResponse,
    UpliftEventRecord,
    UpliftIntervalRecord,
    UpliftRequest,
    UpliftResponse,
    UpliftSegmentRecord,
)
from data.repositories.base import DataAccessError, DataRepository
from features.contracts.specs import FEATURE_VERSION
from ml.base import InsufficientDataError, ModelNotFittedError
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import (
    CausalAssumptionsViolatedError,
    PromoUpliftError,
    UnknownPromotionError,
)
from ml.promo_uplift.model import FittedUpliftModel

logger = get_logger(__name__)


class PromoUpliftService:
    """Serves causal uplift estimates from a completed analysis."""

    def __init__(
        self,
        repository: DataRepository,
        *,
        model: FittedUpliftModel | None = None,
        model_dir: Path | None = None,
        settings: Settings | None = None,
        config: PromoUpliftConfig | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()
        self._config = config or get_promo_uplift_config()
        self._model_dir = model_dir or self._default_model_dir()
        self._model = model

    def _default_model_dir(self) -> Path:
        """Prefer the full analysis, fall back to a sampled one."""
        root = self._settings.project_root / "data" / "local" / "models"
        full = root / "promo_uplift"
        if (full / "uplift.joblib").is_file():
            return full
        return root / "promo_uplift_sampled"

    @property
    def model(self) -> FittedUpliftModel:
        """The persisted analysis, loaded on first use.

        Lazy, so constructing the service at container startup never requires an
        analysis to exist. A missing artifact should surface when someone asks
        for uplift, with the command that produces one - not as a boot failure.
        """
        if self._model is None:
            self._model = FittedUpliftModel.load_from(
                self._model_dir, self._repository, config=self._config
            )
            logger.info("promo_uplift_service.model_loaded", directory=str(self._model_dir))
        return self._model

    @property
    def is_available(self) -> bool:
        if self._model is not None:
            return True
        return (self._model_dir / "uplift.joblib").is_file()

    # -- estimation ---------------------------------------------------------

    def estimate_uplift(
        self, request: UpliftRequest
    ) -> UpliftResponse | UpliftErrorResponse:
        """Return the causal estimate, or a structured reason why not."""
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        invalid = self._validate(request)
        if invalid is not None:
            return UpliftErrorResponse(
                error_code="invalid_input",
                message=invalid,
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        try:
            model = self.model
        except FileNotFoundError as exc:
            return UpliftErrorResponse(
                error_code="model_not_found",
                message=str(exc),
                # No reformulation helps until an analysis has been run.
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )

        try:
            events = self._select_events(model, request)
        except UnknownPromotionError as exc:
            return UpliftErrorResponse(
                error_code=exc.code,
                message=exc.message,
                recoverable=exc.recoverable,
                detail=exc.detail,
                execution_time_ms=elapsed_ms(),
            )

        if events.empty:
            return UpliftErrorResponse(
                error_code="insufficient_data",
                message=(
                    "no analysed promotion matches this request. The analysis "
                    "covers the promotions present when it was run; a promotion "
                    "outside that window needs a re-run"
                ),
                recoverable=True,
                detail=self._request_detail(request),
                execution_time_ms=elapsed_ms(),
            )

        try:
            return self._build_response(model, events, request, elapsed_ms())
        except CausalAssumptionsViolatedError as exc:
            return UpliftErrorResponse(
                error_code=exc.code,
                message=exc.message,
                recoverable=exc.recoverable,
                detail=exc.detail,
                execution_time_ms=elapsed_ms(),
            )
        except (InsufficientDataError, PromoUpliftError) as exc:
            code = getattr(exc, "code", "insufficient_data")
            return UpliftErrorResponse(
                error_code=code,
                message=str(exc),
                recoverable=getattr(exc, "recoverable", True),
                detail=getattr(exc, "detail", {}),
                execution_time_ms=elapsed_ms(),
            )
        except ModelNotFittedError as exc:
            return UpliftErrorResponse(
                error_code="model_not_fitted",
                message=str(exc),
                recoverable=False,
                execution_time_ms=elapsed_ms(),
            )
        except (DataAccessError, ValueError) as exc:
            logger.warning("promo_uplift_service.failed", error=str(exc))
            return UpliftErrorResponse(
                error_code="uplift_failed",
                message=str(exc),
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _validate(request: UpliftRequest) -> str | None:
        if (
            request.analysis_start_date
            and request.analysis_end_date
            and request.analysis_start_date > request.analysis_end_date
        ):
            return (
                f"analysis_start_date ({request.analysis_start_date}) is after "
                f"analysis_end_date ({request.analysis_end_date})"
            )
        return None

    def _select_events(
        self, model: FittedUpliftModel, request: UpliftRequest
    ) -> pd.DataFrame:
        """Filter the analysed events, refusing unknown identifiers.

        An unknown promotion id is refused rather than answered with an empty
        result. "This promotion had no effect" and "this promotion is not in the
        analysis" would otherwise be indistinguishable, and a category manager
        acting on the first when the second is true concludes a mechanic does
        not work on no evidence at all.
        """
        events = model.event_impact
        if events.empty:
            return events

        if request.promotion_ids:
            known = set(events["promotion_id"])
            unknown = [p for p in request.promotion_ids if p not in known]
            if unknown:
                raise UnknownPromotionError(
                    f"{len(unknown)} promotion id(s) are not in the analysis: "
                    f"{', '.join(unknown[:5])}",
                    promotion_ids=unknown,
                )
            events = events[events["promotion_id"].isin(request.promotion_ids)]

        for column, values in (
            ("product_id", request.product_ids),
            ("store_id", request.store_ids),
        ):
            if values:
                events = events[events[column].isin(values)]

        for column, value in (("region", request.region), ("category", request.category)):
            if value and column in events.columns:
                events = events[events[column] == value]

        if request.analysis_start_date and "start_date" in events.columns:
            events = events[
                pd.to_datetime(events["start_date"])
                >= pd.to_datetime(request.analysis_start_date)
            ]
        if request.analysis_end_date and "end_date" in events.columns:
            events = events[
                pd.to_datetime(events["end_date"])
                <= pd.to_datetime(request.analysis_end_date)
            ]
        return events

    def _build_response(
        self,
        model: FittedUpliftModel,
        events: pd.DataFrame,
        request: UpliftRequest,
        elapsed: int,
    ) -> UpliftResponse:
        artifact = model.artifact
        headline = model.headline

        incremental_units = float(events["incremental_units"].sum())
        observed = float(events["observed_units"].sum())
        baseline = max(observed - incremental_units, 0.0)
        spend = float(events["promotion_spend"].fillna(0.0).sum())
        profit = float(events["incremental_profit"].sum())

        interval = None
        if headline is not None:
            band = headline.interval_pct()
            if band is not None and headline.confidence_level is not None:
                interval = UpliftIntervalRecord(
                    lower=band[0],
                    upper=band[1],
                    confidence_level=headline.confidence_level,
                )

        warnings = list(artifact.warnings)
        if not model.event_impact.empty and len(events) < len(model.event_impact):
            warnings.append(
                f"this response covers {len(events):,} of "
                f"{len(model.event_impact):,} analysed promotions. The interval "
                f"and diagnostics describe the full analysis, not this subset"
            )

        response = UpliftResponse(
            model_name=model.name,
            model_version=model.version,
            dataset_version=artifact.dataset_version or self._dataset_version(),
            feature_version=FEATURE_VERSION,
            treatment_definition=artifact.treatment_definition,
            method=artifact.selected or "none",
            method_reason=artifact.selection_reason,
            baseline_units=baseline,
            observed_units=observed,
            incremental_units=incremental_units,
            uplift_pct=incremental_units / baseline if baseline > 0 else 0.0,
            incremental_revenue=float(events["incremental_revenue"].sum()),
            incremental_profit=profit,
            promotion_spend=spend,
            roi=profit / spend if spend > 0 else None,
            confidence_interval=interval,
            events_analysed=len(events),
            treated_days=int(events["treated_days"].sum()),
            validation_status=artifact.validation_status,
            assumptions=list(headline.assumptions) if headline else [],
            warnings=warnings,
            execution_time_ms=elapsed,
        )

        if request.include_events:
            response.events = self._event_records(events, request.max_events)
        if request.include_segments:
            response.segments = self._segment_records(artifact.segments)
        response.comparison = self._comparison_records(artifact)
        response.diagnostics = self._diagnostics(artifact)
        return response

    @staticmethod
    def _event_records(events: pd.DataFrame, limit: int) -> list[UpliftEventRecord]:
        # Ranked by ROI so a truncated list keeps the most decision-relevant
        # rows. Value-destroying promotions sort last but are not filtered -
        # they are the ones a budget should move away from.
        ordered = events.sort_values("roi", ascending=False, na_position="last")
        # `to_dict("records")` rather than `itertuples`: the latter types every
        # attribute as a wide union that `int()` and `float()` reject, and
        # casting each one individually is noisier than this.
        return [
            UpliftEventRecord(
                promotion_id=str(row["promotion_id"]),
                product_id=str(row["product_id"]),
                store_id=str(row["store_id"]),
                treated_days=int(row["treated_days"]),
                incremental_units=float(row["incremental_units"]),
                incremental_revenue=float(row["incremental_revenue"]),
                incremental_profit=float(row["incremental_profit"]),
                promotion_spend=float(row["promotion_spend"])
                if pd.notna(row["promotion_spend"])
                else None,
                roi=float(row["roi"]) if pd.notna(row["roi"]) else None,
                value_destroying=bool(row["value_destroying"]),
            )
            for row in ordered.head(limit).to_dict("records")
        ]

    @staticmethod
    def _segment_records(
        segments: dict[str, pd.DataFrame],
    ) -> list[UpliftSegmentRecord]:
        records: list[UpliftSegmentRecord] = []
        for dimension, frame in segments.items():
            if frame.empty:
                continue
            for row in frame.to_dict("records"):
                records.append(
                    UpliftSegmentRecord(
                        segment=str(row["segment"]),
                        dimension=dimension,
                        n_treated=int(row["n_treated"]),
                        uplift_pct=float(row["uplift_pct"])
                        if pd.notna(row["uplift_pct"])
                        else None,
                        classification=str(row.get("classification", "uncertain")),
                        action=str(row.get("action", "")),
                        estimable=bool(row.get("estimable", True)),
                    )
                )
        return records

    @staticmethod
    def _comparison_records(artifact: Any) -> list[MethodComparisonRecord]:
        records = []
        for name, estimate in artifact.estimates.items():
            band = estimate.interval_pct()
            records.append(
                MethodComparisonRecord(
                    method=name,
                    uplift_pct=estimate.ate_pct,
                    ci_lower_pct=band[0] if band else None,
                    ci_upper_pct=band[1] if band else None,
                    eligible=name != "naive_during_vs_before",
                )
            )
        return records

    @staticmethod
    def _diagnostics(artifact: Any) -> UpliftDiagnostics:
        headline = artifact.estimates.get(artifact.selected) if artifact.selected else None
        diagnostics = headline.diagnostics if headline else {}
        return UpliftDiagnostics(
            effective_sample_fraction=diagnostics.get("effective_sample_fraction"),
            propensity_auc=diagnostics.get("propensity_auc"),
        )

    @staticmethod
    def _request_detail(request: UpliftRequest) -> dict[str, Any]:
        return {
            "promotion_ids": request.promotion_ids,
            "product_ids": request.product_ids,
            "store_ids": request.store_ids,
            "analysis_start_date": str(request.analysis_start_date)
            if request.analysis_start_date
            else None,
            "analysis_end_date": str(request.analysis_end_date)
            if request.analysis_end_date
            else None,
        }

    def _dataset_version(self) -> str:
        try:
            return self._repository.dataset_version()
        except (DataAccessError, AttributeError, OSError):
            return "unknown"


def latest_analysis_window(model: FittedUpliftModel) -> tuple[date, date] | None:
    """The date range the persisted analysis covers.

    Returned so a refusal can say what *would* have worked, the same way the
    forecasting service reports its latest usable as-of.
    """
    events = model.event_impact
    if events.empty or "start_date" not in events.columns:
        return None
    return (
        pd.to_datetime(events["start_date"]).min().date(),
        pd.to_datetime(events["end_date"]).max().date(),
    )


__all__ = ["PromoUpliftService", "latest_analysis_window"]
