"""Elasticity service.

The seam between the estimators and everything that consumes them. Same contract
as the forecasting and uplift services: validate, estimate, attach provenance,
return a structured result **or** a structured refusal.

Unlike those two there is no persisted artifact. Elasticity is estimated on
demand from the panel, because the estimation takes seconds and because the
answer depends on the slice: an elasticity for one region is a different
regression, not a filter over a stored one.
"""

from __future__ import annotations

import time
from typing import Any

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from app.schemas.elasticity import (
    CrossPriceRecord,
    ElasticityErrorResponse,
    ElasticityRequest,
    ElasticityResponse,
    MethodEstimate,
)
from data.repositories.base import DataAccessError, DataRepository
from features.contracts.specs import FEATURE_VERSION
from ml.base import InsufficientDataError
from ml.cross_price_elasticity.model import FittedCrossElasticityModel
from ml.price_elasticity.model import FittedElasticityModel

logger = get_logger(__name__)


class ElasticityService:
    """Serves own-price and cross-price elasticity estimates."""

    def __init__(
        self,
        repository: DataRepository,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()
        self._own: FittedElasticityModel | None = None
        self._cross: FittedCrossElasticityModel | None = None

    @property
    def own_model(self) -> FittedElasticityModel:
        if self._own is None:
            self._own = FittedElasticityModel(self._repository)
        return self._own

    @property
    def cross_model(self) -> FittedCrossElasticityModel:
        if self._cross is None:
            self._cross = FittedCrossElasticityModel(self._repository)
        return self._cross

    @property
    def is_available(self) -> bool:
        """Always true: nothing to load.

        Reported honestly rather than probing for an artifact that does not
        exist, so the health endpoint does not imply a model file is missing.
        """
        return True

    def health_check(self) -> tuple[bool, str]:
        return True, "elasticity (estimated on demand, no artifact)"

    # -- estimation ---------------------------------------------------------

    def estimate(
        self, request: ElasticityRequest
    ) -> ElasticityResponse | ElasticityErrorResponse:
        """Estimate elasticity, or return a structured reason why not."""
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        if request.start_date and request.end_date and request.start_date > request.end_date:
            return ElasticityErrorResponse(
                error_code="invalid_input",
                message=(
                    f"start_date ({request.start_date}) is after end_date "
                    f"({request.end_date})"
                ),
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        try:
            result = self.own_model.predict(
                product_id=request.product_id,
                region=request.region,
                store_ids=request.store_ids,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        except InsufficientDataError as exc:
            return ElasticityErrorResponse(
                error_code="insufficient_data",
                message=str(exc),
                # A wider date range or more stores would supply the price
                # variation this slice lacks.
                recoverable=True,
                detail={"product_id": request.product_id, "region": request.region},
                execution_time_ms=elapsed_ms(),
            )
        except (DataAccessError, ValueError) as exc:
            logger.warning("elasticity_service.failed", error=str(exc))
            return ElasticityErrorResponse(
                error_code="elasticity_failed",
                message=str(exc),
                recoverable=True,
                execution_time_ms=elapsed_ms(),
            )

        response = ElasticityResponse(
            model_name=self.own_model.name,
            model_version=self.own_model.version,
            dataset_version=self._dataset_version(),
            feature_version=FEATURE_VERSION,
            product_id=result.product_id,
            region=result.region,
            elasticity=result.elasticity,
            is_elastic=bool(result.is_elastic),
            confidence_interval=result.confidence_interval,
            standard_error=result.standard_error,
            p_value=result.p_value,
            r_squared=result.r_squared,
            sample_size=result.sample_size,
            method=result.method or "unknown",
            method_reason=result.assumptions[0] if result.assumptions else "",
            estimation_window=result.estimation_window,
            diagnostics=result.diagnostics,
            assumptions=list(result.assumptions),
            warnings=list(result.warnings),
            execution_time_ms=elapsed_ms(),
        )

        if request.include_comparison:
            response.comparison = self._comparison(request)
        if request.include_cross_price:
            self._attach_cross_price(response, request)

        response.execution_time_ms = elapsed_ms()
        return response

    # -- internals ----------------------------------------------------------

    def _comparison(self, request: ElasticityRequest) -> list[MethodEstimate]:
        """Every estimator side by side. Failures are omitted, not faked."""
        try:
            frame = self.own_model.compare_methods(
                product_id=request.product_id, region=request.region
            )
        except (InsufficientDataError, DataAccessError, ValueError) as exc:
            logger.info("elasticity_service.comparison_unavailable", error=str(exc))
            return []

        return [
            MethodEstimate(
                method=str(row["method"]),
                elasticity=float(row["elasticity"]),
                standard_error=self._optional_float(row.get("std_error")),
                ci_lower=self._optional_float(row.get("ci_lower")),
                ci_upper=self._optional_float(row.get("ci_upper")),
                n_obs=int(row["n_obs"]),
                selectable=bool(row["selectable"]),
            )
            for row in frame.to_dict("records")
        ]

    def _attach_cross_price(
        self, response: ElasticityResponse, request: ElasticityRequest
    ) -> None:
        """Cross-price relationships, or a warning saying why not.

        A failure here does not fail the request: own-price elasticity is the
        headline and stands on its own.
        """
        try:
            cross = self.cross_model.predict(
                product_id=request.product_id,
                region=request.region,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        except (InsufficientDataError, DataAccessError, ValueError) as exc:
            response.warnings.append(f"cross-price relationships unavailable: {exc}")
            return

        response.cross_price = [
            CrossPriceRecord(
                source_product_id=pair.source_product_id,
                cross_elasticity=pair.cross_elasticity,
                relationship_type=str(pair.relationship_type),
                strength=pair.strength,
                p_value=pair.p_value,
                is_significant=pair.is_significant,
                sample_size=pair.sample_size,
            )
            for pair in cross.pairs
        ]
        response.substitutes = list(cross.substitutes)
        response.complements = list(cross.complements)
        response.pairs_tested = cross.pairs_tested
        response.assumptions.extend(cross.assumptions)
        response.warnings.extend(cross.warnings)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return None if result != result else result  # NaN check

    def _dataset_version(self) -> str:
        try:
            return self._repository.dataset_version()
        except (DataAccessError, AttributeError, OSError):
            return "unknown"


__all__ = ["ElasticityService"]
