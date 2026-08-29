"""The fitted own-price elasticity model.

Unlike the baseline and forecasting models there is no artifact to persist.
Elasticity is estimated on demand from the panel, because the estimation is
seconds rather than minutes and because the answer depends on the slice being
asked about — an elasticity for one region is a different regression, not a
filter over a stored one.

That makes ``fit`` a no-op and ``predict`` the whole model, which is why this
class is thin. The work is in :mod:`ml.price_elasticity.estimator`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.base import InsufficientDataError, ModelMetadata
from ml.price_elasticity.data import build_elasticity_panel, load_cost_index
from ml.price_elasticity.estimator import (
    MIN_ROWS,
    ElasticityEstimate,
    check_identification,
    comparison_table,
    estimate_all,
    estimation_window,
    prepare_panel,
    select_estimate,
)
from ml.price_elasticity.interface import ElasticityResult, PriceElasticityModel

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"


class FittedElasticityModel(PriceElasticityModel):
    """Own-price elasticity, estimated on request."""

    name = "price_elasticity"

    def __init__(
        self,
        repository: DataRepository,
        *,
        model_version: str = MODEL_VERSION,
    ) -> None:
        super().__init__(repository)
        self.version = model_version
        self._metadata = ModelMetadata(
            name=self.name,
            version=model_version,
            trained_at=datetime.now(UTC),
            approved=False,
        )

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """No-op: there is no artifact.

        Kept rather than raising, because the contract requires it and because
        "this model has nothing to train" is a true and useful answer for a
        caller iterating over every model in the registry.
        """
        return self.metadata

    def predict(  # type: ignore[override]
        self,
        *,
        product_id: str,
        region: str | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ElasticityResult:
        """Estimate own-price elasticity for one product and slice."""
        panel = build_elasticity_panel(
            self._repository,
            product_ids=[product_id],
            store_ids=store_ids,
            region=region,
            start_date=start_date,
            end_date=end_date,
        )
        if panel.empty:
            raise InsufficientDataError(
                f"no sales rows for product {product_id} in this slice"
            )

        categories = (
            sorted(panel["category"].dropna().astype(str).unique())
            if "category" in panel.columns
            else None
        )
        costs = load_cost_index(
            self._repository,
            categories=categories,
            start_date=start_date,
            end_date=end_date,
        )

        frame = prepare_panel(panel, costs=costs if not costs.empty else None)
        if len(frame) < MIN_ROWS:
            raise InsufficientDataError(
                f"only {len(frame)} usable rows for {product_id} after dropping "
                f"promoted, censored and zero-unit days; need at least {MIN_ROWS}. "
                f"Widen the date range or include more stores"
            )

        estimates = estimate_all(frame)
        selected, reason = select_estimate(estimates)
        if selected is None:
            raise InsufficientDataError(
                f"no trustworthy elasticity estimate for {product_id}: {reason}"
            )

        warnings = list(selected.warnings) + check_identification(frame)
        return self._to_result(
            selected,
            product_id=product_id,
            region=region,
            store_ids=store_ids,
            frame=frame,
            reason=reason,
            warnings=warnings,
        )

    def compare_methods(
        self,
        *,
        product_id: str,
        region: str | None = None,
        truth: float | None = None,
    ) -> Any:
        """Every estimator side by side, for the report and for validation."""
        panel = build_elasticity_panel(
            self._repository, product_ids=[product_id], region=region
        )
        categories = (
            sorted(panel["category"].dropna().astype(str).unique())
            if "category" in panel.columns
            else None
        )
        costs = load_cost_index(self._repository, categories=categories)
        frame = prepare_panel(panel, costs=costs if not costs.empty else None)
        return comparison_table(estimate_all(frame), truth=truth)

    def _to_result(
        self,
        estimate: ElasticityEstimate,
        *,
        product_id: str,
        region: str | None,
        store_ids: list[str] | None,
        frame: Any,
        reason: str,
        warnings: list[str],
    ) -> ElasticityResult:
        return ElasticityResult(
            product_id=product_id,
            region=region,
            store_id=store_ids[0] if store_ids and len(store_ids) == 1 else None,
            elasticity=estimate.elasticity,
            confidence_interval=estimate.confidence_interval,
            p_value=estimate.p_value,
            standard_error=estimate.standard_error,
            r_squared=estimate.r_squared,
            sample_size=estimate.n_obs,
            is_elastic=estimate.is_elastic,
            method=estimate.method,
            estimation_window=estimation_window(frame),
            diagnostics={
                **estimate.diagnostics,
                "n_price_points": float(estimate.n_price_points),
            },
            assumptions=[
                reason,
                "Estimated on unpromoted, in-stock days only. A promotion moves "
                "price and applies a mechanic lift at once, so promoted rows "
                "attribute the mechanic to the price cut.",
                "Log-log specification, so the coefficient is a constant "
                "elasticity. Real demand curves bend; this is a local "
                "approximation around the observed price range.",
                "The recorded product elasticity is modulated by a store-level "
                "price-sensitivity multiplier that the platform does not "
                "observe, so a pooled estimate recovers the product elasticity "
                "times the average multiplier.",
            ],
            warnings=warnings,
        )


__all__ = ["MODEL_VERSION", "FittedElasticityModel"]
