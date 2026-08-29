"""The price optimiser, wired to the elasticity estimates it depends on."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.base import InsufficientDataError, ModelMetadata
from ml.cross_price_elasticity.model import FittedCrossElasticityModel
from ml.price_elasticity.model import FittedElasticityModel
from ml.price_optimization.interface import (
    PriceOptimizationModel,
    PriceOptimizationResult,
    PriceScenario,
)
from ml.price_optimization.optimizer import (
    DEFAULT_GRID,
    PriceRecommendation,
    evaluate_candidates,
    recommend,
)

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"


class FittedPriceOptimizer(PriceOptimizationModel):
    """Recommends a price range, given a measured elasticity."""

    name = "price_optimization"

    def __init__(
        self,
        repository: DataRepository,
        *,
        elasticity_model: FittedElasticityModel | None = None,
        cross_model: FittedCrossElasticityModel | None = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        super().__init__(repository)
        self.version = model_version
        self._elasticity = elasticity_model or FittedElasticityModel(repository)
        self._cross = cross_model or FittedCrossElasticityModel(repository)
        self._metadata = ModelMetadata(
            name=self.name, version=model_version, trained_at=datetime.now(UTC), approved=False
        )

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """No-op: solved fresh from the elasticity on every call."""
        return self.metadata

    def predict(  # type: ignore[override]
        self,
        *,
        product_id: str,
        region: str | None = None,
        objective: str = "profit",
        min_price: float | None = None,
        max_price: float | None = None,
        min_margin_pct: float | None = None,
        max_price_change_pct: float | None = None,
        candidate_changes_pct: list[float] | None = None,
    ) -> PriceOptimizationResult:
        """Evaluate candidate prices and recommend one, with its range."""
        elasticity = self._elasticity.predict(product_id=product_id, region=region)
        current_price, current_units, unit_cost = self._current_economics(
            product_id, region
        )

        cross_effects = self._cross_effects(product_id, region)

        candidates = evaluate_candidates(
            current_price=current_price,
            current_units=current_units,
            unit_cost=unit_cost,
            elasticity=elasticity.elasticity,
            grid=tuple(candidate_changes_pct) if candidate_changes_pct else DEFAULT_GRID,
            min_price=min_price,
            max_price=max_price,
            min_margin_pct=min_margin_pct,
            max_change_pct=max_price_change_pct,
            cross_effects=cross_effects,
        )
        recommendation = recommend(
            product_id,
            candidates,
            current_price=current_price,
            elasticity=elasticity.elasticity,
            elasticity_interval=elasticity.confidence_interval,
            objective=objective,
        )
        return self._to_result(recommendation, elasticity_method=elasticity.method or "unknown")

    def _current_economics(
        self, product_id: str, region: str | None
    ) -> tuple[float, float, float]:
        """Current price, daily units and unit cost.

        Taken from recent unpromoted days: a promoted day's price is not the
        price being optimised, and including it would anchor the optimum to a
        discount rather than to the shelf price.
        """
        sales = self._repository.get_sales(
            product_ids=[product_id], region=region, max_rows=2_000_000
        )
        if sales.empty:
            raise InsufficientDataError(f"no sales history for {product_id}")

        sales["date"] = pd.to_datetime(sales["date"])
        recent = sales[sales["date"] >= sales["date"].max() - pd.Timedelta(days=90)]
        if "promotion_flag" in recent.columns:
            unpromoted = recent[~recent["promotion_flag"].astype(bool)]
            recent = unpromoted if not unpromoted.empty else recent

        units = float(recent["units"].sum())
        if units <= 0:
            raise InsufficientDataError(
                f"no unpromoted sales for {product_id} in the last 90 days"
            )

        price_column = "regular_price" if "regular_price" in recent.columns else "selling_price"
        price = float(recent[price_column].mean())
        cost = (
            float(recent["cost"].sum()) / units
            if "cost" in recent.columns and recent["cost"].notna().any()
            else price * 0.62
        )
        daily_units = units / max(recent["date"].nunique(), 1)
        return price, daily_units, cost

    def _cross_effects(
        self, product_id: str, region: str | None
    ) -> dict[str, tuple[float, float, float]]:
        """Significant substitutes and complements, with their economics.

        Failure here degrades the answer rather than blocking it - but the
        result says so, because optimising in isolation reliably recommends a
        rise that moves volume to a neighbour and books a phantom gain.
        """
        try:
            cross = self._cross.predict(product_id=product_id, region=region)
        except (InsufficientDataError, ValueError) as exc:
            logger.info("price_optimization.cross_unavailable", error=str(exc))
            return {}

        effects: dict[str, tuple[float, float, float]] = {}
        for pair in cross.pairs:
            if not pair.is_significant:
                continue
            try:
                price, units, cost = self._current_economics(
                    pair.source_product_id, region
                )
            except InsufficientDataError:
                continue
            effects[pair.source_product_id] = (pair.cross_elasticity, units, price - cost)
        return effects

    def _to_result(
        self, recommendation: PriceRecommendation, *, elasticity_method: str
    ) -> PriceOptimizationResult:
        scenarios = [
            PriceScenario(
                label=candidate.label,
                price=candidate.price,
                price_change_pct=candidate.change_pct,
                expected_units=max(candidate.units, 0.0),
                expected_revenue=candidate.revenue,
                expected_profit=candidate.profit,
                expected_margin_pct=candidate.margin_pct,
                cannibalisation_units=candidate.cannibalisation_units,
                net_portfolio_profit=candidate.net_portfolio_profit,
                risk=recommendation.risk,
                constraint_violations=candidate.violations,
            )
            for candidate in recommendation.candidates
        ]

        warnings = list(recommendation.warnings)
        if not any(c.cannibalisation_units for c in recommendation.candidates):
            warnings.append(
                "no significant cross-price relationships were found, so this "
                "optimum ignores portfolio spillover. A rise that simply moves "
                "volume to a neighbour would not be detected"
            )

        return PriceOptimizationResult(
            product_id=recommendation.product_id,
            current_price=recommendation.current_price,
            recommended_price=recommendation.recommended_price,
            recommended_range=recommendation.recommended_range,
            recommended_change_pct=recommendation.change_pct,
            scenarios=scenarios,
            elasticity_used=recommendation.elasticity,
            elasticity_confidence_interval=recommendation.elasticity_interval,
            expected_revenue_impact=recommendation.revenue_impact,
            expected_profit_impact=recommendation.profit_impact,
            risk=recommendation.risk,
            binding_constraints=recommendation.binding_constraints,
            assumptions=[
                f"Elasticity of {recommendation.elasticity:.2f} identified by "
                f"{elasticity_method}.",
                "Constant-elasticity demand, which is a local approximation "
                "around observed prices. A recommendation far outside the "
                "historical range is extrapolation.",
                "Scored on net portfolio profit, not this product's alone: a "
                "rise that moves volume to a substitute you also own has not "
                "created anything.",
                "The recommended RANGE, not the point, is the output. An optimum "
                "computed from an elasticity with a wide interval is a point "
                "estimate of a point estimate.",
            ],
            warnings=warnings,
        )


__all__ = ["MODEL_VERSION", "FittedPriceOptimizer"]
