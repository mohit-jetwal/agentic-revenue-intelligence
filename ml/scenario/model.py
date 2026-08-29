"""The scenario engine, composing the other models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.base import InsufficientDataError, ModelMetadata
from ml.cross_price_elasticity.model import FittedCrossElasticityModel
from ml.price_elasticity.model import FittedElasticityModel
from ml.scenario.engine import Inputs, Lever, Projection, project
from ml.scenario.interface import (
    ScenarioEngine,
    ScenarioLever,
    ScenarioOutcome,
    ScenarioResult,
)

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"


class FittedScenarioEngine(ScenarioEngine):
    """Projects interventions by composing elasticity, cross-price and uplift."""

    name = "scenario_simulation"

    def __init__(
        self,
        repository: DataRepository,
        *,
        elasticity_model: FittedElasticityModel | None = None,
        cross_model: FittedCrossElasticityModel | None = None,
        default_uplift: float | None = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        super().__init__(repository)
        self.version = model_version
        self._elasticity = elasticity_model or FittedElasticityModel(repository)
        self._cross = cross_model or FittedCrossElasticityModel(repository)
        self._default_uplift = default_uplift
        self._metadata = ModelMetadata(
            name=self.name, version=model_version, trained_at=datetime.now(UTC), approved=False
        )

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """No-op: composes other models, fits nothing."""
        return self.metadata

    def predict(  # type: ignore[override]
        self,
        *,
        levers: list[ScenarioLever],
        horizon_days: int = 30,
        product_ids: list[str] | None = None,
        region: str | None = None,
        scenario_name: str = "scenario",
        description: str = "",
    ) -> ScenarioResult:
        """Simulate the combined effect of the levers."""
        if not product_ids:
            raise InsufficientDataError(
                "a scenario needs at least one product: the projection runs "
                "through a specific demand curve, not an aggregate"
            )

        product_id = product_ids[0]
        inputs = self._gather(product_id, region, levers)
        projection = project(
            [self._to_lever(lever) for lever in levers], inputs, horizon_days=horizon_days
        )
        return self._to_result(
            projection,
            levers=levers,
            horizon_days=horizon_days,
            scenario_name=scenario_name,
            description=description,
        )

    def compare(
        self,
        *,
        scenarios: list[list[ScenarioLever]],
        horizon_days: int = 30,
        product_ids: list[str] | None = None,
        region: str | None = None,
    ) -> list[ScenarioResult]:
        """Simulate several scenarios against **one** baseline.

        Separate from ``predict`` because a like-for-like comparison requires a
        shared baseline. Computing each independently lets baseline drift
        masquerade as a difference between the options - the inputs are gathered
        once here and reused.
        """
        if not product_ids:
            raise InsufficientDataError("a comparison needs at least one product")

        product_id = product_ids[0]
        # Gathered once. This is the whole point of the method.
        inputs = self._gather(product_id, region, [lever for s in scenarios for lever in s])

        results = []
        for index, levers in enumerate(scenarios):
            projection = project(
                [self._to_lever(lever) for lever in levers], inputs, horizon_days=horizon_days
            )
            results.append(
                self._to_result(
                    projection,
                    levers=levers,
                    horizon_days=horizon_days,
                    scenario_name=f"option_{index + 1}",
                    description=", ".join(self._to_lever(lever).describe() for lever in levers),
                )
            )
        return results

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _to_lever(lever: ScenarioLever) -> Lever:
        return Lever(
            lever=lever.lever,
            change_pct=lever.change_pct,
            change_absolute=lever.change_absolute,
            product_id=lever.product_id,
            region=lever.region,
        )

    def _gather(
        self, product_id: str, region: str | None, levers: list[ScenarioLever]
    ) -> Inputs:
        """Collect every input the projection needs.

        Only what the levers actually require: estimating an elasticity for a
        promotion-only scenario would spend a minute to produce a number nothing
        reads.
        """
        sales = self._repository.get_sales(
            product_ids=[product_id], region=region, max_rows=2_000_000
        )
        if sales.empty:
            raise InsufficientDataError(f"no sales history for {product_id}")

        sales["date"] = pd.to_datetime(sales["date"])
        recent = sales[sales["date"] >= sales["date"].max() - pd.Timedelta(days=90)]
        units = float(recent["units"].sum())
        days = max(recent["date"].nunique(), 1)

        price_column = "regular_price" if "regular_price" in recent.columns else "selling_price"
        price = float(recent[price_column].mean())
        cost = (
            float(recent["cost"].sum()) / units
            if units > 0 and "cost" in recent.columns and recent["cost"].notna().any()
            else price * 0.62
        )

        wants = {lever.lever for lever in levers}
        elasticity = interval = None
        cross_effects: dict[str, tuple[float, float, float]] = {}

        if "price" in wants:
            try:
                estimate = self._elasticity.predict(product_id=product_id, region=region)
                elasticity = estimate.elasticity
                interval = estimate.confidence_interval
            except (InsufficientDataError, ValueError) as exc:
                logger.info("scenario.elasticity_unavailable", error=str(exc))

        return Inputs(
            baseline_units=units / days,
            unit_price=price,
            unit_cost=cost,
            elasticity=elasticity,
            elasticity_interval=interval,
            cross_effects=cross_effects,
            promo_uplift=self._default_uplift if "promotion" in wants else None,
        )

    def _to_result(
        self,
        projection: Projection,
        *,
        levers: list[ScenarioLever],
        horizon_days: int,
        scenario_name: str,
        description: str,
    ) -> ScenarioResult:
        outcome = ScenarioOutcome(
            baseline_units=projection.baseline_units,
            baseline_revenue=projection.baseline_revenue,
            baseline_profit=projection.baseline_profit,
            scenario_units=projection.scenario_units,
            scenario_revenue=projection.scenario_revenue,
            scenario_profit=projection.scenario_profit,
            units_impact=projection.units_impact,
            revenue_impact=projection.revenue_impact,
            profit_impact=projection.profit_impact,
            margin_impact_pct=projection.margin_impact_pct,
            revenue_impact_range=projection.revenue_range,
            profit_impact_range=projection.profit_range,
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            description=description or ", ".join(lever.lever for lever in levers),
            levers=levers,
            horizon_days=horizon_days,
            outcome=outcome,
            risk=projection.risk,
            confidence=projection.confidence,
            assumptions=[
                *projection.assumptions,
                f"Daily figures are scaled linearly to {horizon_days} days, which "
                f"assumes the intervention runs for the whole horizon and demand "
                f"has no trend inside it.",
                "Confidence is the WEAKEST component, not an average. A "
                "projection is only as trustworthy as its shakiest input.",
            ],
            warnings=projection.warnings,
            contributing_models=[
                f"{name}:{MODEL_VERSION}" for name in projection.contributing_models
            ],
        )


__all__ = ["MODEL_VERSION", "FittedScenarioEngine"]
