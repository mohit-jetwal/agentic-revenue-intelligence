"""Optimisation and scenario service.

One service for three capabilities, because they share their dependencies: all
three consume the elasticity and uplift estimates the earlier steps produced, and
constructing three services would build three copies of the same models.

Nothing is loaded eagerly. The trade-promo optimiser needs Step 7's event table
and the price optimiser needs an elasticity estimate; both surface as readable
errors at the point of use rather than as a container failure at boot.
"""

from __future__ import annotations

from datetime import date

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.cross_price_elasticity.model import FittedCrossElasticityModel
from ml.price_elasticity.model import FittedElasticityModel
from ml.price_optimization.interface import PriceOptimizationResult
from ml.price_optimization.model import FittedPriceOptimizer
from ml.scenario.interface import ScenarioLever, ScenarioResult
from ml.scenario.model import FittedScenarioEngine
from ml.trade_promo_optimization.interface import AllocationConstraint, OptimizationResult
from ml.trade_promo_optimization.model import FittedTradePromoOptimizer

logger = get_logger(__name__)


class OptimizationService:
    """Budget allocation, price optimisation and scenario simulation."""

    def __init__(
        self,
        repository: DataRepository,
        *,
        settings: Settings | None = None,
        default_promo_uplift: float | None = 0.30,
    ) -> None:
        self._repository = repository
        self._settings = settings or get_settings()
        self._default_uplift = default_promo_uplift

        self._elasticity: FittedElasticityModel | None = None
        self._cross: FittedCrossElasticityModel | None = None
        self._allocator: FittedTradePromoOptimizer | None = None
        self._pricer: FittedPriceOptimizer | None = None
        self._scenario: FittedScenarioEngine | None = None

    # -- lazily constructed models ------------------------------------------

    @property
    def elasticity_model(self) -> FittedElasticityModel:
        if self._elasticity is None:
            self._elasticity = FittedElasticityModel(self._repository)
        return self._elasticity

    @property
    def cross_model(self) -> FittedCrossElasticityModel:
        if self._cross is None:
            self._cross = FittedCrossElasticityModel(self._repository)
        return self._cross

    @property
    def allocator(self) -> FittedTradePromoOptimizer:
        if self._allocator is None:
            self._allocator = FittedTradePromoOptimizer(self._repository)
        return self._allocator

    @property
    def pricer(self) -> FittedPriceOptimizer:
        if self._pricer is None:
            self._pricer = FittedPriceOptimizer(
                self._repository,
                elasticity_model=self.elasticity_model,
                cross_model=self.cross_model,
            )
        return self._pricer

    @property
    def scenario_engine(self) -> FittedScenarioEngine:
        if self._scenario is None:
            self._scenario = FittedScenarioEngine(
                self._repository,
                elasticity_model=self.elasticity_model,
                cross_model=self.cross_model,
                default_uplift=self._default_uplift,
            )
        return self._scenario

    def health_check(self) -> tuple[bool, str]:
        """Reports whether the *inputs* exist, since nothing is trained here."""
        try:
            events = self.allocator.events
        except Exception:  # noqa: BLE001 - health must never raise
            return True, "optimisation (no uplift event table; allocation unavailable)"
        return True, f"optimisation ({len(events):,} promotions available to allocate)"

    # -- capabilities -------------------------------------------------------

    def allocate(
        self,
        *,
        total_budget: float,
        product_ids: list[str] | None = None,
        regions: list[str] | None = None,
        retailers: list[str] | None = None,
        constraints: list[AllocationConstraint] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> OptimizationResult:
        return self.allocator.predict(
            total_budget=total_budget,
            product_ids=product_ids,
            regions=regions,
            retailers=retailers,
            constraints=constraints,
            start_date=start_date,
            end_date=end_date,
        )

    def optimize_price(
        self,
        *,
        product_id: str,
        region: str | None = None,
        objective: str = "profit",
        min_margin_pct: float | None = None,
        max_price_change_pct: float | None = None,
    ) -> PriceOptimizationResult:
        return self.pricer.predict(
            product_id=product_id,
            region=region,
            objective=objective,
            min_margin_pct=min_margin_pct,
            max_price_change_pct=max_price_change_pct,
        )

    def simulate(
        self,
        *,
        levers: list[ScenarioLever],
        product_ids: list[str],
        region: str | None = None,
        horizon_days: int = 30,
        scenario_name: str = "scenario",
    ) -> ScenarioResult:
        return self.scenario_engine.predict(
            levers=levers,
            product_ids=product_ids,
            region=region,
            horizon_days=horizon_days,
            scenario_name=scenario_name,
        )

    def compare_scenarios(
        self,
        *,
        scenarios: list[list[ScenarioLever]],
        product_ids: list[str],
        region: str | None = None,
        horizon_days: int = 30,
    ) -> list[ScenarioResult]:
        """Several options against **one** baseline. See the engine's docstring."""
        return self.scenario_engine.compare(
            scenarios=scenarios,
            product_ids=product_ids,
            region=region,
            horizon_days=horizon_days,
        )


__all__ = ["OptimizationService"]
