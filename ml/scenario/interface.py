"""Scenario simulation engine interface (Stage 1 Step 11).

Answers "what if" questions by composing the other models rather than fitting
anything of its own: elasticity supplies the price response, cross-price
elasticity the portfolio spillover, promotion uplift the promotional response,
and the forecast the volume base.

Because it composes, it also compounds uncertainty. A scenario chaining a
forecast, an elasticity and an uplift estimate inherits all three error terms,
and the honest output is a range with an explicit assumption list - not a
crisp number that looks more certain than its inputs. ``confidence`` here should
be the weakest link, never the average.
"""

from __future__ import annotations

from abc import abstractmethod

from pydantic import BaseModel, Field

from app.schemas.domain import RiskLevel
from ml.base import AnalyticalModel


class ScenarioLever(BaseModel):
    """One intervention being simulated.

    Multiple levers may be applied together, which is what makes questions like
    "cut price 3% *and* move budget from North to South" answerable.
    """

    #: "price" | "promotion_spend" | "competitor_price" | "inventory"
    #: | "budget_reallocation"
    lever: str
    product_id: str | None = None
    region: str | None = None
    from_region: str | None = None
    to_region: str | None = None
    change_pct: float | None = None
    change_absolute: float | None = None


class ScenarioOutcome(BaseModel):
    """Projected impact of the levers, stated against an explicit baseline."""

    baseline_units: float
    baseline_revenue: float
    baseline_profit: float

    scenario_units: float
    scenario_revenue: float
    scenario_profit: float

    units_impact: float
    revenue_impact: float
    profit_impact: float
    margin_impact_pct: float

    revenue_impact_range: tuple[float, float] | None = None
    profit_impact_range: tuple[float, float] | None = None


class ScenarioResult(BaseModel):
    """A simulated scenario with full provenance of its component models."""

    scenario_name: str
    description: str
    levers: list[ScenarioLever] = Field(default_factory=list)
    horizon_days: int = Field(gt=0)

    outcome: ScenarioOutcome

    risk: RiskLevel = RiskLevel.MEDIUM
    #: Weakest link across the composed models, not their average.
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: name:version of every model contributing to this projection.
    contributing_models: list[str] = Field(default_factory=list)


class ScenarioEngine(AnalyticalModel[ScenarioResult]):
    """Composes other models to project the impact of interventions."""

    name = "scenario_simulation"

    @abstractmethod
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
        """Simulate the combined effect of the levers over the horizon."""

    @abstractmethod
    def compare(
        self,
        *,
        scenarios: list[list[ScenarioLever]],
        horizon_days: int = 30,
        product_ids: list[str] | None = None,
        region: str | None = None,
    ) -> list[ScenarioResult]:
        """Simulate several scenarios against a common baseline.

        Separate from ``predict`` because a like-for-like comparison requires
        every scenario to share one baseline; computing them independently lets
        baseline drift masquerade as a difference between the options.
        """
