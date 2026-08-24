"""Price optimisation interface (Stage 1 Step 10).

Recommends a price (or a defensible price range) that maximises profit or
revenue subject to margin floors, competitive position, inventory and
price-change limits.

Deliberately returns a *set of evaluated candidates* alongside the recommended
price, not a single number. Two reasons. First, an optimum computed from an
elasticity with a wide confidence interval is a false precision - the honest
output is a range over which the objective is near-flat. Second, a category
manager will not act on "set it to 104.37"; they will act on "anywhere between
103 and 106 is roughly equivalent, and here is what each does to volume".

Cross-price effects are required input, not optional. Optimising a product in
isolation reliably recommends a rise that simply moves volume to its own
category neighbour, booking a phantom gain.
"""

from __future__ import annotations

from abc import abstractmethod

from pydantic import BaseModel, Field

from app.schemas.domain import RiskLevel
from ml.base import AnalyticalModel


class PriceScenario(BaseModel):
    """Projected outcome at one candidate price."""

    label: str
    price: float = Field(gt=0)
    price_change_pct: float

    expected_units: float = Field(ge=0)
    expected_revenue: float
    expected_profit: float
    expected_margin_pct: float

    #: Volume lost on related own-portfolio products at this price.
    cannibalisation_units: float | None = None
    #: Net of cannibalisation - the number that actually matters.
    net_portfolio_profit: float | None = None

    risk: RiskLevel = RiskLevel.MEDIUM
    constraint_violations: list[str] = Field(default_factory=list)


class PriceOptimizationResult(BaseModel):
    """Recommended price with the evaluated alternatives behind it."""

    product_id: str
    region: str | None = None
    current_price: float = Field(gt=0)

    recommended_price: float = Field(gt=0)
    #: Range over which the objective is within tolerance of the optimum.
    recommended_range: tuple[float, float] | None = None
    recommended_change_pct: float

    scenarios: list[PriceScenario] = Field(default_factory=list)

    objective: str = "profit"
    elasticity_used: float | None = None
    #: Confidence interval of the elasticity the optimum rests on. A wide
    #: interval should widen the recommended range, not be silently ignored.
    elasticity_confidence_interval: tuple[float, float] | None = None

    expected_revenue_impact: float | None = None
    expected_profit_impact: float | None = None
    risk: RiskLevel = RiskLevel.MEDIUM
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    binding_constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PriceOptimizationModel(AnalyticalModel[PriceOptimizationResult]):
    """Constrained price optimiser."""

    name = "price_optimization"

    @abstractmethod
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
