"""Trade promotion spend optimisation interface (Stage 1 Step 7).

Allocates a fixed promotional budget across products, regions and retailers to
maximise incremental profit subject to business constraints.

Two design points worth stating:

*Diminishing returns.* Promotional response saturates - the tenth point of
discount buys far less than the second. A linear program over a constant
ROI-per-rupee will therefore pour the entire budget into the single
highest-ROI cell, which is both wrong and obviously wrong to any category
manager. Implementations must use a concave response (piecewise-linear
approximation of a saturating curve, or an explicitly non-linear solver).

*Constraint feasibility is a result, not an error.* When constraints conflict,
the tool must report which ones bind rather than raising - "your minimum
regional spends already exceed the budget" is a genuine business finding the
agent should surface, not a crash.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from ml.base import AnalyticalModel


class AllocationConstraint(BaseModel):
    """A bound on spend for one dimension value."""

    #: "product" | "region" | "retailer" | "category"
    dimension: str
    value: str
    min_spend: float | None = Field(default=None, ge=0)
    max_spend: float | None = Field(default=None, ge=0)


class AllocationLine(BaseModel):
    """Recommended spend for one product/region/retailer cell."""

    product_id: str
    region: str | None = None
    retailer: str | None = None

    allocated_spend: float = Field(ge=0)
    expected_incremental_units: float
    expected_incremental_revenue: float
    expected_incremental_profit: float
    expected_roi: float
    #: Marginal profit per additional unit of spend at the optimum. Where the
    #: next rupee should go if the budget grows.
    marginal_roi: float | None = None


class OptimizationResult(BaseModel):
    """Budget allocation with its objective value and binding constraints."""

    total_budget: float = Field(ge=0)
    allocated_budget: float = Field(ge=0)
    objective: str = "incremental_profit"

    allocations: list[AllocationLine] = Field(default_factory=list)

    expected_incremental_units: float = 0.0
    expected_incremental_revenue: float = 0.0
    expected_incremental_profit: float = 0.0
    overall_roi: float | None = None

    #: "optimal" | "feasible" | "infeasible" | "unbounded".
    optimization_status: str = "optimal"
    #: Constraints active at the optimum - these are what limit further gain.
    binding_constraints: list[str] = Field(default_factory=list)
    solver: str | None = None
    solve_time_ms: int | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TradePromoOptimizationModel(AnalyticalModel[OptimizationResult]):
    """Constrained allocator for trade promotion budgets."""

    name = "trade_promo_optimization"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        total_budget: float,
        product_ids: list[str] | None = None,
        regions: list[str] | None = None,
        retailers: list[str] | None = None,
        constraints: list[AllocationConstraint] | None = None,
        objective: str = "incremental_profit",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> OptimizationResult:
        """Allocate the budget to maximise the objective under constraints."""
