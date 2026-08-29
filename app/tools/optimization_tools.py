"""Optimisation and scenario tools.

Three tools in one module because they share a shape: each consumes estimates
the other steps produced and returns a *decision* rather than a measurement.

**What separates these from the estimation tools.** A forecast or an elasticity
is a statement about the world. An allocation or a recommended price is a
statement about what to do, and it inherits every assumption underneath it. So
each of these carries the assumptions of the models it composed, and the
scenario tool reports the **weakest** component confidence rather than an
average - a projection is only as trustworthy as its shakiest input.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import get_logger
from app.schemas.tool_contract import ModelProvenance, ToolErrorCode
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput
from ml.base import InsufficientDataError
from ml.scenario.interface import ScenarioLever
from ml.trade_promo_optimization.interface import AllocationConstraint

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Trade promotion budget allocation
# --------------------------------------------------------------------------


class BudgetConstraintInput(BaseModel):
    """A spend bound on one dimension value."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="One of: product, region, retailer, category.")
    value: str
    min_spend: float | None = None
    max_spend: float | None = None


class AllocateBudgetInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_budget: float = Field(gt=0, description="Budget to allocate.")
    product_ids: list[str] | None = None
    regions: list[str] | None = None
    retailers: list[str] | None = None
    constraints: list[BudgetConstraintInput] = Field(default_factory=list)
    max_lines: int = Field(default=25, gt=0, le=500)


class AllocateBudgetOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_budget: float
    allocated_budget: float
    expected_incremental_profit: float
    expected_incremental_units: float
    overall_roi: float | None = None
    #: optimal | feasible | infeasible | unbounded
    optimization_status: str
    #: What limits further gain. An infeasible result names the conflict here.
    binding_constraints: list[str] = Field(default_factory=list)
    allocations: list[dict[str, object]] = Field(default_factory=list)


class AllocateBudgetTool(AnalyticalTool[AllocateBudgetInput, AllocateBudgetOutput]):
    """Allocate a promotional budget to maximise incremental profit."""

    name = "allocate_promotion_budget"
    description = (
        "Allocate a fixed trade promotion budget across products, regions and "
        "retailers to maximise INCREMENTAL profit, subject to constraints like "
        "minimum regional spend or per-retailer caps. Uses measured causal "
        "uplift per promotion, and models diminishing returns so the budget "
        "spreads rather than pouring into one cell. Returns the allocation, the "
        "marginal return on the next rupee, and which constraints bind. An "
        "infeasible result is a finding - 'your minimum spends already exceed "
        "the budget' - not an error. Use for 'where should I spend the next "
        "10M', 'how should I split the budget across regions'."
    )
    input_schema = AllocateBudgetInput
    output_schema = AllocateBudgetOutput
    permission = "optimise"
    timeout_seconds = 120.0

    def __init__(self, service: object) -> None:
        self._service = service

    def _execute(self, payload: AllocateBudgetInput) -> ToolOutput[AllocateBudgetOutput]:
        try:
            result = self._service.allocate(  # type: ignore[attr-defined]
                total_budget=payload.total_budget,
                product_ids=payload.product_ids,
                regions=payload.regions,
                retailers=payload.retailers,
                constraints=[
                    AllocationConstraint(
                        dimension=c.dimension,
                        value=c.value,
                        min_spend=c.min_spend,
                        max_spend=c.max_spend,
                    )
                    for c in payload.constraints
                ],
            )
        except InsufficientDataError as exc:
            raise ToolExecutionError(
                str(exc), code=ToolErrorCode.INSUFFICIENT_DATA, recoverable=True
            ) from exc

        warnings = list(result.warnings)
        if result.optimization_status != "optimal":
            # Promoted to the front: a supervisor reading a truncated list must
            # not miss that no valid allocation was found.
            warnings.insert(
                0,
                f"OPTIMISATION STATUS {result.optimization_status.upper()}: the "
                f"allocation below is not a solved optimum.",
            )

        return ToolOutput(
            payload=AllocateBudgetOutput(
                total_budget=result.total_budget,
                allocated_budget=round(result.allocated_budget, 2),
                expected_incremental_profit=round(result.expected_incremental_profit, 2),
                expected_incremental_units=round(result.expected_incremental_units, 1),
                overall_roi=round(result.overall_roi, 3) if result.overall_roi else None,
                optimization_status=result.optimization_status,
                binding_constraints=result.binding_constraints,
                allocations=[
                    {
                        "product_id": line.product_id,
                        "region": line.region,
                        "retailer": line.retailer,
                        "spend": round(line.allocated_spend, 2),
                        "incremental_profit": round(line.expected_incremental_profit, 2),
                        "roi": round(line.expected_roi, 3),
                        "marginal_roi": round(line.marginal_roi, 3)
                        if line.marginal_roi
                        else None,
                    }
                    for line in result.allocations[: payload.max_lines]
                ],
            ),
            provenance=ModelProvenance(
                model_name="trade_promo_optimization", model_version="v1.0"
            ),
            confidence=None,
            assumptions=list(result.assumptions),
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# Price optimisation
# --------------------------------------------------------------------------


class OptimizePriceInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    product_id: str
    region: str | None = None
    objective: str = Field(default="profit", description="profit | revenue | units")
    min_margin_pct: float | None = Field(
        default=None, description="Margin floor as a fraction, e.g. 0.30."
    )
    max_price_change_pct: float | None = Field(
        default=None, description="Cap on the move, e.g. 0.10 for +/-10%."
    )


class OptimizePriceOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    product_id: str
    current_price: float
    recommended_price: float
    #: The output that matters. A single price from an uncertain elasticity is
    #: false precision.
    recommended_range: dict[str, float] | None = None
    change_pct: float
    expected_profit_impact: float | None = None
    expected_revenue_impact: float | None = None
    elasticity_used: float | None = None
    risk: str
    binding_constraints: list[str] = Field(default_factory=list)
    scenarios: list[dict[str, object]] = Field(default_factory=list)


class OptimizePriceTool(AnalyticalTool[OptimizePriceInput, OptimizePriceOutput]):
    """Recommend a price range that maximises profit under constraints."""

    name = "optimize_price"
    description = (
        "Recommend a price for a product, subject to margin floors and "
        "price-change caps. Returns a RANGE rather than a single price, because "
        "an optimum computed from an elasticity with a wide confidence interval "
        "is false precision - anywhere in the range is roughly equivalent. "
        "Scores on portfolio profit net of cannibalisation, so it will not "
        "recommend a rise that merely moves volume to a substitute you also "
        "sell. Use for 'what price should we set', 'can we afford a price "
        "rise', 'what would a 5% cut do'."
    )
    input_schema = OptimizePriceInput
    output_schema = OptimizePriceOutput
    permission = "optimise"
    timeout_seconds = 180.0

    def __init__(self, service: object) -> None:
        self._service = service

    def _execute(self, payload: OptimizePriceInput) -> ToolOutput[OptimizePriceOutput]:
        try:
            result = self._service.optimize_price(  # type: ignore[attr-defined]
                product_id=payload.product_id,
                region=payload.region,
                objective=payload.objective,
                min_margin_pct=payload.min_margin_pct,
                max_price_change_pct=payload.max_price_change_pct,
            )
        except InsufficientDataError as exc:
            raise ToolExecutionError(
                str(exc), code=ToolErrorCode.INSUFFICIENT_DATA, recoverable=True
            ) from exc

        band = (
            {"lower": result.recommended_range[0], "upper": result.recommended_range[1]}
            if result.recommended_range
            else None
        )

        return ToolOutput(
            payload=OptimizePriceOutput(
                product_id=result.product_id,
                current_price=round(result.current_price, 2),
                recommended_price=round(result.recommended_price, 2),
                recommended_range=band,
                change_pct=round(result.recommended_change_pct, 4),
                expected_profit_impact=round(result.expected_profit_impact, 2)
                if result.expected_profit_impact is not None
                else None,
                expected_revenue_impact=round(result.expected_revenue_impact, 2)
                if result.expected_revenue_impact is not None
                else None,
                elasticity_used=round(result.elasticity_used, 3)
                if result.elasticity_used
                else None,
                risk=str(result.risk),
                binding_constraints=result.binding_constraints,
                scenarios=[
                    {
                        "label": s.label,
                        "price": s.price,
                        "units": round(s.expected_units, 1),
                        "profit": round(s.expected_profit, 2),
                        "feasible": not s.constraint_violations,
                    }
                    for s in result.scenarios
                ],
            ),
            provenance=ModelProvenance(model_name="price_optimization", model_version="v1.0"),
            confidence=None,
            assumptions=list(result.assumptions),
            warnings=list(result.warnings),
        )


# --------------------------------------------------------------------------
# Scenario simulation
# --------------------------------------------------------------------------


class ScenarioLeverInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    lever: str = Field(description="price | promotion | competitor_price")
    change_pct: float | None = Field(
        default=None, description="Fractional change, e.g. -0.05 for a 5% cut."
    )


class SimulateScenarioInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    product_id: str
    levers: list[ScenarioLeverInput] = Field(min_length=1)
    region: str | None = None
    horizon_days: int = Field(default=30, gt=0, le=365)
    scenario_name: str = "scenario"


class SimulateScenarioOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scenario_name: str
    horizon_days: int
    baseline_units: float
    scenario_units: float
    units_impact: float
    revenue_impact: float
    profit_impact: float
    profit_impact_range: dict[str, float] | None = None
    #: The weakest component, not an average.
    confidence: float
    risk: str
    contributing_models: list[str] = Field(default_factory=list)


class SimulateScenarioTool(AnalyticalTool[SimulateScenarioInput, SimulateScenarioOutput]):
    """Project the impact of price, promotion or competitor changes."""

    name = "simulate_scenario"
    description = (
        "Project what happens if you change price, run a promotion, or a "
        "competitor moves - alone or in combination. Composes the elasticity, "
        "cross-price and uplift models rather than fitting anything, so it "
        "inherits all their uncertainty: the output is a RANGE and a confidence "
        "that reflects the WEAKEST component, not an average. Use for 'what if "
        "we cut price 5%', 'what happens if we promote and the competitor "
        "responds'. A lever that is not modelled is reported, never silently "
        "ignored."
    )
    input_schema = SimulateScenarioInput
    output_schema = SimulateScenarioOutput
    permission = "run_model"
    timeout_seconds = 180.0

    def __init__(self, service: object) -> None:
        self._service = service

    def _execute(self, payload: SimulateScenarioInput) -> ToolOutput[SimulateScenarioOutput]:
        try:
            result = self._service.simulate(  # type: ignore[attr-defined]
                levers=[
                    ScenarioLever(lever=lever.lever, change_pct=lever.change_pct)
                    for lever in payload.levers
                ],
                product_ids=[payload.product_id],
                region=payload.region,
                horizon_days=payload.horizon_days,
                scenario_name=payload.scenario_name,
            )
        except InsufficientDataError as exc:
            raise ToolExecutionError(
                str(exc), code=ToolErrorCode.INSUFFICIENT_DATA, recoverable=True
            ) from exc

        outcome = result.outcome
        band = (
            {
                "lower": round(outcome.profit_impact_range[0], 2),
                "upper": round(outcome.profit_impact_range[1], 2),
            }
            if outcome.profit_impact_range
            else None
        )

        return ToolOutput(
            payload=SimulateScenarioOutput(
                scenario_name=result.scenario_name,
                horizon_days=result.horizon_days,
                baseline_units=round(outcome.baseline_units, 1),
                scenario_units=round(outcome.scenario_units, 1),
                units_impact=round(outcome.units_impact, 1),
                revenue_impact=round(outcome.revenue_impact, 2),
                profit_impact=round(outcome.profit_impact, 2),
                profit_impact_range=band,
                confidence=result.confidence,
                risk=str(result.risk),
                contributing_models=result.contributing_models,
            ),
            provenance=ModelProvenance(model_name="scenario_simulation", model_version="v1.0"),
            # The measured expression of uncertainty is the profit range plus
            # the weakest-link confidence, both in the payload.
            confidence=result.confidence,
            assumptions=list(result.assumptions),
            warnings=list(result.warnings),
        )


__all__ = [
    "AllocateBudgetInput",
    "AllocateBudgetOutput",
    "AllocateBudgetTool",
    "OptimizePriceInput",
    "OptimizePriceOutput",
    "OptimizePriceTool",
    "SimulateScenarioInput",
    "SimulateScenarioOutput",
    "SimulateScenarioTool",
]
