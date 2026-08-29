"""Price optimisation: evaluate candidate prices, recommend a range.

Given a constant-elasticity demand curve ``q = q0 · (p/p0)^e``, profit is

.. code-block:: text

    profit(p) = q0 · (p/p0)^e · (p - c)

which has a closed-form optimum at ``p* = c · e/(1+e)`` for elastic demand.

**The closed form is deliberately not what ships.** Three reasons.

*Constraints.* Margin floors, price-change caps and competitive bounds are the
normal case, and the unconstrained optimum usually violates one of them.

*False precision.* An optimum computed from an elasticity whose confidence
interval spans -1.8 to -2.6 is not a price, it is a point estimate of a point
estimate. The honest output is the **range over which profit is near-flat**,
which is typically wide - and a category manager will act on "103 to 106 are
equivalent" where they would rightly distrust "set it to 104.37".

*Cannibalisation.* Optimising a product alone reliably recommends a rise that
moves volume to its own category neighbour and books a phantom gain. Cross-price
elasticities are an input, not an enhancement.

So a grid of candidate prices is evaluated, each scored on portfolio profit net
of cannibalisation, and the recommendation is the best candidate plus every
candidate within a tolerance of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.observability.logging import get_logger
from app.schemas.domain import RiskLevel

logger = get_logger(__name__)

#: Candidate price changes evaluated by default, as fractions.
#:
#: The range matters. For constant elasticity the unconstrained optimum is
#: ``p* = c·e/(1+e)`` - at ``e = -2`` and a 60% cost ratio that is +20%, outside
#: a +/-15% grid. A grid that stops short pins every recommendation to its own
#: edge and the answer becomes an artifact of the grid rather than of the demand
#: curve. Widened to +/-30%, and :func:`recommend` reports when the optimum still
#: lands on the boundary.
DEFAULT_GRID: tuple[float, ...] = (
    -0.30, -0.25, -0.20, -0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01,
    0.0,
    0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30,
)

#: A candidate within this fraction of the best profit is treated as equivalent.
#: 2% is chosen against the elasticity's own uncertainty: with a confidence
#: interval routinely +/-0.2 on the coefficient, differences smaller than this
#: are not distinguishable from estimation error.
PROFIT_TOLERANCE = 0.02


@dataclass
class PriceCandidate:
    """A candidate price and its projected outcome."""

    label: str
    price: float
    change_pct: float
    units: float
    revenue: float
    profit: float
    margin_pct: float
    cannibalisation_units: float = 0.0
    net_portfolio_profit: float = 0.0
    violations: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.violations


@dataclass
class PriceRecommendation:
    """The recommended price, its range, and what limited it."""

    product_id: str
    current_price: float
    recommended_price: float
    recommended_range: tuple[float, float] | None
    change_pct: float
    candidates: list[PriceCandidate]
    elasticity: float
    elasticity_interval: tuple[float, float] | None
    revenue_impact: float
    profit_impact: float
    risk: RiskLevel
    binding_constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        band = (
            f" (range {self.recommended_range[0]:.2f}-{self.recommended_range[1]:.2f})"
            if self.recommended_range
            else ""
        )
        return (
            f"{self.current_price:.2f} -> {self.recommended_price:.2f}"
            f"{band}, {self.change_pct:+.1%}, profit {self.profit_impact:+,.0f}"
        )


def evaluate_candidates(
    *,
    current_price: float,
    current_units: float,
    unit_cost: float,
    elasticity: float,
    grid: tuple[float, ...] = DEFAULT_GRID,
    min_price: float | None = None,
    max_price: float | None = None,
    min_margin_pct: float | None = None,
    max_change_pct: float | None = None,
    cross_effects: dict[str, tuple[float, float, float]] | None = None,
) -> list[PriceCandidate]:
    """Project every candidate price through the demand curve.

    ``cross_effects`` maps a related product to
    ``(cross_elasticity, its_units, its_unit_margin)``. A positive cross
    elasticity means substitutes: raising this product's price sends volume
    *to* them, which is a portfolio gain rather than a loss. The sign is
    handled explicitly because getting it backwards inverts the recommendation.
    """
    candidates: list[PriceCandidate] = []

    for change in grid:
        price = current_price * (1.0 + change)
        if price <= 0:
            continue

        # Constant-elasticity response.
        units = current_units * (price / current_price) ** elasticity
        revenue = units * price
        profit = units * (price - unit_cost)
        margin = (price - unit_cost) / price if price > 0 else 0.0

        cannibalisation = 0.0
        portfolio_delta = 0.0
        for cross_elasticity, related_units, related_margin in (cross_effects or {}).values():
            # Their volume responds to OUR price change.
            related_new = related_units * (1.0 + change) ** cross_elasticity
            delta_units = related_new - related_units
            cannibalisation += delta_units
            portfolio_delta += delta_units * related_margin

        violations = _violations(
            price=price,
            change=change,
            margin=margin,
            current_price=current_price,
            min_price=min_price,
            max_price=max_price,
            min_margin_pct=min_margin_pct,
            max_change_pct=max_change_pct,
        )

        candidates.append(
            PriceCandidate(
                label=_label(change),
                price=round(price, 2),
                change_pct=change,
                units=units,
                revenue=revenue,
                profit=profit,
                margin_pct=margin,
                cannibalisation_units=cannibalisation,
                net_portfolio_profit=profit + portfolio_delta,
                violations=violations,
            )
        )

    return candidates


def _violations(
    *,
    price: float,
    change: float,
    margin: float,
    current_price: float,
    min_price: float | None,
    max_price: float | None,
    min_margin_pct: float | None,
    max_change_pct: float | None,
) -> list[str]:
    """Constraints this candidate breaks, named rather than counted."""
    violations: list[str] = []
    if min_price is not None and price < min_price:
        violations.append(f"below the price floor of {min_price:.2f}")
    if max_price is not None and price > max_price:
        violations.append(f"above the price ceiling of {max_price:.2f}")
    if min_margin_pct is not None and margin < min_margin_pct:
        violations.append(
            f"margin {margin:.1%} below the floor of {min_margin_pct:.1%}"
        )
    if max_change_pct is not None and abs(change) > max_change_pct:
        violations.append(
            f"price change {change:+.1%} exceeds the cap of +/-{max_change_pct:.1%}"
        )
    _ = current_price
    return violations


def _label(change: float) -> str:
    if abs(change) < 1e-9:
        return "hold"
    return f"{change:+.0%}" if abs(change) >= 0.01 else f"{change:+.1%}"


def recommend(
    product_id: str,
    candidates: list[PriceCandidate],
    *,
    current_price: float,
    elasticity: float,
    elasticity_interval: tuple[float, float] | None = None,
    objective: str = "profit",
    tolerance: float = PROFIT_TOLERANCE,
) -> PriceRecommendation:
    """Pick the best feasible candidate and the range equivalent to it.

    Scored on **net portfolio profit** by default, not the product's own profit.
    A rise that moves volume to a substitute in the same portfolio has not
    created anything, and an optimiser scoring only the focal product would book
    that transfer as a gain.
    """
    feasible = [c for c in candidates if c.feasible]
    warnings: list[str] = []
    binding: list[str] = []

    if not feasible:
        # Report why rather than raising: "every price violates your margin
        # floor" is a business finding.
        reasons = sorted({v for c in candidates for v in c.violations})
        binding = reasons
        warnings.append(
            "no candidate price satisfies every constraint; holding the current "
            "price. Binding: " + "; ".join(reasons)
        )
        feasible = [c for c in candidates if abs(c.change_pct) < 1e-9] or candidates[:1]

    def score(candidate: PriceCandidate) -> float:
        if objective == "revenue":
            return candidate.revenue
        if objective == "units":
            return candidate.units
        return candidate.net_portfolio_profit

    best = max(feasible, key=score)
    best_score = score(best)

    # Everything within tolerance of the best is equivalent given how uncertain
    # the elasticity behind it is.
    equivalent = [
        c for c in feasible if best_score > 0 and score(c) >= best_score * (1 - tolerance)
    ] or [best]
    price_range = (
        (min(c.price for c in equivalent), max(c.price for c in equivalent))
        if len(equivalent) > 1
        else None
    )

    baseline = next((c for c in candidates if abs(c.change_pct) < 1e-9), None)
    revenue_impact = best.revenue - baseline.revenue if baseline else 0.0
    profit_impact = (
        best.net_portfolio_profit - baseline.net_portfolio_profit if baseline else 0.0
    )

    risk = _risk(best, elasticity_interval, equivalent_count=len(equivalent))

    if elasticity_interval:
        low, high = elasticity_interval
        if high - low > 0.8:
            warnings.append(
                f"the elasticity interval spans {low:.2f} to {high:.2f}. The "
                f"recommended range reflects that width - a single price would "
                f"be false precision"
            )
        if low < -1.0 < high:
            warnings.append(
                "the elasticity interval straddles -1, so whether a price rise "
                "raises or reduces revenue is not established by this estimate"
            )

    # The optimum sitting on the edge of the evaluated range means the true
    # optimum is outside it, and the recommendation is a property of the grid
    # rather than of the demand curve. Unless a constraint put it there, in
    # which case the constraint is the finding.
    grid_edge = max(abs(c.change_pct) for c in candidates)
    if abs(best.change_pct) >= grid_edge - 1e-9 and grid_edge > 0:
        if binding:
            binding.append(f"price change capped at the evaluated range of +/-{grid_edge:.0%}")
        else:
            warnings.append(
                f"the optimum is at the edge of the evaluated range "
                f"(+/-{grid_edge:.0%}), so the unconstrained optimum lies "
                f"beyond it. Treat this as 'move at least this far', not as the "
                f"best price - and widen the grid if a larger move is allowed"
            )

    if best.cannibalisation_units and abs(best.cannibalisation_units) > best.units * 0.1:
        warnings.append(
            f"{best.cannibalisation_units:+,.0f} units move on related products "
            f"at this price; the recommendation is scored on portfolio profit, "
            f"not this product's alone"
        )

    recommendation = PriceRecommendation(
        product_id=product_id,
        current_price=current_price,
        recommended_price=best.price,
        recommended_range=price_range,
        change_pct=best.change_pct,
        candidates=candidates,
        elasticity=elasticity,
        elasticity_interval=elasticity_interval,
        revenue_impact=revenue_impact,
        profit_impact=profit_impact,
        risk=risk,
        binding_constraints=binding,
        warnings=warnings,
    )
    logger.info(
        "price_optimization.recommended",
        product=product_id,
        change=round(best.change_pct, 4),
        profit_impact=round(profit_impact, 2),
    )
    return recommendation


def _risk(
    best: PriceCandidate,
    interval: tuple[float, float] | None,
    *,
    equivalent_count: int,
) -> RiskLevel:
    """Risk of acting on this recommendation.

    Driven by the size of the move and the width of the elasticity behind it,
    not by the size of the projected gain - a large gain computed from a shaky
    coefficient is more dangerous than a small one, not less.
    """
    if abs(best.change_pct) < 0.02:
        return RiskLevel.LOW
    wide = interval is not None and (interval[1] - interval[0]) > 0.8
    if abs(best.change_pct) >= 0.10 or wide or equivalent_count > 6:
        return RiskLevel.HIGH
    return RiskLevel.MEDIUM


__all__ = [
    "DEFAULT_GRID",
    "PROFIT_TOLERANCE",
    "PriceCandidate",
    "PriceRecommendation",
    "evaluate_candidates",
    "recommend",
]
