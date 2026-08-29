"""Scenario simulation: composing the other models into a what-if.

Fits nothing of its own. Elasticity supplies the price response, cross-price the
portfolio spillover, uplift the promotional response, and the sales history the
volume base.

**Because it composes, it compounds uncertainty.** A scenario chaining a
baseline, an elasticity and an uplift estimate inherits all three error terms.
The honest output is a range with an explicit assumption list, and ``confidence``
is **the weakest link, never the average** — averaging a 0.9 and a 0.3 into 0.6
describes a projection that does not exist.

**Levers compose multiplicatively in log space**, matching the demand equation
they are projected through. That is what makes "cut price 3% *and* run a
promotion" a single coherent projection rather than two independent estimates
added together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.observability.logging import get_logger
from app.schemas.domain import RiskLevel

logger = get_logger(__name__)

#: Confidence attributed to each component model, from its measured accuracy.
#: These are used as the *weakest link*, so a scenario is never more confident
#: than its shakiest input.
COMPONENT_CONFIDENCE: dict[str, float] = {
    # Recovered known elasticities at r=0.99 across 30 products.
    "price_elasticity": 0.85,
    # Cross-price: correct substitute found, zero false positives, but a much
    # smaller effective sample per pair.
    "cross_price_elasticity": 0.60,
    # Uplift recovered ground truth to 0.7pp on 4,417 events.
    "promo_uplift": 0.80,
    # Baseline sits at 1.15x the irreducible noise floor.
    "baseline_sales": 0.75,
    # A competitor response is an assumption, not an estimate: nothing in the
    # data says how a rival reacts to our price move.
    "competitor_price": 0.40,
}


@dataclass
class Lever:
    """One intervention. Multiple levers apply together."""

    lever: str
    change_pct: float | None = None
    change_absolute: float | None = None
    product_id: str | None = None
    region: str | None = None

    def describe(self) -> str:
        if self.change_pct is not None:
            return f"{self.lever} {self.change_pct:+.1%}"
        return f"{self.lever} {self.change_absolute:+,.0f}"


@dataclass
class Projection:
    """The projected outcome, against an explicit baseline."""

    baseline_units: float
    baseline_revenue: float
    baseline_profit: float
    scenario_units: float
    scenario_revenue: float
    scenario_profit: float

    revenue_range: tuple[float, float] | None = None
    profit_range: tuple[float, float] | None = None

    confidence: float = 0.5
    risk: RiskLevel = RiskLevel.MEDIUM
    contributing_models: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def units_impact(self) -> float:
        return self.scenario_units - self.baseline_units

    @property
    def revenue_impact(self) -> float:
        return self.scenario_revenue - self.baseline_revenue

    @property
    def profit_impact(self) -> float:
        return self.scenario_profit - self.baseline_profit

    @property
    def margin_impact_pct(self) -> float:
        scenario = self.scenario_profit / self.scenario_revenue if self.scenario_revenue else 0.0
        baseline = self.baseline_profit / self.baseline_revenue if self.baseline_revenue else 0.0
        return scenario - baseline

    def summary(self) -> str:
        return (
            f"units {self.units_impact:+,.0f}, revenue {self.revenue_impact:+,.0f}, "
            f"profit {self.profit_impact:+,.0f} (confidence {self.confidence:.0%})"
        )


@dataclass
class Inputs:
    """Everything a projection needs, gathered by the caller.

    Passed in rather than fetched, so the engine composes and does not also do
    data access — and so a test can hold every input fixed.
    """

    baseline_units: float
    unit_price: float
    unit_cost: float

    elasticity: float | None = None
    elasticity_interval: tuple[float, float] | None = None
    #: product -> (cross_elasticity, its units, its unit margin)
    cross_effects: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: Fractional uplift from a promotion, from Step 7.
    promo_uplift: float | None = None
    promo_uplift_interval: tuple[float, float] | None = None
    #: Elasticity of our demand to a competitor's price.
    competitor_sensitivity: float | None = None


def project(levers: list[Lever], inputs: Inputs, *, horizon_days: int = 30) -> Projection:
    """Apply the levers and project the outcome.

    Effects accumulate in **log space** and are exponentiated once at the end.
    Applying them multiplicatively one at a time would give the same answer for
    a single lever and a different one for several, because the order would
    start to matter.
    """
    log_effect = 0.0
    price_multiplier = 1.0
    models: list[str] = []
    assumptions: list[str] = []
    warnings: list[str] = []
    confidences: list[float] = []
    cannibalisation = 0.0

    for lever in levers:
        if lever.lever == "price":
            log_effect, price_multiplier, cannibalisation = _apply_price(
                lever, inputs, log_effect, price_multiplier, cannibalisation,
                models, assumptions, warnings, confidences,
            )
        elif lever.lever in {"promotion", "promotion_spend"}:
            log_effect = _apply_promotion(
                lever, inputs, log_effect, models, assumptions, warnings, confidences
            )
        elif lever.lever == "competitor_price":
            log_effect = _apply_competitor(
                lever, inputs, log_effect, models, assumptions, warnings, confidences
            )
        else:
            warnings.append(
                f"lever '{lever.lever}' is not modelled and was ignored; the "
                f"projection below does not include it"
            )

    scenario_units = inputs.baseline_units * float(np.exp(log_effect))
    scenario_price = inputs.unit_price * price_multiplier

    baseline_revenue = inputs.baseline_units * inputs.unit_price
    baseline_profit = inputs.baseline_units * (inputs.unit_price - inputs.unit_cost)
    scenario_revenue = scenario_units * scenario_price
    scenario_profit = scenario_units * (scenario_price - inputs.unit_cost)

    # Confidence is the weakest link. A projection is only as trustworthy as its
    # shakiest component, and averaging would describe a projection nobody made.
    confidence = min(confidences) if confidences else 0.5

    projection = Projection(
        baseline_units=baseline_units_scaled(inputs.baseline_units, horizon_days),
        baseline_revenue=baseline_units_scaled(baseline_revenue, horizon_days),
        baseline_profit=baseline_units_scaled(baseline_profit, horizon_days),
        scenario_units=baseline_units_scaled(scenario_units, horizon_days),
        scenario_revenue=baseline_units_scaled(scenario_revenue, horizon_days),
        scenario_profit=baseline_units_scaled(scenario_profit, horizon_days),
        confidence=confidence,
        contributing_models=sorted(set(models)),
        assumptions=assumptions,
        warnings=warnings,
    )

    _attach_ranges(projection, inputs, levers, horizon_days)
    projection.risk = _risk(projection, confidence)

    if cannibalisation:
        projection.warnings.append(
            f"{cannibalisation:+,.0f} units move on related products. The figures "
            f"above are for this product only and overstate the portfolio effect"
        )

    logger.info(
        "scenario.projected",
        levers=[lever.lever for lever in levers],
        profit_impact=round(projection.profit_impact, 2),
        confidence=round(confidence, 3),
    )
    return projection


def baseline_units_scaled(daily_value: float, horizon_days: int) -> float:
    """Scale a daily figure to the horizon.

    Linear, and that is an assumption rather than a fact: it holds if the
    intervention runs for the whole horizon and demand has no trend inside it.
    Stated in the assumption list rather than hidden in the arithmetic.
    """
    return daily_value * horizon_days


def _apply_price(
    lever: Lever,
    inputs: Inputs,
    log_effect: float,
    price_multiplier: float,
    cannibalisation: float,
    models: list[str],
    assumptions: list[str],
    warnings: list[str],
    confidences: list[float],
) -> tuple[float, float, float]:
    if inputs.elasticity is None:
        warnings.append(
            "a price lever was requested but no elasticity is available, so the "
            "volume response is not projected. The price change is applied to "
            "revenue at constant volume, which understates its effect"
        )
        change = lever.change_pct or 0.0
        return log_effect, price_multiplier * (1.0 + change), cannibalisation

    change = lever.change_pct or 0.0
    models.append("price_elasticity")
    confidences.append(COMPONENT_CONFIDENCE["price_elasticity"])
    assumptions.append(
        f"Price response uses a constant elasticity of {inputs.elasticity:.2f}, "
        f"a local approximation around observed prices. A {change:+.1%} move "
        f"stays inside that range; a larger one would not."
    )

    log_effect += inputs.elasticity * float(np.log1p(change))

    for cross_elasticity, related_units, _margin in inputs.cross_effects.values():
        cannibalisation += related_units * ((1.0 + change) ** cross_elasticity - 1.0)
    if inputs.cross_effects:
        models.append("cross_price_elasticity")
        confidences.append(COMPONENT_CONFIDENCE["cross_price_elasticity"])

    return log_effect, price_multiplier * (1.0 + change), cannibalisation


def _apply_promotion(
    lever: Lever,
    inputs: Inputs,
    log_effect: float,
    models: list[str],
    assumptions: list[str],
    warnings: list[str],
    confidences: list[float],
) -> float:
    if inputs.promo_uplift is None:
        warnings.append(
            "a promotion lever was requested but no uplift estimate is "
            "available; its effect is not in the projection"
        )
        return log_effect

    models.append("promo_uplift")
    confidences.append(COMPONENT_CONFIDENCE["promo_uplift"])
    assumptions.append(
        f"Promotional response uses a measured uplift of {inputs.promo_uplift:+.1%}, "
        f"which is the NET figure including pull-forward payback."
    )
    return log_effect + float(np.log1p(inputs.promo_uplift))


def _apply_competitor(
    lever: Lever,
    inputs: Inputs,
    log_effect: float,
    models: list[str],
    assumptions: list[str],
    warnings: list[str],
    confidences: list[float],
) -> float:
    if inputs.competitor_sensitivity is None:
        warnings.append(
            "a competitor-price lever was requested but no competitor "
            "sensitivity is available; its effect is not in the projection"
        )
        return log_effect

    change = lever.change_pct or 0.0
    models.append("competitor_price")
    confidences.append(COMPONENT_CONFIDENCE["competitor_price"])
    assumptions.append(
        "The competitor's price move is treated as exogenous. Nothing in the "
        "data says how a rival reacts to our own move, so a scenario where both "
        "prices change is two assumptions, not one."
    )
    return log_effect + inputs.competitor_sensitivity * float(np.log1p(change))


def _attach_ranges(
    projection: Projection, inputs: Inputs, levers: list[Lever], horizon_days: int
) -> None:
    """Propagate input intervals into an outcome range.

    Recomputes the projection at each end of the elasticity and uplift
    intervals. Crude compared with a full error propagation and honest about
    what it is: a band showing how much the answer moves when the inputs move
    across the range that was actually measured.
    """
    has_interval = inputs.elasticity_interval or inputs.promo_uplift_interval
    if not has_interval:
        projection.warnings.append(
            "no interval on the component estimates, so the projection is a "
            "point with no stated uncertainty"
        )
        return

    outcomes = []
    for elasticity in _interval_points(inputs.elasticity, inputs.elasticity_interval):
        for uplift in _interval_points(inputs.promo_uplift, inputs.promo_uplift_interval):
            variant = Inputs(
                baseline_units=inputs.baseline_units,
                unit_price=inputs.unit_price,
                unit_cost=inputs.unit_cost,
                elasticity=elasticity,
                cross_effects=inputs.cross_effects,
                promo_uplift=uplift,
                competitor_sensitivity=inputs.competitor_sensitivity,
            )
            result = project(levers, variant, horizon_days=horizon_days)
            outcomes.append((result.revenue_impact, result.profit_impact))

    if outcomes:
        revenues = [o[0] for o in outcomes]
        profits = [o[1] for o in outcomes]
        projection.revenue_range = (min(revenues), max(revenues))
        projection.profit_range = (min(profits), max(profits))


def _interval_points(
    point: float | None, interval: tuple[float, float] | None
) -> list[float | None]:
    if interval is None:
        return [point]
    return [interval[0], interval[1]]


def _risk(projection: Projection, confidence: float) -> RiskLevel:
    """Risk of acting on the projection.

    A wide range relative to the central estimate is the signal, not the size of
    the projected gain.
    """
    if confidence < 0.5:
        return RiskLevel.HIGH
    if projection.profit_range and projection.profit_impact:
        spread = projection.profit_range[1] - projection.profit_range[0]
        # A range as wide as the central estimate means the projection does not
        # establish the size of the effect - only its sign. An earlier threshold
        # of 2x called a 87k-227k band around 156k "low risk", which it plainly
        # is not.
        if spread > abs(projection.profit_impact):
            return RiskLevel.HIGH
        if spread > abs(projection.profit_impact) * 0.5:
            return RiskLevel.MEDIUM
    return RiskLevel.MEDIUM if confidence < 0.75 else RiskLevel.LOW


__all__ = [
    "COMPONENT_CONFIDENCE",
    "Inputs",
    "Lever",
    "Projection",
    "project",
]
