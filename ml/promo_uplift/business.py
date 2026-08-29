"""Translating a causal estimate into money (brief section 18).

An uplift percentage does not settle anything. "+18% incremental volume" is
compatible with a promotion that made money and one that lost a fortune, and the
difference is entirely in the margin and the spend. This module does the
arithmetic that turns an effect into a decision:

.. code-block:: text

    incremental_units   = ATT x treated_days
    incremental_revenue = incremental_units x promotional selling price
    incremental_margin  = incremental_units x promotional unit margin
    incremental_profit  = incremental_margin - promotion_spend
    roi                 = incremental_profit / promotion_spend

Four decisions that change the answer, each defensible and each stated.

**Margin is taken at the promotional price, not the regular one.** The
incremental units were sold at a discount, so they earn the discounted margin.
Valuing them at full margin is the most common way a losing promotion is
reported as a winner, and on a deep discount the two differ by more than the
uplift being measured.

**Revenue is incremental, not total.** The promotion did not "generate" the sales
that would have happened anyway. Reporting total promoted revenue against
promotional spend is the other common inflation, and it makes almost every
promotion look profitable.

**Cannibalisation is not deducted, and that is a gap, not a decision.** A
promotion on one SKU takes volume from its substitutes. This model measures the
promoted SKU only, so profit here is an *upper bound* on category profit. Every
result says so. Cross-price effects arrive in Step 9 and can be subtracted then.

**ROI is undefined when spend is zero, and returns None.** Not infinity, not
zero. A display-only mechanic with no recorded spend has no return on
investment - it has an incremental profit, which is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.treatment import AnalysisFrame, RowRole

logger = get_logger(__name__)


@dataclass
class BusinessImpact:
    """What an effect estimate is worth."""

    incremental_units: float
    incremental_revenue: float
    incremental_margin: float
    incremental_profit: float
    promotion_spend: float
    roi: float | None
    #: Per-unit economics, so the arithmetic above can be checked by hand.
    unit_price: float
    unit_margin: float
    margin_rate: float
    treated_days: int
    #: Bounds implied by the effect's confidence interval, when it has one.
    profit_lower: float | None = None
    profit_upper: float | None = None
    roi_lower: float | None = None
    roi_upper: float | None = None
    assumptions: list[str] | None = None
    warnings: list[str] | None = None

    @property
    def profitable(self) -> bool:
        return self.incremental_profit > 0

    @property
    def breaks_even(self) -> bool:
        return self.roi is not None and self.roi >= 1.0

    def summary(self) -> str:
        roi = f"{self.roi:.2f}" if self.roi is not None else "n/a"
        return (
            f"{self.incremental_units:,.0f} incremental units, "
            f"{self.incremental_profit:,.0f} incremental profit on "
            f"{self.promotion_spend:,.0f} spend (ROI {roi})"
        )


def unit_economics(
    analysis: AnalysisFrame, *, config: PromoUpliftConfig | None = None
) -> tuple[float, float, float]:
    """Average promotional price, unit margin and margin rate on treated rows.

    Computed from the sales fact - ``revenue`` and ``cost`` per row - rather than
    from a configured constant. The realised numbers already reflect the
    discount that was actually given, which a static margin assumption cannot.
    Falls back to the configured rate only when cost is genuinely absent.
    """
    settings = config or get_promo_uplift_config()
    treated = analysis.frame[analysis.frame["role"] == RowRole.TREATED]

    if treated.empty:
        return 0.0, 0.0, settings.business.default_gross_margin

    units = float(treated[settings.target].sum())
    if units <= 0:
        return 0.0, 0.0, settings.business.default_gross_margin

    if "revenue" in treated.columns:
        price = float(treated["revenue"].sum()) / units
    elif "selling_price" in treated.columns:
        price = float(treated["selling_price"].mean())
    else:
        return 0.0, 0.0, settings.business.default_gross_margin

    if "cost" in treated.columns and treated["cost"].notna().any():
        unit_cost = float(treated["cost"].sum()) / units
        margin = price - unit_cost
        rate = margin / price if price > 0 else settings.business.default_gross_margin
    else:
        rate = settings.business.default_gross_margin
        margin = price * rate

    return price, margin, rate


def promotion_spend(
    analysis: AnalysisFrame, *, config: PromoUpliftConfig | None = None
) -> tuple[float, list[str]]:
    """Total spend across the qualifying events, and any caveat about it.

    Spend is an *event-level* fact, so it is summed over events rather than over
    rows - summing a per-event figure that has been broadcast across a
    twenty-day window would multiply it by twenty.
    """
    _ = config
    events = analysis.events
    warnings: list[str] = []

    if "promotion_spend" not in events.columns:
        warnings.append(
            "no promotion_spend column, so ROI cannot be computed; incremental "
            "profit is reported before promotional cost"
        )
        return 0.0, warnings

    spend = events["promotion_spend"]
    missing = int(spend.isna().sum())
    if missing:
        warnings.append(
            f"{missing} of {len(events)} events have no recorded spend; ROI is "
            f"computed over the {len(events) - missing} that do, so it "
            f"overstates the return on the full programme"
        )
    total = float(spend.fillna(0.0).sum())
    return total, warnings


def business_impact(
    estimate: EffectEstimate,
    analysis: AnalysisFrame,
    *,
    config: PromoUpliftConfig | None = None,
) -> BusinessImpact:
    """Convert an ATT into units, revenue, profit and ROI."""
    settings = config or get_promo_uplift_config()
    price, margin, rate = unit_economics(analysis, config=settings)
    spend, spend_warnings = promotion_spend(analysis, config=settings)

    treated_days = estimate.n_treated
    incremental_units = estimate.ate * treated_days
    incremental_revenue = incremental_units * price
    incremental_margin = incremental_units * margin
    incremental_profit = incremental_margin - spend

    roi = incremental_profit / spend if spend > 0 else None

    def profit_at(effect: float | None) -> float | None:
        if effect is None:
            return None
        return effect * treated_days * margin - spend

    profit_lower = profit_at(estimate.ci_lower)
    profit_upper = profit_at(estimate.ci_upper)

    warnings = list(spend_warnings)
    if incremental_units < 0:
        warnings.append(
            "incremental units are negative: this promotion is estimated to "
            "have reduced volume. Profit and ROI below are negative accordingly "
            "and have not been floored"
        )
    if roi is not None and roi < settings.business.roi_break_even:
        warnings.append(
            f"ROI of {roi:.2f} is below break-even; this promotion destroyed "
            f"value even though uplift was "
            f"{'positive' if estimate.ate > 0 else 'negative'}"
        )

    impact = BusinessImpact(
        incremental_units=incremental_units,
        incremental_revenue=incremental_revenue,
        incremental_margin=incremental_margin,
        incremental_profit=incremental_profit,
        promotion_spend=spend,
        roi=roi,
        unit_price=price,
        unit_margin=margin,
        margin_rate=rate,
        treated_days=treated_days,
        profit_lower=profit_lower,
        profit_upper=profit_upper,
        roi_lower=profit_lower / spend if profit_lower is not None and spend > 0 else None,
        roi_upper=profit_upper / spend if profit_upper is not None and spend > 0 else None,
        assumptions=[
            f"Incremental units are valued at the realised promotional margin "
            f"of {margin:,.2f} per unit ({rate:.1%}), not the regular margin - "
            f"these units were sold at a discount.",
            "Revenue and profit are INCREMENTAL, not total promoted volume. "
            "Sales that would have happened anyway are excluded.",
            "Cannibalisation is NOT deducted. A promotion takes volume from "
            "substitutes, so category profit is lower than the figure here. "
            "This is an upper bound.",
            "Pull-forward is handled by the estimand: the net figure includes "
            "the post-promotion dip, the gross figure does not.",
        ],
        warnings=warnings,
    )
    logger.info(
        "promo_uplift.business_impact",
        incremental_units=round(incremental_units, 1),
        incremental_profit=round(incremental_profit, 2),
        roi=round(roi, 3) if roi is not None else None,
    )
    return impact


def event_level_impact(
    cate: np.ndarray,
    analysis: AnalysisFrame,
    *,
    treated_rows: pd.DataFrame | None = None,
    config: PromoUpliftConfig | None = None,
) -> pd.DataFrame:
    """Per-event incremental profit and ROI, for Step 8 to allocate against.

    This is the table the future optimiser consumes: one row per candidate
    promotion, with the incremental profit it produced and the spend it
    consumed. Events are ranked by ROI, and value-destroying ones are kept in
    the table rather than filtered - the optimiser's job is partly to allocate
    *away* from them, which it cannot do if they are missing.

    ``treated_rows`` is the frame the ``cate`` values belong to. It matters that
    this is passed rather than inferred: the covariate frame drops rows without
    a complete pre-treatment history, so its treated rows are a *subset* of the
    analysis frame's. Aligning against the wrong one silently pairs each effect
    with a different promotion, which is worse than producing no table at all.
    """
    settings = config or get_promo_uplift_config()
    panel = analysis.frame

    if treated_rows is None:
        treated_mask = (panel["role"] == RowRole.TREATED).to_numpy()
        treated_rows = panel[treated_mask]

    if len(cate) != len(treated_rows):
        raise ValueError(
            f"cate has {len(cate)} values but treated_rows has {len(treated_rows)}"
        )

    treated = treated_rows.copy()
    treated["_cate"] = cate

    price, margin, rate = unit_economics(analysis, config=settings)

    aggregations: dict[str, tuple[str, str]] = {
        "product_id": ("product_id", "first"),
        "store_id": ("store_id", "first"),
        "treated_days": ("_cate", "size"),
        "uplift_units_per_day": ("_cate", "mean"),
        "observed_units": (settings.target, "sum"),
    }
    # Carried through for Step 9's allocator, which constrains spend by region
    # and category. Without them a constraint like "at least 20% in the North"
    # matches no candidate and cannot be applied at all.
    for dimension in ("region", "category", "channel"):
        if dimension in treated.columns:
            aggregations[dimension] = (dimension, "first")

    grouped = treated.groupby("promotion_id", observed=True).agg(**aggregations)
    grouped["incremental_units"] = (
        grouped["uplift_units_per_day"] * grouped["treated_days"]
    )
    grouped["incremental_revenue"] = grouped["incremental_units"] * price
    grouped["incremental_margin"] = grouped["incremental_units"] * margin

    events = analysis.events
    if "promotion_spend" in events.columns:
        spend = dict(zip(events["promotion_id"], events["promotion_spend"], strict=True))
        grouped["promotion_spend"] = [float(spend.get(pid, np.nan)) for pid in grouped.index]
    else:
        grouped["promotion_spend"] = np.nan
    grouped["incremental_profit"] = grouped["incremental_margin"] - grouped[
        "promotion_spend"
    ].fillna(0.0)
    grouped["roi"] = np.where(
        grouped["promotion_spend"] > 0,
        grouped["incremental_profit"] / grouped["promotion_spend"],
        np.nan,
    )
    grouped["value_destroying"] = grouped["incremental_profit"] < 0
    grouped["margin_rate"] = rate

    return grouped.sort_values("roi", ascending=False, na_position="last").reset_index()


__all__ = [
    "BusinessImpact",
    "business_impact",
    "event_level_impact",
    "promotion_spend",
    "unit_economics",
]
