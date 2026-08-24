"""Promotions, their response curves, and the pull-forward that follows.

Three properties this module exists to create, each of which makes a *later*
step honest:

**Diminishing returns.** ``uplift = a * (1 - exp(-b * discount))``, additive in
log space. A 10% discount buys ~12% uplift; 30% buys ~28%, not 36%. Without
saturation, Step 7's optimiser would pour the entire budget into the deepest
discount available, which is both wrong and obviously wrong to any category
manager.

**Pull-forward.** After a promotion ends, demand dips for a couple of weeks
because buyers loaded their pantry. This is the single most important realism
detail here: it is *why* a naive "sales during promo minus sales before promo"
calculation overstates incrementality. Without it, Step 6 could use the naive
method and look correct, and the Critic in Step 18 would have nothing real to
catch.

**Targeting.** Promotions are not assigned at random - they go to products whose
baseline is already soft, or that are already seasonal winners. That correlation
between treatment and outcome is the confounder Step 6 has to overcome, and it
is what makes a proper control group necessary rather than decorative.

Promotion *effectiveness* varies by product, type and region, so some events are
genuinely value-destroying: positive uplift, negative ROI. Step 7 needs those to
have anything interesting to allocate away from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.generation.calendar_math import annual_seasonality_series
from data.generation.coerce import as_float
from data.generation.config import GenerationConfig
from data.generation.ground_truth import GroundTruth
from data.generation.rng import RngFactory, Stream


@dataclass
class PromotionPaths:
    """Dense promotional state per product-store listing."""

    #: (pairs, days) fractional discount in [0, 1).
    discount: np.ndarray
    #: (pairs, days) log-space demand lift from the promotion mechanic.
    lift: np.ndarray
    #: (pairs, days) negative log-space term from post-promotion pantry loading.
    pull_forward: np.ndarray
    #: (pairs, days) index into ``promotions.promotion_id``; -1 when no promotion.
    promotion_index: np.ndarray
    #: Event-level analytical table.
    frame: pd.DataFrame


def generate_promotions(
    listings: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    ground_truth: GroundTruth,
    config: GenerationConfig,
    rngs: RngFactory,
) -> PromotionPaths:
    """Build the promotion calendar and its demand effects."""
    rng = rngs.get(Stream.PROMOTION)
    settings = config.promotions

    n_pairs = len(listings)
    n_days = len(calendar)
    dates = calendar["date"].to_numpy()
    day_of_year = calendar["day_of_year"].to_numpy()

    product_ids = listings["product_id"].to_numpy()
    store_ids = listings["store_id"].to_numpy()

    product_lookup = products.set_index("product_id")
    categories = listings["product_id"].map(product_lookup["category"]).to_numpy()

    store_lookup = stores.set_index("store_id")
    regions = listings["store_id"].map(store_lookup["region"]).to_numpy()
    promo_responsiveness = (
        listings["store_id"].map(store_lookup["_promo_responsiveness"]).to_numpy(dtype=float)
    )

    type_names = list(settings.types)
    type_weights = np.array([settings.types[t].weight for t in type_names], dtype=float)
    type_weights = type_weights / type_weights.sum()

    # Seasonal shape per category, used for the targeting confounder.
    seasonal_by_category = {
        name: annual_seasonality_series(
            day_of_year, category.seasonal_amplitude, category.seasonal_peak_month
        )
        for name, category in config.categories.items()
    }

    discount = np.zeros((n_pairs, n_days), dtype=np.float32)
    lift = np.zeros((n_pairs, n_days), dtype=np.float32)
    pull_forward = np.zeros((n_pairs, n_days), dtype=np.float32)
    promotion_index = np.full((n_pairs, n_days), -1, dtype=np.int32)

    years = max(n_days / 365.25, 1.0)
    low, high = settings.events_per_product_per_year
    duration_low, duration_high = settings.duration_days
    depth_low, depth_high = settings.discount_depth
    targeting = settings.targeting_strength
    decay_days = settings.pull_forward_decay_days
    decay_weights = np.exp(-np.arange(decay_days) / max(decay_days / 2.5, 1e-6))
    decay_weights = decay_weights / decay_weights.sum()

    events: list[dict[str, object]] = []

    for i in range(n_pairs):
        product_id = str(product_ids[i])
        category = str(categories[i])
        region = str(regions[i])
        seasonal = seasonal_by_category[category]
        responsiveness = float(promo_responsiveness[i])

        n_events = int(rng.integers(max(int(low * years), 1), max(int(high * years), 2) + 1))

        # Candidate start days, biased by the targeting confounder toward
        # periods the merchandiser expects to matter.
        weights = np.exp(targeting * 2.0 * seasonal)
        weights = weights / weights.sum()

        starts = rng.choice(n_days, size=n_events, replace=False, p=weights)

        for start in np.sort(starts):
            start = int(start)
            duration = int(rng.integers(duration_low, duration_high + 1))
            end = min(start + duration, n_days)
            if end <= start:
                continue
            # Skip overlaps: two promotions on the same listing on the same day
            # would make the event table ambiguous and uplift unattributable.
            if promotion_index[i, start:end].max() >= 0:
                continue

            type_name = str(rng.choice(type_names, p=type_weights))
            type_config = settings.types[type_name]
            depth = float(rng.uniform(depth_low, depth_high))

            response = ground_truth.promo_response[product_id][type_name]
            a = float(response["a"])
            b = float(response["b"])

            # Regional effectiveness spread, plus the store's segment-driven
            # promo responsiveness.
            regional_factor = 1.0 + 0.18 * float(rng.normal(0.0, 1.0))
            saturating = a * (1.0 - np.exp(-b * depth))
            event_lift = float(saturating * responsiveness * max(regional_factor, 0.4))

            has_display = rng.random() < type_config.display
            has_bundle = rng.random() < type_config.bundle
            if has_display:
                event_lift += settings.display_lift
            if has_bundle:
                event_lift += settings.bundle_lift

            event_id = len(events)
            discount[i, start:end] = depth
            lift[i, start:end] = event_lift
            promotion_index[i, start:end] = event_id

            # Pantry loading: a share of the incremental demand is borrowed from
            # the days after the promotion, decaying away.
            tail_start = end
            tail_end = min(end + decay_days, n_days)
            if tail_end > tail_start:
                borrowed = settings.pull_forward_fraction * event_lift
                window = decay_weights[: tail_end - tail_start]
                pull_forward[i, tail_start:tail_end] -= (
                    borrowed * window / max(window.sum(), 1e-9)
                ).astype(np.float32)

            events.append(
                {
                    "promotion_id": f"PR{event_id + 1:07d}",
                    "product_id": product_id,
                    "store_id": str(store_ids[i]),
                    "promotion_type": type_name,
                    "start_date": dates[start],
                    "end_date": dates[end - 1],
                    "duration_days": end - start,
                    "discount_percentage": round(depth * 100.0, 2),
                    "display_flag": bool(has_display),
                    "bundle_flag": bool(has_bundle),
                    "promotion_channel": str(
                        rng.choice(["In-store", "Digital", "Both"], p=[0.45, 0.25, 0.30])
                    ),
                    "region": region,
                    # Spend is finalised after sales are known (it depends on
                    # units for per-unit mechanics); the fixed component is set
                    # here so the event table is complete on its own.
                    "fixed_spend": round(float(rng.uniform(*settings.fixed_spend)), 2),
                    "spend_per_unit": type_config.spend_per_unit,
                    "_pair_index": i,
                    "_start_day": start,
                    "_end_day": end,
                }
            )

    columns = [
        "promotion_id",
        "product_id",
        "store_id",
        "promotion_type",
        "start_date",
        "end_date",
        "duration_days",
        "discount_percentage",
        "display_flag",
        "bundle_flag",
        "promotion_channel",
        "region",
        "fixed_spend",
        "spend_per_unit",
        "_pair_index",
        "_start_day",
        "_end_day",
    ]
    frame = pd.DataFrame(events, columns=columns) if events else pd.DataFrame(columns=columns)

    return PromotionPaths(
        discount=discount,
        lift=lift,
        pull_forward=pull_forward,
        promotion_index=promotion_index,
        frame=frame,
    )


def generate_trade_promotions(
    promotions: pd.DataFrame,
    products: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> pd.DataFrame:
    """Retailer-level trade promotion plans with budgets, uplift and ROI.

    This is the table Step 7 optimises over. Two properties make that
    optimisation a real problem rather than a formality:

    * ROI varies systematically by product and region, so there is genuinely a
      better and a worse allocation to find.
    * Planned and actual diverge. Real trade spend overruns and real uplift
      undershoots the plan, so a model that trusts ``expected_uplift`` will be
      beaten by one that learns from ``actual_uplift``.
    """
    rng = rngs.get(Stream.TRADE_PROMOTION)
    settings = config.trade_promotions

    if promotions.empty:
        return pd.DataFrame(
            columns=[
                "trade_promo_id",
                "retailer",
                "product_id",
                "region",
                "start_date",
                "end_date",
                "planned_spend",
                "actual_spend",
                "expected_uplift",
                "actual_uplift",
                "margin",
                "roi",
            ]
        )

    retailers = settings.retailers
    years = max((config.time.end_date - config.time.start_date).days / 365.25, 1.0)
    low, high = settings.events_per_retailer_per_year
    target_events = int(len(retailers) * rng.integers(low, high + 1) * years)
    target_events = max(min(target_events, len(promotions)), 1)

    sampled = promotions.sample(n=target_events, random_state=int(rng.integers(0, 2**31 - 1)))

    product_margin = products.set_index("product_id")["_margin"]

    # Retailer efficiency: some partners simply execute better. A persistent
    # effect, so the optimiser can learn to favour them.
    retailer_efficiency = {r: float(rng.uniform(0.7, 1.3)) for r in retailers}
    region_efficiency = {
        r: float(rng.uniform(0.8, 1.25)) for r in sorted(promotions["region"].unique())
    }

    rows: list[dict[str, object]] = []
    for n, promo in enumerate(sampled.itertuples(index=False)):
        retailer = str(rng.choice(retailers))
        region = str(promo.region)
        margin = float(product_margin.get(str(promo.product_id), 0.3))

        planned_spend = as_float(promo.fixed_spend) * float(rng.uniform(0.8, 1.6))
        # Overruns are more common than underspend.
        variance = settings.planned_vs_actual_spend_variance
        actual_spend = planned_spend * float(1.0 + rng.normal(0.03, variance))
        actual_spend = max(actual_spend, planned_spend * 0.5)

        expected_uplift = as_float(promo.discount_percentage) / 100.0 * float(rng.uniform(0.9, 1.5))
        efficiency = retailer_efficiency[retailer] * region_efficiency[region]
        actual_uplift = expected_uplift * efficiency
        actual_uplift *= float(1.0 + rng.normal(0.0, settings.expected_vs_actual_uplift_variance))
        actual_uplift = max(actual_uplift, -0.05)

        # Incremental profit relative to spend. Deliberately allowed to fall
        # below 1.0 - value-destroying promotions must exist in the data.
        incremental_revenue = actual_uplift * planned_spend * float(rng.uniform(2.0, 6.0))
        incremental_profit = incremental_revenue * margin
        roi = incremental_profit / max(actual_spend, 1.0)

        rows.append(
            {
                "trade_promo_id": f"TP{n + 1:06d}",
                "retailer": retailer,
                "product_id": str(promo.product_id),
                "region": region,
                "start_date": promo.start_date,
                "end_date": promo.end_date,
                "planned_spend": round(planned_spend, 2),
                "actual_spend": round(actual_spend, 2),
                "expected_uplift": round(expected_uplift, 4),
                "actual_uplift": round(actual_uplift, 4),
                "margin": round(margin, 4),
                "roi": round(roi, 4),
            }
        )

    frame = pd.DataFrame(rows)

    # Scale spend so the annual total lands near the configured budget, which is
    # what makes "how should we allocate Rs 10M?" a well-posed question.
    total_planned = frame["planned_spend"].sum()
    if total_planned > 0:
        scale = (settings.annual_budget * years) / total_planned
        frame["planned_spend"] = (frame["planned_spend"] * scale).round(2)
        frame["actual_spend"] = (frame["actual_spend"] * scale).round(2)

    return frame
