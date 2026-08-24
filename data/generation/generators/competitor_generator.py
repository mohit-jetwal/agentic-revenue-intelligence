"""Competitor pricing.

Brief section 13 asks for a real relationship: competitor price down should
depress our demand, competitor price up should lift it. That relationship is
produced by the ``gamma`` term in the demand equation, drawn in
``ground_truth.competitor_sensitivity``; this module produces the price series
it acts on.

Two design points:

*Competitor price is tracked at product x competitor x day, not per store.*
Competitive intelligence in CPG arrives as market-level price feeds, not
store-by-store. Generating it per store would imply a data source that does not
exist and would inflate the table by two orders of magnitude for no gain.

*Competitor prices partly track the shared commodity cost index.* This is
confounder 5: our price and theirs move together because both respond to input
costs. A model that regresses our demand on competitor price without accounting
for that shared driver will misattribute a cost shock to competitive dynamics -
a mistake worth being able to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

_COMPETITOR_NAMES = ["RivalCorp", "PrimeGoods", "NextBrand", "AlphaTrade", "OmniMart"]


@dataclass
class CompetitorPaths:
    """Dense competitor price paths, aligned to the product axis."""

    #: (products, days) volume-weighted mean competitor price.
    mean_price: np.ndarray
    #: (products,) reference level, so log(price / ref) is centred near zero.
    reference_price: np.ndarray
    #: Long frame for the analytical table.
    frame: pd.DataFrame


def generate_competitor_pricing(
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    cost_index: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> CompetitorPaths:
    """Build competitor price series and the aggregate our demand responds to."""
    rng = rngs.get(Stream.COMPETITOR)

    n_products = len(products)
    n_days = len(calendar)
    dates = calendar["date"].to_numpy()
    n_competitors = min(config.scale.competitors, len(_COMPETITOR_NAMES))

    product_ids = products["product_id"].to_numpy()
    categories = products["category"].to_numpy()
    base_price = products["base_price"].to_numpy(dtype=float)

    cost_wide = cost_index.pivot(index="date", columns="category", values="cost_index")
    cost_wide = cost_wide.reindex(pd.Index(dates, name="date"))
    cost_by_category = {
        str(category): cost_wide[str(category)].to_numpy(dtype=float)
        for category in cost_wide.columns
    }

    years = max(n_days / 365.25, 1.0)
    low, high = config.competitor.price_changes_per_year
    correlation = config.competitor.cost_index_correlation

    # Accumulators for the volume-weighted mean across competitors.
    price_sum = np.zeros((n_products, n_days), dtype=np.float64)

    records: list[pd.DataFrame] = []

    for c in range(n_competitors):
        competitor_id = f"COMP{c + 1:02d}"
        competitor_name = _COMPETITOR_NAMES[c]

        # Each competitor holds a persistent price position relative to us.
        position = rng.uniform(*config.competitor.price_index_vs_ours, size=n_products)

        prices = np.empty((n_products, n_days), dtype=np.float32)
        promo_flags = np.zeros((n_products, n_days), dtype=bool)
        discounts = np.zeros((n_products, n_days), dtype=np.float32)

        for p in range(n_products):
            costs = cost_by_category[str(categories[p])]
            anchor = float(base_price[p] * position[p])

            n_change = int(rng.integers(max(int(low * years), 1), max(int(high * years), 2) + 1))
            change_days = np.sort(rng.choice(np.arange(1, n_days), size=n_change, replace=False))

            row = np.empty(n_days, dtype=np.float32)
            current = anchor
            cursor = 0
            previous = 0
            for change_day in change_days:
                row[cursor:change_day] = current
                # CONFOUNDER 5: partly the shared cost index, partly idiosyncratic.
                cost_move = float(costs[change_day] - costs[previous])
                delta = correlation * cost_move + (1.0 - correlation) * float(rng.normal(0.0, 0.03))
                current = float(np.clip(current * (1.0 + delta), anchor * 0.6, anchor * 1.7))
                cursor = int(change_day)
                previous = int(change_day)
            row[cursor:] = current
            prices[p] = row

            # Competitor promotional bursts, as short discount windows.
            n_promos = int(config.competitor.promotion_frequency * years * 12)
            for _ in range(max(n_promos, 0)):
                start = int(rng.integers(0, n_days))
                duration = int(rng.integers(4, 15))
                end = min(start + duration, n_days)
                depth = float(rng.uniform(0.08, 0.30))
                promo_flags[p, start:end] = True
                discounts[p, start:end] = depth

        effective = prices * (1.0 - discounts)
        price_sum += effective

        records.append(
            pd.DataFrame(
                {
                    "date": np.tile(dates, n_products),
                    "product_id": np.repeat(product_ids, n_days),
                    "competitor_id": competitor_id,
                    "competitor_name": competitor_name,
                    "competitor_product_id": np.repeat(
                        np.array([f"{competitor_id}-{pid}" for pid in product_ids]), n_days
                    ),
                    "competitor_price": np.round(prices.reshape(-1), 2),
                    "competitor_discount": np.round(discounts.reshape(-1) * 100.0, 2),
                    "competitor_promotion_flag": promo_flags.reshape(-1),
                    "competitor_effective_price": np.round(effective.reshape(-1), 2),
                }
            )
        )

    mean_price = (price_sum / max(n_competitors, 1)).astype(np.float32)
    # Reference = each product's own mean over the window, so the log ratio is
    # centred and gamma is interpretable as an elasticity around normal levels.
    reference = mean_price.mean(axis=1).astype(np.float32)

    frame = pd.concat(records, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date

    return CompetitorPaths(mean_price=mean_price, reference_price=reference, frame=frame)
