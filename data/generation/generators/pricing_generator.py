"""Pricing, and the confounding that makes elasticity estimation non-trivial.

If prices moved at random, recovering elasticity would be arithmetic and the
whole project a toy. Real prices do not move at random, and the ways they move
are exactly what breaks naive estimation. Three mechanisms are built in here:

**Price endogeneity (the problem).** Managers raise prices into anticipated
strong demand and discount into weak demand. A regression of log quantity on log
price then partly recovers the pricing manager's behaviour rather than the
shopper's, biasing elasticity *toward zero* - products look less price-sensitive
than they are, which encourages exactly the wrong recommendation.

**A commodity cost index (the instrument).** Input costs shift price through
pass-through but do not enter demand directly. That is a textbook instrument, so
Step 8 can compare naive OLS, panel fixed effects and 2SLS against known truth
and show which one works.

**Randomised price tests (the clean subset).** A configurable fraction of price
changes are exogenous, tagged in ``price_change_reason``. On that subset even a
simple estimator is unbiased, giving a second, independent route to the answer.

Prices move in *steps*, not daily. Real price files are event-based - a few
changes a year, held flat in between. Daily jitter would create variation that
does not exist and make elasticity look far easier to estimate than it is.

Output is a dense ``(pairs, days)`` float32 matrix rather than a long dataframe.
At dev scale the long form is 6.6M rows whose id columns would be repeated
object strings - hundreds of megabytes for information already implied by the
row index. The pipeline slices the matrix into date windows when writing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.generation.calendar_math import annual_seasonality_series
from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

#: Index positions match the codes stored in ``PricePaths.change_reason``.
PRICE_CHANGE_REASONS = np.array(
    ["none", "scheduled", "cost_passthrough", "randomised_test"], dtype=object
)


@dataclass
class PricePaths:
    """Dense daily price paths for every product-store listing."""

    #: (pairs, days) regular price.
    regular_price: np.ndarray
    #: (pairs, days) True on days the regular price changed.
    change_flag: np.ndarray
    #: (pairs, days) index into :data:`PRICE_CHANGE_REASONS`.
    change_reason: np.ndarray
    #: (pairs,) the pair's undiscounted anchor price, used as the elasticity
    #: reference point so log(price / ref) is centred near zero.
    reference_price: np.ndarray


def generate_cost_index(
    calendar: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> pd.DataFrame:
    """Commodity cost index per category - the instrument for price.

    A mean-reverting random walk, which is what input-cost series look like.
    Category-specific because edible oil and shampoo inputs do not move
    together; a single shared index would be nearly collinear with time and so
    useless as an instrument once time fixed effects are included.
    """
    rng = rngs.get(Stream.COST_INDEX)
    n_days = len(calendar)
    daily_volatility = config.pricing.cost_index_volatility / np.sqrt(365.0)

    frames: list[pd.DataFrame] = []
    for category in sorted(config.categories):
        shocks = rng.normal(0.0, daily_volatility, size=n_days)
        level = np.zeros(n_days)
        for t in range(1, n_days):
            # Mean reversion keeps the index from wandering off over three years
            # and swamping every other effect in the price equation.
            level[t] = level[t - 1] * 0.998 + shocks[t]

        frames.append(
            pd.DataFrame(
                {
                    "date": calendar["date"].to_numpy(),
                    "category": category,
                    "cost_index": np.round(np.exp(level), 6),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def generate_price_paths(
    listings: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    cost_index: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> PricePaths:
    """Build step-change regular-price paths with the confounders described above.

    Produces *regular* price only. Selling price is regular price less any active
    promotional discount, resolved in the sales generator once the promotion
    calendar is known.
    """
    rng = rngs.get(Stream.PRICING)

    n_pairs = len(listings)
    n_days = len(calendar)
    dates = calendar["date"].to_numpy()
    day_of_year = calendar["day_of_year"].to_numpy()

    product_lookup = products.set_index("product_id")
    base_price = listings["product_id"].map(product_lookup["base_price"]).to_numpy(dtype=float)
    categories = listings["product_id"].map(product_lookup["category"]).to_numpy()

    # Regional price positioning: the same SKU is not priced identically in
    # every region. Also a genuine source of cross-sectional price variation a
    # panel model can exploit.
    store_lookup = stores.set_index("store_id")
    regions = listings["store_id"].map(store_lookup["region"]).to_numpy()
    spread = config.pricing.regional_price_spread
    region_offsets = {
        region: float(rng.uniform(-spread, spread)) for region in sorted(set(regions.tolist()))
    }
    regional_factor = np.array([1.0 + region_offsets[str(r)] for r in regions])
    store_factor = np.exp(rng.normal(0.0, 0.02, size=n_pairs))
    anchor_price = base_price * regional_factor * store_factor

    # Cost index per category, aligned to the calendar.
    cost_wide = cost_index.pivot(index="date", columns="category", values="cost_index")
    cost_wide = cost_wide.reindex(pd.Index(dates, name="date"))
    cost_by_category = {
        str(category): cost_wide[str(category)].to_numpy(dtype=float)
        for category in cost_wide.columns
    }

    # Seasonal shape per category - deliberately the same curve the demand
    # equation uses, because that shared term is what creates the endogeneity.
    seasonal_by_category = {
        name: annual_seasonality_series(
            day_of_year, category.seasonal_amplitude, category.seasonal_peak_month
        )
        for name, category in config.categories.items()
    }

    years = max(n_days / 365.25, 1.0)
    low, high = config.pricing.price_changes_per_year
    n_changes = rng.integers(max(int(low * years), 1), max(int(high * years), 2) + 1, size=n_pairs)

    regular_price = np.empty((n_pairs, n_days), dtype=np.float32)
    change_flag = np.zeros((n_pairs, n_days), dtype=bool)
    change_reason = np.zeros((n_pairs, n_days), dtype=np.int8)

    endogeneity = config.pricing.endogeneity_strength
    passthrough = config.pricing.cost_passthrough
    test_fraction = config.pricing.randomised_test_fraction
    magnitude_low, magnitude_high = config.pricing.price_change_magnitude

    # A 30-day burn-in guarantees every pair has pre-change history, so a
    # before/after comparison is always possible.
    burn_in = min(30, max(n_days // 8, 1))

    for i in range(n_pairs):
        category = str(categories[i])
        seasonal = seasonal_by_category[category]
        costs = cost_by_category[category]

        available = n_days - burn_in
        n_change = int(min(n_changes[i], max(available, 1)))
        change_days = np.sort(rng.choice(np.arange(burn_in, n_days), size=n_change, replace=False))

        row = np.empty(n_days, dtype=np.float32)
        current = float(anchor_price[i])
        cursor = 0
        previous_change = 0

        for change_day in change_days:
            row[cursor:change_day] = current

            if rng.random() < test_fraction:
                # Exogenous by construction: unrelated to demand or cost. This
                # is the subset on which even naive OLS is unbiased.
                delta = float(rng.uniform(magnitude_low, magnitude_high))
                reason = 3
            else:
                delta = float(rng.uniform(magnitude_low, magnitude_high))
                # CONFOUNDER: lean price into anticipated seasonal demand.
                delta += endogeneity * 0.05 * float(seasonal[change_day])
                # INSTRUMENT: pass input-cost movement through to price.
                cost_move = float(costs[change_day] - costs[previous_change])
                delta += passthrough * cost_move
                reason = 2 if abs(cost_move) > 0.01 else 1

            current = float(
                np.clip(current * (1.0 + delta), anchor_price[i] * 0.55, anchor_price[i] * 1.9)
            )
            change_flag[i, change_day] = True
            change_reason[i, change_day] = reason
            cursor = int(change_day)
            previous_change = int(change_day)

        row[cursor:] = current
        regular_price[i] = np.round(row, 2)

    return PricePaths(
        regular_price=regular_price,
        change_flag=change_flag,
        change_reason=change_reason,
        reference_price=anchor_price.astype(np.float32),
    )
