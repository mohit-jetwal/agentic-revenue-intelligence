"""Price and competitor features (brief sections 16-17).

Price is the one driver that is *both* an input a model reasons about and a
decision the business controls, which makes its features unusual: today's price
is legitimately known at prediction time, because the pricing manager set it.
So unlike sales, price features may use the current value.

That distinction is why ``pricing`` is classified ``KNOWN_IN_ADVANCE``. It also
means the guard in :func:`~features.engineering.panel.pct_change_on_shifted`
matters - comparing today's price to last week's is fine; comparing today's
*sales* to last week's is not.

**Definitions**, since brief sections 16-17 ask for them explicitly:

``price_index``
    Own selling price divided by the mean selling price of the same category on
    the same date. Above 1 means priced above the category. Category rather than
    a fixed basket, because the comparison a shopper makes is against the
    alternatives on the shelf, and because a fixed reference basket goes stale
    as the assortment changes.

``price_gap``
    ``own_price - competitor_effective_price``. Absolute currency, which is what
    a category manager negotiates in.

``price_ratio``
    ``own_price / competitor_effective_price``. Scale-free, so it is comparable
    across a Rs 30 SKU and a Rs 600 one, and it is the form that enters a
    log-log demand model linearly. Both are provided because they answer
    different questions.

``price_vs_rolling_average``
    Current price against this product-store's own trailing average. Captures
    "is this cheap *for this item*", which is the comparison that drives
    stock-up behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.engineering.panel import (
    DATE_KEY,
    PANEL_KEYS,
    expanding_on_shifted,
    pct_change_on_shifted,
    rolling_on_shifted,
    shifted_group,
)


def add_price_features(
    panel: pd.DataFrame,
    *,
    price_column: str = "selling_price",
    regular_column: str = "regular_price",
    keys: Sequence[str] = PANEL_KEYS,
    rolling_window: int = 28,
) -> pd.DataFrame:
    """Own-price features for a product-store panel."""
    result = panel.copy()

    if regular_column in result.columns:
        # Effective discount off list, recomputed rather than trusting a stored
        # column: the two must agree, and deriving it here means a discrepancy
        # in the source shows up as a contract failure rather than silently
        # feeding two different numbers to two different models.
        result["discount_depth"] = (
            1.0 - result[price_column] / result[regular_column].replace(0.0, np.nan)
        ).clip(lower=0.0)

    # Price change is knowable today: the manager set today's price.
    result["price_change_pct_1"] = pct_change_on_shifted(result, price_column, periods=1, keys=keys)
    result["price_change_pct_7"] = pct_change_on_shifted(result, price_column, periods=7, keys=keys)
    result["price_changed_flag"] = (
        result[price_column] != shifted_group(result, price_column, periods=1, keys=keys)
    ).fillna(False)

    # Trailing average excludes today, so "cheap relative to normal" is measured
    # against genuine history rather than partly against itself.
    trailing = rolling_on_shifted(
        result, price_column, window=rolling_window, statistic="mean", keys=keys
    )
    result[f"price_rolling_{rolling_window}"] = trailing
    result["price_vs_rolling_average"] = result[price_column] / trailing.replace(0.0, np.nan)

    result["historical_average_price"] = expanding_on_shifted(
        result, price_column, statistic="mean", keys=keys
    )

    return result


def add_price_index(
    panel: pd.DataFrame,
    products: pd.DataFrame,
    *,
    price_column: str = "selling_price",
    date_column: str = DATE_KEY,
    level: str = "category",
) -> pd.DataFrame:
    """Price relative to the category average on the same date.

    Computed per date so the index reflects the competitive set *as it was*, not
    as it is on average. A category-wide price rise should leave the index flat -
    a product is only "expensive" relative to its alternatives at the time.
    """
    if level not in products.columns:
        raise KeyError(f"products has no {level!r} column to index against")

    result = panel.copy()
    if level not in result.columns:
        result = result.merge(
            products[["product_id", level]].drop_duplicates("product_id"),
            on="product_id",
            how="left",
        )

    group_mean = result.groupby([date_column, level], observed=True)[price_column].transform("mean")
    result["price_index"] = result[price_column] / group_mean.replace(0.0, np.nan)
    return result


def add_competitor_features(
    panel: pd.DataFrame,
    competitor: pd.DataFrame,
    *,
    price_column: str = "selling_price",
    date_column: str = DATE_KEY,
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.DataFrame:
    """Competitor price position (brief section 17).

    Competitor data is at product x date, not per store - competitive
    intelligence arrives as market feeds rather than shelf-by-shelf - so it
    broadcasts across the stores carrying that product. That is the honest
    representation of what the business actually knows.

    Competitor price is **observed**, so under a point-in-time view it is
    already cut at the as-of date. A forecast horizon therefore has no
    competitor price, and the correct handling is to carry the last observed
    value forward, which :func:`~features.engineering.panel.shifted_group`-based
    features do naturally via the merge producing nulls the caller can ffill.
    """
    if competitor.empty:
        return panel.copy()

    price_field = (
        "competitor_effective_price"
        if "competitor_effective_price" in competitor.columns
        else "competitor_price"
    )

    aggregated = (
        competitor.groupby([date_column, "product_id"], observed=True)
        .agg(
            competitor_price=(price_field, "mean"),
            competitor_price_min=(price_field, "min"),
            competitor_discount=("competitor_discount", "mean"),
            competitor_promotion_flag=("competitor_promotion_flag", "max"),
        )
        .reset_index()
    )
    aggregated[date_column] = pd.to_datetime(aggregated[date_column])

    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    result = result.merge(aggregated, on=[date_column, "product_id"], how="left")

    own = result[price_column]
    rival = result["competitor_price"].replace(0.0, np.nan)

    # Both forms, because they answer different questions - see module docstring.
    result["price_gap"] = own - result["competitor_price"]
    result["price_ratio"] = own / rival
    result["competitor_price_index"] = rival / own.replace(0.0, np.nan)
    result["cheaper_than_competitor_flag"] = (own < result["competitor_price"]).fillna(False)

    # Competitor movement is knowable only in arrears, so this is a lagged
    # comparison by construction.
    result["competitor_price_change_pct_7"] = pct_change_on_shifted(
        result, "competitor_price", periods=7, keys=keys
    )

    return result
