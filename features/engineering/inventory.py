"""Inventory features (brief section 19).

Inventory is the most leakage-prone family in the set, and the reason is worth
stating plainly: **closing inventory on day t is a function of day t's sales**.
``closing = opening + received - sold``. Handing a model ``closing_inventory``
for the day it is predicting is handing it the answer, rearranged.

So the rule here is stricter than elsewhere. Only *opening* position and
*shifted* quantities are exposed. ``inventory_available`` (opening + received)
is knowable at the start of the day, before any sale, and is therefore fair.
Everything derived from ``sold_units`` or ``closing_inventory`` on the current
row is shifted by at least one day.

Section 19 also asks for ``days_since_stockout`` and ``rolling_stockout_count``,
both of which summarise history rather than the present - they use the shared
shifted helpers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.engineering.panel import (
    DATE_KEY,
    PANEL_KEYS,
    as_bool,
    days_since_flag,
    rolling_count_of_flag,
    rolling_on_shifted,
    shifted_group,
)

#: Columns that are a function of the current day's sales. Never exposed
#: unshifted - see the module docstring.
CENSORED_COLUMNS: tuple[str, ...] = ("closing_inventory", "sold_units", "inventory_days")


def add_inventory_features(
    panel: pd.DataFrame,
    inventory: pd.DataFrame | None = None,
    *,
    date_column: str = DATE_KEY,
    keys: Sequence[str] = PANEL_KEYS,
    drop_censored: bool = True,
) -> pd.DataFrame:
    """Attach availability features.

    ``drop_censored`` removes the columns that encode the current day's sales.
    Default ``True``, because leaving them is a silent leak and a caller who
    genuinely wants them (a reconciliation report, say) can ask.
    """
    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    if inventory is not None and not inventory.empty:
        right = inventory.copy()
        right[date_column] = pd.to_datetime(right[date_column])
        overlapping = [
            c
            for c in right.columns
            if c in result.columns and c not in (date_column, "product_id", "store_id")
        ]
        right = right.drop(columns=overlapping)
        result = result.merge(right, on=[date_column, "product_id", "store_id"], how="left")

    result = result.sort_values([*keys, date_column]).reset_index(drop=True)

    # Knowable at the start of the day, before a single unit sells.
    if {"opening_inventory", "received_units"} <= set(result.columns):
        result["inventory_available"] = result["opening_inventory"].fillna(0) + result[
            "received_units"
        ].fillna(0)

    if "stockout_flag" in result.columns:
        flags = result["stockout_flag"].astype(bool)
        result["stockout_flag"] = flags
        # Yesterday's stockout is knowable today; today's is not, since it is
        # determined by whether today's demand exceeded supply.
        result["stockout_yesterday"] = as_bool(
            shifted_group(result, "stockout_flag", periods=1, keys=keys)
        )
        result["days_since_stockout"] = days_since_flag(
            result, "stockout_flag", keys=keys, date_column=date_column
        )
        result["stockouts_last_28d"] = rolling_count_of_flag(
            result, "stockout_flag", window=28, keys=keys
        )
        result["stockouts_last_90d"] = rolling_count_of_flag(
            result, "stockout_flag", window=90, keys=keys
        )

    # Cover, computed against *trailing* demand rather than today's. Using
    # today's sold_units would be circular - it is the quantity being predicted.
    if "inventory_available" in result.columns:
        trailing_demand = (
            rolling_on_shifted(result, "units", window=28, statistic="mean", keys=keys)
            if "units" in result.columns
            else None
        )

        if trailing_demand is not None:
            result["inventory_days_cover"] = result[
                "inventory_available"
            ] / trailing_demand.replace(0.0, np.nan)
            # Availability relative to normal demand. Below 1 means less than a
            # day of cover on the shelf - a genuine stockout risk signal.
            result["inventory_ratio"] = result["inventory_available"] / trailing_demand.replace(
                0.0, np.nan
            )

    if drop_censored:
        present = [c for c in CENSORED_COLUMNS if c in result.columns]
        if present:
            # Retained in shifted form so the information is not simply lost:
            # yesterday's closing position is legitimate and useful.
            for column in present:
                result[f"{column}_lag_1"] = shifted_group(result, column, periods=1, keys=keys)
            result = result.drop(columns=present)

    return result
