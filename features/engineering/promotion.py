"""Promotion features (brief section 18).

Section 18 makes the delicate point directly: ``days_until_promotion_end`` is
legitimate *only when that information would genuinely be known at prediction
time*. It is, and this module explains why - but the reasoning has to be
explicit, because this is the one feature family where forward-looking values
are correct rather than leakage.

**What is knowable ahead.** Promotion mechanics are agreed with retailers weeks
in advance: which SKU, which stores, what depth, which dates. A demand planner
on 1 June genuinely knows a 20% promotion runs 10-24 June. Features describing
the *schedule* - active flag, depth, duration, days remaining, days until the
next one - are therefore fair.

**What is not.** Anything realised. ``promotion_spend`` and ``promotion_units``
are actuals booked after the fact; on a future-dated promotion they do not exist
yet. :class:`~data.repositories.point_in_time.PointInTimeView` nulls them beyond
the as-of date, and the features here that use spend are marked as
backward-looking only.

The distinction is not academic. A model handed a future promotion's realised
spend has been handed a function of the demand it is trying to predict.
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
)


def expand_promotion_calendar(
    promotions: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
) -> pd.DataFrame:
    """Expand promotion events into one row per active day.

    Events are stored as windows; the panel is daily. Expanding lets the join be
    a simple key merge rather than an interval join, which pandas does not do
    efficiently, and which would dominate runtime at panel scale.
    """
    if promotions.empty:
        return pd.DataFrame(
            columns=[
                date_column,
                "product_id",
                "store_id",
                "promotion_id",
                "promotion_type",
                "promotion_discount",
                "promotion_duration",
                "display_flag",
                "bundle_flag",
                "promotion_spend",
            ]
        )

    events = promotions.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    # Build the day range per event, then explode. Vectorised over events; the
    # only Python-level iteration is over event count, not panel rows.
    events["_dates"] = [
        pd.date_range(start, end, freq="D")
        for start, end in zip(events["start_date"], events["end_date"], strict=True)
    ]
    expanded = events.explode("_dates", ignore_index=True)
    expanded[date_column] = pd.to_datetime(expanded["_dates"])

    expanded["promotion_discount"] = expanded["discount_percentage"] / 100.0
    expanded["promotion_duration"] = expanded["duration_days"]
    expanded["days_into_promotion"] = (expanded[date_column] - expanded["start_date"]).dt.days
    expanded["days_until_promotion_end"] = (expanded["end_date"] - expanded[date_column]).dt.days

    columns = [
        date_column,
        "product_id",
        "store_id",
        "promotion_id",
        "promotion_type",
        "promotion_discount",
        "promotion_duration",
        "days_into_promotion",
        "days_until_promotion_end",
        "display_flag",
        "bundle_flag",
    ]
    if "promotion_spend" in expanded.columns:
        columns.append("promotion_spend")
    if "promotion_units" in expanded.columns:
        columns.append("promotion_units")

    result = expanded[columns]
    # Overlapping promotions on one product-store-day would duplicate panel rows
    # on merge. The generator prevents overlaps, but a real feed would not, so
    # the deepest discount wins - that is the offer a shopper would take.
    return (
        result.sort_values("promotion_discount", ascending=False)
        .drop_duplicates(subset=[date_column, "product_id", "store_id"], keep="first")
        .reset_index(drop=True)
    )


def add_promotion_features(
    panel: pd.DataFrame,
    promotions: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
    keys: Sequence[str] = PANEL_KEYS,
    include_spend: bool = True,
) -> pd.DataFrame:
    """Attach promotion schedule features to a panel.

    ``include_spend`` controls whether realised spend features are added. Set it
    ``False`` for a forecasting feature set covering future dates, where spend
    is unknowable; leave it ``True`` for uplift and ROI work over history.
    """
    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    calendar = expand_promotion_calendar(promotions, date_column=date_column)

    if calendar.empty:
        result["promotion_flag"] = False
        result["promotion_discount"] = 0.0
        result["days_since_promotion"] = np.nan
        return result

    overlapping = [
        c
        for c in calendar.columns
        if c in result.columns and c not in (date_column, "product_id", "store_id")
    ]
    if overlapping:
        result = result.drop(columns=overlapping)

    result = result.merge(calendar, on=[date_column, "product_id", "store_id"], how="left")

    result["promotion_flag"] = result["promotion_id"].notna()
    result["promotion_discount"] = result["promotion_discount"].fillna(0.0)
    result["promotion_duration"] = result["promotion_duration"].fillna(0)
    result["days_into_promotion"] = result["days_into_promotion"].fillna(-1)
    result["days_until_promotion_end"] = result["days_until_promotion_end"].fillna(-1)
    for flag in ("display_flag", "bundle_flag"):
        if flag in result.columns:
            result[flag] = as_bool(result[flag])

    # Backward-looking history. Sorted first because both helpers rely on panel
    # ordering to shift correctly.
    result = result.sort_values([*keys, date_column]).reset_index(drop=True)
    result["days_since_promotion"] = days_since_flag(
        result, "promotion_flag", keys=keys, date_column=date_column
    )
    result["promotions_last_28d"] = rolling_count_of_flag(
        result, "promotion_flag", window=28, keys=keys
    )
    result["promotions_last_90d"] = rolling_count_of_flag(
        result, "promotion_flag", window=90, keys=keys
    )

    if include_spend and "promotion_spend" in result.columns:
        result["promotion_spend"] = result["promotion_spend"].fillna(0.0)
        # Spend per day of the event, so a long cheap promotion and a short
        # expensive one are comparable.
        duration = result["promotion_duration"].replace(0, np.nan)
        result["promotion_intensity"] = result["promotion_spend"] / duration
    elif not include_spend:
        # Dropped rather than zeroed: a zero would read as "this promotion cost
        # nothing", which is a claim. Absence is the honest representation.
        result = result.drop(
            columns=[c for c in ("promotion_spend", "promotion_units") if c in result.columns]
        )

    return result


def add_time_to_next_promotion(
    panel: pd.DataFrame,
    promotions: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
) -> pd.DataFrame:
    """Days until the next scheduled promotion starts.

    Forward-looking and legitimate, for the reason in the module docstring: the
    promotion calendar is committed ahead. Genuinely useful too - demand
    typically softens just before a known promotion as the trade holds off.
    """
    if promotions.empty:
        result = panel.copy()
        result["days_to_next_promotion"] = np.nan
        return result

    starts = promotions[["product_id", "store_id", "start_date"]].copy()
    starts["start_date"] = pd.to_datetime(starts["start_date"])
    starts = starts.sort_values("start_date")

    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    # merge_asof with direction="forward" finds the next start per group in one
    # sorted pass, rather than a cross join per product-store.
    left = result.sort_values(date_column)
    merged = pd.merge_asof(
        left,
        starts.rename(columns={"start_date": "_next_promotion_start"}),
        left_on=date_column,
        right_on="_next_promotion_start",
        by=["product_id", "store_id"],
        direction="forward",
        allow_exact_matches=True,
    )
    merged["days_to_next_promotion"] = (
        merged["_next_promotion_start"] - merged[date_column]
    ).dt.days
    return merged.drop(columns=["_next_promotion_start"])
