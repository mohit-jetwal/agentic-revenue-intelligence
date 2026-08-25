"""Time features (brief section 13).

Calendar attributes are the one feature family that is *always* safe to read at
any horizon: the day of week for 2026-03-14 is knowable now, and so is whether
it is a public holiday. That is why ``calendar`` is classified
``KNOWN_IN_ADVANCE`` in :mod:`data.repositories.availability`.

On encoding (section 13's "avoid unnecessary one-hot expansion"): cyclical time
fields get **sine/cosine pairs** rather than dummies. Two reasons. One-hot on
day-of-year would add 366 columns to a frame that already has millions of rows.
More importantly, dummies discard the ordering - a model would have no way to
know that 31 December is adjacent to 1 January, so it cannot smooth across the
boundary, which is precisely where retail demand is most interesting.

Low-cardinality nominal fields (season, financial quarter) are left as
categorical labels for the model to encode as it sees fit. Tree models take them
directly; a linear model can dummy them cheaply at four levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.engineering.panel import DATE_KEY


def _cyclical(values: pd.Series, period: float, name: str) -> pd.DataFrame:
    """Sine/cosine encoding of a cyclical value.

    Maps a wrap-around quantity onto a circle so that the last value and the
    first are adjacent, which is exactly the property an integer column loses.
    """
    radians = 2.0 * np.pi * values.astype(float) / period
    return pd.DataFrame({f"{name}_sin": np.sin(radians), f"{name}_cos": np.cos(radians)})


def add_time_features(
    frame: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
    cyclical: bool = True,
) -> pd.DataFrame:
    """Derive calendar features from a date column.

    Standalone, so a caller without the calendar table still gets the basics.
    Holiday and festival flags need :func:`join_calendar_features` - those are
    genuine business data, not arithmetic on a timestamp.
    """
    result = frame.copy()
    dates = pd.to_datetime(result[date_column])

    result["day_of_week"] = dates.dt.dayofweek.astype("int8")
    result["day_of_month"] = dates.dt.day.astype("int8")
    result["day_of_year"] = dates.dt.dayofyear.astype("int16")
    result["week_of_year"] = dates.dt.isocalendar().week.astype("int8")
    result["month"] = dates.dt.month.astype("int8")
    result["quarter"] = dates.dt.quarter.astype("int8")
    result["year"] = dates.dt.year.astype("int16")
    result["weekend_flag"] = dates.dt.dayofweek >= 5

    # Indian financial year: April is month 1. Matches the calendar table built
    # in Step 2, so the two agree when both are present.
    result["financial_month"] = (((dates.dt.month - 4) % 12) + 1).astype("int8")
    result["financial_quarter"] = (((result["financial_month"] - 1) // 3) + 1).astype("int8")
    result["financial_year"] = np.where(
        dates.dt.month >= 4, dates.dt.year, dates.dt.year - 1
    ).astype("int16")

    if cyclical:
        for column, period, name in (
            ("day_of_week", 7.0, "dow"),
            ("month", 12.0, "month"),
            ("day_of_year", 365.25, "doy"),
        ):
            result = pd.concat([result, _cyclical(result[column], period, name)], axis=1)

    # Linear time index, so a model can fit trend without inferring it from
    # dates. Anchored on the frame's own minimum, so it is comparable within a
    # training set but must not be treated as an absolute scale across runs.
    result["time_index"] = (dates - dates.min()).dt.days.astype("int32")

    return result


def join_calendar_features(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
    columns: tuple[str, ...] = (
        "holiday_flag",
        "festival_flag",
        "season",
        "weekend_flag",
        "financial_month",
        "financial_quarter",
    ),
) -> pd.DataFrame:
    """Attach business calendar attributes.

    Holidays and festivals are business facts rather than arithmetic - Diwali
    moves with the lunar calendar and cannot be derived from a timestamp - so
    they come from the calendar table.

    A left join, so a date missing from the calendar keeps its row with null
    flags rather than vanishing. Silently dropping rows here would shorten a
    time series and corrupt every lag computed afterwards.
    """
    available = [c for c in columns if c in calendar.columns]
    if not available:
        return frame.copy()

    right = calendar[[date_column, *available]].copy()
    right[date_column] = pd.to_datetime(right[date_column])

    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column])

    # Drop overlapping columns from the left so the join does not produce
    # `_x`/`_y` suffixes that then break every downstream column reference.
    overlapping = [c for c in available if c in result.columns]
    if overlapping:
        result = result.drop(columns=overlapping)

    return result.merge(right, on=date_column, how="left")


def add_festival_proximity(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
) -> pd.DataFrame:
    """Days to the next festival and since the last one.

    Legitimately forward-looking: festival dates are published years ahead, so a
    planner on any date knows when the next one falls. This is the clearest
    example of why blanket as-of clamping would be wrong - it would delete
    information that genuinely exists.
    """
    if "festival_flag" not in calendar.columns:
        return frame.copy()

    festival_dates = pd.to_datetime(
        calendar.loc[calendar["festival_flag"].astype(bool), date_column]
    ).sort_values()
    if festival_dates.empty:
        return frame.copy()

    result = frame.copy()
    dates = pd.to_datetime(result[date_column])
    festivals = festival_dates.to_numpy()

    # searchsorted over a sorted array: O(n log m) rather than a cross join.
    next_index = np.searchsorted(festivals, dates.to_numpy(), side="left")
    previous_index = next_index - 1

    next_date = pd.Series(
        np.where(
            next_index < len(festivals),
            festivals[np.clip(next_index, 0, len(festivals) - 1)],
            np.datetime64("NaT"),
        ),
        index=result.index,
    )
    previous_date = pd.Series(
        np.where(
            previous_index >= 0,
            festivals[np.clip(previous_index, 0, len(festivals) - 1)],
            np.datetime64("NaT"),
        ),
        index=result.index,
    )

    result["days_to_festival"] = (pd.to_datetime(next_date) - dates).dt.days
    result["days_since_festival"] = (dates - pd.to_datetime(previous_date)).dt.days
    return result
