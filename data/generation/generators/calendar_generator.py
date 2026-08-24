"""Calendar dimension.

Provides the time features every downstream model needs: weekly and annual
seasonality anchors, holiday and festival flags, and a financial calendar.

Festivals are geography-configurable (brief section 7). The India set matters
more than it first appears: Diwali is a genuine demand event in CPG, and its
date moves 10-20 days year to year with the lunar calendar. A fixed-date
approximation would let a forecasting model learn "late October" and score well
for the wrong reason, so the actual observed dates are used.

The Indian financial year runs April-March, so ``financial_month`` is offset
from the calendar month rather than equal to it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Fixed-date public holidays (India), month-day.
_FIXED_HOLIDAYS_IN: dict[tuple[int, int], str] = {
    (1, 1): "New Year",
    (1, 26): "Republic Day",
    (5, 1): "Labour Day",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (12, 25): "Christmas",
}

# Lunar/observed festival dates. Hard-coded per year because they genuinely move;
# deriving them would need a full panchang implementation for no analytical gain.
_FESTIVALS_IN: dict[str, list[date]] = {
    "Holi": [date(2023, 3, 8), date(2024, 3, 25), date(2025, 3, 14), date(2026, 3, 4)],
    "Eid al-Fitr": [date(2023, 4, 22), date(2024, 4, 11), date(2025, 3, 31), date(2026, 3, 20)],
    "Raksha Bandhan": [date(2023, 8, 31), date(2024, 8, 19), date(2025, 8, 9), date(2026, 8, 28)],
    "Ganesh Chaturthi": [date(2023, 9, 19), date(2024, 9, 7), date(2025, 8, 27), date(2026, 9, 14)],
    "Dussehra": [date(2023, 10, 24), date(2024, 10, 12), date(2025, 10, 2), date(2026, 10, 20)],
    "Diwali": [date(2023, 11, 12), date(2024, 11, 1), date(2025, 10, 20), date(2026, 11, 8)],
}

_FIXED_HOLIDAYS_GLOBAL: dict[tuple[int, int], str] = {
    (1, 1): "New Year",
    (12, 25): "Christmas",
    (12, 31): "New Year's Eve",
}

_SEASON_BY_MONTH_IN = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Summer",
    5: "Summer",
    6: "Summer",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Autumn",
    11: "Autumn",
    12: "Winter",
}


def generate_calendar(
    start_date: date,
    end_date: date,
    *,
    geography: str = "IN",
    festival_lead_days: int = 6,
) -> pd.DataFrame:
    """Build the date dimension for the inclusive range."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    frame = pd.DataFrame({"date": dates})

    frame["day"] = frame["date"].dt.day.astype("int16")
    frame["week"] = frame["date"].dt.isocalendar().week.astype("int16")
    frame["month"] = frame["date"].dt.month.astype("int8")
    frame["quarter"] = frame["date"].dt.quarter.astype("int8")
    frame["year"] = frame["date"].dt.year.astype("int16")
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype("int8")  # Monday = 0
    frame["day_name"] = frame["date"].dt.day_name().astype("string")
    frame["week_of_year"] = frame["week"]
    frame["day_of_year"] = frame["date"].dt.dayofyear.astype("int16")
    frame["weekend_flag"] = (frame["day_of_week"] >= 5).astype("bool")

    fixed = _FIXED_HOLIDAYS_IN if geography == "IN" else _FIXED_HOLIDAYS_GLOBAL
    month_day = list(zip(frame["month"], frame["day"], strict=True))
    holiday_names = [fixed.get((int(m), int(d))) for m, d in month_day]
    frame["holiday_name"] = pd.Series(holiday_names, dtype="string")
    frame["holiday_flag"] = frame["holiday_name"].notna()

    frame["festival_flag"] = False
    frame["festival_name"] = pd.Series([None] * len(frame), dtype="string")
    if geography == "IN":
        date_only = frame["date"].dt.date
        for name, occurrences in _FESTIVALS_IN.items():
            for occurrence in occurrences:
                lead = pd.Timestamp(occurrence) - pd.Timedelta(days=festival_lead_days)
                mask = (frame["date"] >= lead) & (date_only <= occurrence)
                frame.loc[mask, "festival_flag"] = True
                frame.loc[mask, "festival_name"] = name

    if geography == "IN":
        frame["season"] = frame["month"].map(_SEASON_BY_MONTH_IN).astype("string")
        # Indian financial year: April = month 1.
        frame["financial_month"] = (((frame["month"] - 4) % 12) + 1).astype("int8")
        frame["financial_year"] = np.where(
            frame["month"] >= 4, frame["year"], frame["year"] - 1
        ).astype("int16")
    else:
        frame["season"] = pd.cut(
            frame["month"],
            bins=[0, 2, 5, 8, 11, 12],
            labels=["Winter", "Spring", "Summer", "Autumn", "Winter"],
            ordered=False,
        ).astype("string")
        frame["financial_month"] = frame["month"]
        frame["financial_year"] = frame["year"]

    frame["financial_quarter"] = (((frame["financial_month"] - 1) // 3) + 1).astype("int8")
    frame["date"] = frame["date"].dt.date

    return frame


def annual_seasonality(
    day_of_year: np.ndarray,
    amplitude: float,
    peak_month: int,
) -> np.ndarray:
    """Smooth annual seasonal factor in log space.

    A single sine harmonic rather than per-month dummies: it is smooth (no
    artificial discontinuity on the 1st of a month), fully described by two
    parameters, and still recoverable by a model using month features or Fourier
    terms. Returns a log-space additive term.
    """
    peak_day = (peak_month - 1) * 30.44 + 15.0
    phase = 2.0 * np.pi * (day_of_year - peak_day) / 365.25
    return amplitude * np.cos(phase)
