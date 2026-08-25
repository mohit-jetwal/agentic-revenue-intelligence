"""Demand features: lags, rolling statistics, trend (brief sections 14-15).

The features every temporal model in Steps 4-11 starts from. All of them route
through :mod:`features.engineering.panel`, so the shift discipline that keeps
them point-in-time correct lives in one place rather than being re-derived here.

A note on which column to lag. The panel's ``units`` is *observed* sales, which
during a stockout is supply rather than demand. Lagging it propagates the
censoring forward, so a stockout last week depresses this week's features and
the model learns that a supply failure predicts low demand. That is exactly
backwards. Callers who care - the baseline model in Step 4 especially - should
consider masking or imputing stockout days before lagging; the helper
:func:`mask_censored` exists for that, and the decision is left to the model
because it is a modelling choice, not a data one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from features.engineering.panel import (
    PANEL_KEYS,
    prepare_panel,
    rolling_on_shifted,
    shifted_group,
)

#: Lags from brief section 14. 364 rather than 365 so the same-week-last-year
#: comparison lands on the same day of week - retail demand is far more
#: sensitive to weekday than to calendar date, and a 365-day lag would compare
#: a Saturday to a Friday.
DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28, 56, 364)

#: Rolling windows from brief section 15.
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 28, 56)


def add_lag_features(
    panel: pd.DataFrame,
    *,
    column: str = "units",
    lags: Sequence[int] = DEFAULT_LAGS,
    keys: Sequence[str] = PANEL_KEYS,
    prefix: str = "lag",
) -> pd.DataFrame:
    """Add ``lag_{n}_{column}`` for each requested lag.

    Computed within each product-store series, so the tail of one product never
    leaks into the head of the next.
    """
    result = panel.copy()
    for lag in lags:
        result[f"{prefix}_{lag}_{column}"] = shifted_group(result, column, periods=lag, keys=keys)
    return result


def add_rolling_features(
    panel: pd.DataFrame,
    *,
    column: str = "units",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    statistics: Sequence[str] = ("mean",),
    keys: Sequence[str] = PANEL_KEYS,
    prefix: str = "rolling",
) -> pd.DataFrame:
    """Add ``rolling_{window}_{column}`` over the window ending *yesterday*.

    Excludes the current row by construction - see
    :func:`~features.engineering.panel.rolling_on_shifted`.
    """
    result = panel.copy()
    for window in windows:
        for statistic in statistics:
            suffix = "" if statistic == "mean" else f"_{statistic}"
            result[f"{prefix}_{window}_{column}{suffix}"] = rolling_on_shifted(
                result, column, window=window, statistic=statistic, keys=keys
            )
    return result


def add_demand_dynamics(
    panel: pd.DataFrame,
    *,
    column: str = "units",
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.DataFrame:
    """Derived demand signals: momentum, volatility and trend.

    All built from already-shifted rolling statistics, so nothing here reaches
    into the present.
    """
    result = panel.copy()

    short = rolling_on_shifted(result, column, window=7, statistic="mean", keys=keys)
    long = rolling_on_shifted(result, column, window=28, statistic="mean", keys=keys)
    volatility = rolling_on_shifted(result, column, window=28, statistic="std", keys=keys)

    # Momentum: recent demand relative to the established base. Above 1 means
    # accelerating. A ratio rather than a difference so it is comparable across
    # a hero SKU and a slow mover.
    result["demand_momentum"] = short / long.replace(0.0, np.nan)

    # Coefficient of variation - how erratic this series is. A forecasting model
    # can use it to widen intervals where the history is noisy.
    result["demand_volatility"] = volatility / long.replace(0.0, np.nan)

    # Whether the series has been growing over the last four weeks, measured
    # between two non-overlapping shifted windows.
    four_weeks_ago = shifted_group(result, column, periods=28, keys=keys)
    recent = rolling_on_shifted(result, column, window=7, statistic="mean", keys=keys)
    result["demand_trend_28"] = (recent - four_weeks_ago) / four_weeks_ago.replace(0.0, np.nan)

    return result


def mask_censored(
    panel: pd.DataFrame,
    *,
    column: str = "units",
    stockout_column: str = "stockout_flag",
) -> pd.DataFrame:
    """Blank observed sales on stockout days, producing ``{column}_uncensored``.

    Observed sales during a stockout measure *availability*, not demand.
    Feeding them into lags teaches a model that supply failures predict weak
    demand, which inverts the relationship the Root Cause agent needs.

    The column is added rather than substituted, so a caller can compare
    censored and uncensored features and decide. Step 4's baseline model is the
    natural consumer.
    """
    if stockout_column not in panel.columns:
        raise KeyError(
            f"{stockout_column!r} is required to mask censored demand; "
            f"request it from the repository or skip this step"
        )
    result = panel.copy()
    result[f"{column}_uncensored"] = result[column].where(~result[stockout_column].astype(bool))
    return result


def build_demand_features(
    panel: pd.DataFrame,
    *,
    column: str = "units",
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    keys: Sequence[str] = PANEL_KEYS,
    include_dynamics: bool = True,
) -> pd.DataFrame:
    """Full demand feature block: lags, rolling means and dynamics."""
    result = prepare_panel(panel, keys=keys)
    result = add_lag_features(result, column=column, lags=lags, keys=keys)
    result = add_rolling_features(
        result, column=column, windows=windows, statistics=("mean", "std"), keys=keys
    )
    if include_dynamics:
        result = add_demand_dynamics(result, column=column, keys=keys)
    return result
