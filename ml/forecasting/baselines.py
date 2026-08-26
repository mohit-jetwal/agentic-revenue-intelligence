"""Naive benchmarks, and why Step 4's seasonal naive cannot be one of them.

These are not filler. Forecast Value Added is defined *against* them, so if the
benchmark is weak every model looks good and the comparison table becomes
flattery. Section 45 is explicit that a LightGBM which loses to a seasonal naive
must be reported as such - that only means anything if the naive is a fair fight.

**``SeasonalNaiveBaseline`` from Step 4 is not reusable here, and quietly reusing
it would be the easy mistake.** Two reasons:

1. Its ``lag_364_units`` is measured at the **origin**, so for a row at horizon
   ``h`` it reads sales from ``origin - 364``, which is ``364 + h`` days before
   the date being forecast. Wrong weekday, wrong point in the season, and
   progressively more wrong as ``h`` grows.
2. Its fallback chain includes ``lag_1_units`` - yesterday's sales relative to
   the origin. That is the illegal nowcast feature this entire step exists to
   avoid.

The correct seasonal benchmark reads the same weekday one year before the
**target** date, which is knowable at the origin for any ``h <= 364``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.models import BaselineEstimator

logger = get_logger(__name__)

#: 364, not 365: a multiple of 7, so "same day last year" is the same weekday.
#: Demand is far more weekly than annual, so an off-by-one-day comparison
#: against a different weekday is worse than useless.
SEASONAL_LAG_DAYS = 364


class HorizonNaive(BaselineEstimator):
    """Carry the last observed value forward: ``y_hat(t+h) = units(t)``.

    The "no forecasting process at all" rung. Deliberately weak, and it flatters
    anything compared against it - which is why FVA is reported against the
    seasonal benchmark as well, and why the seasonal one is the headline.

    Reported anyway because it establishes the floor: a model that cannot beat
    *this* has learned nothing whatsoever.
    """

    name = "horizon_naive"

    #: Ordered fallbacks. `lag_1_units` at the origin is the most recent
    #: observation available, and is legitimate here precisely because it is
    #: read at the origin and used as a *constant* across the horizon - not as a
    #: feature describing the target date.
    FALLBACK_CHAIN = ("lag_1_units", "rolling_7_units", "rolling_28_units")

    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._global_mean = 0.0

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
    ) -> None:
        # The only thing to learn: a last-resort level for series with no history
        # at all.
        self._global_mean = float(y.mean()) if len(y) else 0.0

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return _first_available(X, self.FALLBACK_CHAIN, default=self._global_mean)


class HorizonSeasonalNaive(BaselineEstimator):
    """Same weekday one year before the **target** date.

    The benchmark that matters, because it is roughly what a planner does
    unaided: look at what happened this week last year. Beating it is the
    minimum bar for a model to justify its own existence, and Forecast Value
    Added is defined against it.

    Requires a ``seasonal_reference`` column, attached by
    :func:`attach_seasonal_reference` from the historical panel. That column is
    computed from ``target_date - 364``, which is knowable at the origin for
    every horizon this model serves.
    """

    name = "horizon_seasonal_naive"

    #: Weight on the seasonal reference when blending with the recent level.
    #: Pure seasonal naive is very noisy - it inherits the full random error of
    #: one day a year ago - so blending with a rolling mean is both standard and
    #: a materially stronger benchmark.
    SEASONAL_WEIGHT = 0.5

    FALLBACK_CHAIN = ("rolling_28_units", "rolling_7_units", "lag_1_units")

    def __init__(self, *, seed: int = 42, seasonal_weight: float | None = None) -> None:
        super().__init__(seed=seed)
        self.seasonal_weight = (
            seasonal_weight if seasonal_weight is not None else self.SEASONAL_WEIGHT
        )
        self._global_mean = 0.0

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
    ) -> None:
        self._global_mean = float(y.mean()) if len(y) else 0.0

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        recent = _first_available(X, self.FALLBACK_CHAIN, default=self._global_mean)

        if "seasonal_reference" not in X.columns:
            # No seasonal column attached - degrade to the recent level rather
            # than failing. A benchmark that raises is a benchmark nobody runs.
            return recent

        seasonal = pd.to_numeric(X["seasonal_reference"], errors="coerce").to_numpy(dtype=float)
        available = np.isfinite(seasonal)

        blended = recent.copy()
        blended[available] = (
            self.seasonal_weight * seasonal[available]
            + (1.0 - self.seasonal_weight) * recent[available]
        )
        return blended

    def get_params(self) -> dict[str, object]:
        return {"seed": self.seed, "seasonal_weight": self.seasonal_weight}


def _first_available(
    X: pd.DataFrame, columns: tuple[str, ...], *, default: float
) -> np.ndarray:
    """Take the first column that has a value, falling back down the chain.

    A cold-start series has no lags at all; returning NaN would drop those rows
    from every metric silently, which is how a benchmark comes to look better
    than it is.
    """
    result = np.full(len(X), np.nan, dtype=float)
    for column in columns:
        if column not in X.columns:
            continue
        values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(result)
        result[missing] = values[missing]
        if np.isfinite(result).all():
            break

    result[~np.isfinite(result)] = default
    return result


def attach_seasonal_reference(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    *,
    target: str = "units",
    target_date_column: str = "target_date",
    keys: tuple[str, ...] = ("product_id", "store_id"),
) -> pd.DataFrame:
    """Attach units from ``target_date - 364`` for the seasonal benchmark.

    Sourced from the *target* date rather than the origin, which is the whole
    correction over Step 4's estimator. Legitimate at forecast time for any
    horizon under 364 days: the reference date is more than a year in the past,
    so it is firmly inside observed history.
    """
    if frame.empty or history.empty:
        return frame

    reference = history[[*keys, "date", target]].copy()
    reference["date"] = pd.to_datetime(reference["date"])
    # Shift the lookup date forward a year, so joining on the target date
    # retrieves the value from a year before it.
    reference[target_date_column] = reference["date"] + pd.Timedelta(days=SEASONAL_LAG_DAYS)
    reference = reference.drop(columns="date").rename(columns={target: "seasonal_reference"})

    result = frame.copy()
    result[target_date_column] = pd.to_datetime(result[target_date_column])
    merged = result.merge(reference, on=[*keys, target_date_column], how="left")

    coverage = float(merged["seasonal_reference"].notna().mean()) if len(merged) else 0.0
    logger.info(
        "forecast.seasonal_reference_attached",
        rows=len(merged),
        coverage=round(coverage, 4),
    )
    return merged
