"""Producing an actual forecast, and refusing to produce a fake one.

Two things live here that have no counterpart in Step 4.

**The as-of validation.** A forecast needs target-side features - the calendar,
the promotion schedule, the price plan - for every day in the horizon. Those
tables are ``KNOWN_IN_ADVANCE``, so reading them forward is legitimate; but in
this dataset they stop on 2025-12-31 like everything else. Asking for 90 days
from 2025-12-01 means 60 days with no planned promotion data at all.

The tempting answer is to assume no promotion runs and carry the last price
forward. That is a fabrication with a predictable direction: promotions raise
demand, so those days come back systematically low, and the number looks like a
forecast rather than a guess. So the request is **refused**, with a recoverable
error naming the latest as-of that would work. Section 45 asks for a credible
system over an impressive one, and a forecast nobody can distinguish from an
assumption is not credible.

**The fallback chain.** When the primary model cannot produce a value for a
series - no history, a cold-start listing - the service degrades to the seasonal
naive and then to a category mean, and says so. ``fallback_used`` and
``fallback_reason`` are part of the output because a caller must be able to tell
an estimate from a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.point_in_time import PointInTimeView
from ml.base import InsufficientDataError
from ml.forecasting.baselines import attach_seasonal_reference
from ml.forecasting.config import ForecastConfig
from ml.forecasting.conformal import HorizonCalibration, add_horizon_intervals, aggregate_interval
from ml.forecasting.dataset import (
    HORIZON_STEP,
    KEYS,
    build_future_scaffold,
    latest_known_date,
)

logger = get_logger(__name__)

PREDICTED = "predicted_units"


class SupportsPredict(Protocol):
    """Anything that turns a feature frame into predicted units.

    A protocol rather than a concrete type because both the trained candidate
    and the naive fallback are passed here, and they share nothing but this one
    method. Naming the contract is more useful than naming a union of two
    unrelated classes.
    """

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ForecastFrame:
    """Daily forecasts for one or more series, with provenance."""

    frame: pd.DataFrame
    as_of: date
    horizon_days: int
    fallback_rows: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def total_units(self) -> float:
        return float(self.frame[PREDICTED].sum()) if not self.frame.empty else 0.0

    def total_revenue(self, *, price_column: str = "h_selling_price") -> float | None:
        """Revenue at the **planned** price, or ``None`` when it is unknown.

        Deliberately not falling back to the last observed price. That would
        produce a confident revenue number resting on an assumption nobody
        stated, and the field is optional precisely so it can be absent.
        """
        if self.frame.empty or price_column not in self.frame.columns:
            return None
        price = pd.to_numeric(self.frame[price_column], errors="coerce")
        if price.isna().all():
            return None
        units = self.frame[PREDICTED]
        usable = price.notna()
        return float((units[usable] * price[usable]).sum())


def validate_as_of(
    view: PointInTimeView, as_of: date, horizon_days: int
) -> None:
    """Refuse a horizon that runs past the known-in-advance calendar.

    Raises :class:`~ml.base.InsufficientDataError`, which the service maps to a
    *recoverable* error: a different request - a shorter horizon, an earlier
    as-of - would succeed, so an agent can re-plan around it rather than giving
    up.
    """
    known_until = latest_known_date(view)
    horizon_end = as_of + timedelta(days=horizon_days)

    if horizon_end > known_until:
        latest_valid = known_until - timedelta(days=horizon_days)
        raise InsufficientDataError(
            f"a {horizon_days}-day horizon from {as_of} reaches {horizon_end}, but "
            f"the calendar, promotion schedule and price plan end {known_until}. "
            f"Forecasting past that would mean assuming no promotions are planned, "
            f"which biases those days low. The latest as-of that supports a "
            f"{horizon_days}-day horizon is {latest_valid}."
        )

    if as_of > known_until:
        raise InsufficientDataError(
            f"as_of {as_of} is past the end of available data ({known_until})"
        )


def latest_supported_as_of(view: PointInTimeView, horizon_days: int) -> date:
    """The most recent as-of for which a full horizon is available."""
    return latest_known_date(view) - timedelta(days=horizon_days)


def generate_forecast(
    view: PointInTimeView,
    history: pd.DataFrame,
    pairs: pd.DataFrame,
    predictor: SupportsPredict,
    config: ForecastConfig,
    *,
    as_of: date,
    horizon_days: int,
    calibration: HorizonCalibration | None = None,
    fallback: SupportsPredict | None = None,
) -> ForecastFrame:
    """Forecast every day from ``as_of + 1`` to ``as_of + horizon_days``.

    Validates first, builds the future scaffold, predicts, attaches intervals and
    applies the fallback chain to any row the primary model could not serve.
    """
    validate_as_of(view, as_of, horizon_days)

    scaffold = build_future_scaffold(
        view, history, pairs, as_of=as_of, horizon_days=horizon_days
    )
    if scaffold.empty:
        raise InsufficientDataError(
            f"no history for the requested series at or before {as_of}"
        )

    # The seasonal benchmark's reference column, needed by the fallback and by
    # the model when it is the seasonal naive itself.
    scaffold = attach_seasonal_reference(scaffold, history)

    predictions = np.asarray(predictor.predict(scaffold), dtype=float)
    scaffold[PREDICTED] = predictions

    fallback_rows = 0
    reasons: dict[str, int] = {}

    invalid = ~np.isfinite(predictions)
    if invalid.any():
        fallback_rows, reasons = _apply_fallback(scaffold, invalid, fallback, reasons)

    # Demand cannot be negative, and a negative forecast propagates into
    # nonsense downstream rather than failing loudly.
    scaffold[PREDICTED] = scaffold[PREDICTED].clip(lower=0.0)

    if calibration is not None:
        scaffold = add_horizon_intervals(
            scaffold, calibration, config, predicted_column=PREDICTED
        )

    scaffold = scaffold.sort_values([*KEYS, HORIZON_STEP]).reset_index(drop=True)
    logger.info(
        "forecast.generated",
        as_of=str(as_of),
        horizon_days=horizon_days,
        rows=len(scaffold),
        series=len(scaffold.groupby(list(KEYS), observed=True)),
        fallback_rows=fallback_rows,
    )
    return ForecastFrame(
        frame=scaffold,
        as_of=as_of,
        horizon_days=horizon_days,
        fallback_rows=fallback_rows,
        fallback_reasons=reasons,
    )


def _apply_fallback(
    scaffold: pd.DataFrame,
    invalid: np.ndarray,
    fallback: SupportsPredict | None,
    reasons: dict[str, int],
) -> tuple[int, dict[str, int]]:
    """Fill rows the primary model could not serve (brief section 28).

    Order: seasonal naive, then the series' own recent mean, then zero. Each step
    records why, because a silent fallback is indistinguishable from a
    successful prediction and that is precisely the confusion that erodes trust
    in a forecasting system.
    """
    count = int(invalid.sum())

    if fallback is not None:
        try:
            replacement = np.asarray(fallback.predict(scaffold[invalid]), dtype=float)
            scaffold.loc[invalid, PREDICTED] = replacement
            reasons["seasonal_naive"] = count
            still_invalid = ~np.isfinite(scaffold[PREDICTED].to_numpy())
        except Exception as exc:  # noqa: BLE001 - a failed fallback must not fail the call
            logger.warning("forecast.fallback_failed", error=str(exc))
            still_invalid = invalid
    else:
        still_invalid = invalid

    if still_invalid.any():
        # Last resort: the series' own recent level, then zero.
        for column in ("rolling_28_units", "rolling_7_units", "lag_1_units"):
            if column not in scaffold.columns:
                continue
            values = pd.to_numeric(scaffold[column], errors="coerce")
            usable = still_invalid & values.notna().to_numpy()
            if usable.any():
                scaffold.loc[usable, PREDICTED] = values[usable]
                reasons["recent_mean"] = reasons.get("recent_mean", 0) + int(usable.sum())
                still_invalid = still_invalid & ~usable

        if still_invalid.any():
            scaffold.loc[still_invalid, PREDICTED] = 0.0
            reasons["no_history"] = int(still_invalid.sum())

    return count, reasons


def summarise_series(
    forecast: ForecastFrame,
    *,
    calibration: HorizonCalibration | None = None,
    price_column: str = "h_selling_price",
) -> pd.DataFrame:
    """One row per series: horizon total, plus its own calibrated interval.

    The aggregate bound comes from the aggregate calibration, never from summing
    the daily bounds. Summing assumes the daily errors move together perfectly;
    if they were independent the sum would be too wide by roughly ``sqrt(90)``.
    Neither assumption is measured, so the total is calibrated directly instead.
    """
    if forecast.frame.empty:
        return pd.DataFrame()

    grouped = forecast.frame.groupby(list(KEYS), observed=True)
    rows = []
    for (product_id, store_id), block in grouped:
        total = float(block[PREDICTED].sum())
        lower, upper = (
            aggregate_interval(total, calibration) if calibration else (None, None)
        )
        price = (
            pd.to_numeric(block[price_column], errors="coerce")
            if price_column in block.columns
            else None
        )
        revenue = (
            float((block[PREDICTED] * price).sum())
            if price is not None and price.notna().any()
            else None
        )
        rows.append(
            {
                "product_id": product_id,
                "store_id": store_id,
                "total_predicted_units": total,
                "total_lower_bound": lower,
                "total_upper_bound": upper,
                "total_predicted_revenue": revenue,
                "days": len(block),
            }
        )

    return pd.DataFrame(rows)
