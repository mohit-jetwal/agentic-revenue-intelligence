"""Prediction intervals for forecasts, which need two things a nowcast did not.

Step 4's :mod:`ml.baseline.conformal` calibrates a single scalar quantile of
scaled residuals. That is right for a same-day baseline and wrong here, twice:

**1. Width must grow with horizon.** Tomorrow's demand is far more predictable
than demand in three months. A single quantile produces a width proportional to
the prediction alone, and predictions at h=90 are not systematically larger than
at h=1 - so the interval barely widens. It over-covers short horizons,
under-covers long ones, and reports one blended coverage figure that looks
healthy while being wrong at both ends. The fix is Mondrian conformal: calibrate
independently within each horizon bucket, so each keeps its own finite-sample
guarantee.

**2. The horizon total needs its own calibration.** ``total_predicted_units`` is
the number an agent actually acts on, and its interval cannot be obtained by
summing the daily bounds. Doing so assumes the daily errors are perfectly
correlated; if they were independent the sum would be too wide by roughly
``sqrt(90) ~ 9.5x``. The truth is in between and is not knowable a priori - so
the aggregate is calibrated directly on aggregate residuals instead of derived
from an assumption.

Both compose :func:`ml.baseline.conformal.calibrate` rather than reimplementing
it, so the finite-sample correction and the scaling floor stay in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.conformal import ConformalCalibration, CoverageReport, calibrate
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import HORIZON_STEP, ORIGIN_DATE

logger = get_logger(__name__)

#: Minimum usable points before a bucket gets its own quantile. Below this the
#: empirical quantile is noise, and a noisy interval is worse than a wider
#: pooled one because it claims a precision it does not have.
_MIN_BUCKET_POINTS = 100


@dataclass
class HorizonCalibration:
    """Per-bucket daily intervals plus a separately calibrated aggregate."""

    alpha: float
    #: One calibration per horizon bucket, keyed by bucket label.
    buckets: dict[str, ConformalCalibration] = field(default_factory=dict)
    #: Fallback for buckets with too few calibration points.
    pooled: ConformalCalibration | None = None
    #: Calibrated on *aggregate* residuals, for the horizon total.
    aggregate: ConformalCalibration | None = None

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def for_step(self, horizon_step: int, config: ForecastConfig) -> ConformalCalibration | None:
        """The calibration governing one horizon step."""
        label = config.intervals.bucket_for(int(horizon_step))
        return self.buckets.get(label) or self.pooled

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "buckets": {label: c.to_dict() for label, c in self.buckets.items()},
            "pooled": self.pooled.to_dict() if self.pooled else None,
            "aggregate": self.aggregate.to_dict() if self.aggregate else None,
        }

    def summary(self) -> str:
        parts = [
            f"{label}: q={c.quantile:.3f} (n={c.n_calibration:,})"
            for label, c in sorted(self.buckets.items())
        ]
        aggregate = (
            f" | aggregate q={self.aggregate.quantile:.3f}" if self.aggregate else ""
        )
        return f"{self.nominal_coverage:.0%} interval - " + "; ".join(parts) + aggregate


def calibrate_by_horizon(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    actual_column: str,
    predicted_column: str,
) -> HorizonCalibration:
    """Calibrate one quantile per horizon bucket, plus one for the total.

    Calibration origins are thinned to at least
    ``calibration_origin_spacing_days`` apart. Rows sharing an origin differ only
    in horizon step and see identical history, so their residuals are strongly
    correlated; counting them as independent inflates the effective sample size
    and makes the finite-sample correction optimistic. Thinning trades raw count
    for something closer to the exchangeability the guarantee assumes.
    """
    alpha = config.intervals.alpha
    if frame.empty:
        return HorizonCalibration(alpha=alpha)

    working = _thin_origins(frame, config)
    working = working[
        np.isfinite(pd.to_numeric(working[actual_column], errors="coerce"))
        & np.isfinite(pd.to_numeric(working[predicted_column], errors="coerce"))
    ]
    if working.empty:
        return HorizonCalibration(alpha=alpha)

    calibration = HorizonCalibration(alpha=alpha)

    # Pooled fallback first, so a thin bucket always has something valid to use.
    try:
        calibration.pooled = calibrate(
            working[actual_column], working[predicted_column], alpha=alpha
        )
    except ValueError:
        logger.warning("forecast.calibration_skipped", reason="too few points overall")
        return calibration

    working = working.assign(
        _bucket=working[HORIZON_STEP].map(config.intervals.bucket_for)
    )
    for label in config.intervals.bucket_labels():
        block = working[working["_bucket"] == label]
        if len(block) < _MIN_BUCKET_POINTS:
            logger.info(
                "forecast.bucket_pooled",
                bucket=label,
                points=len(block),
                reason="below the minimum for an independent quantile",
            )
            continue
        calibration.buckets[label] = calibrate(
            block[actual_column], block[predicted_column], alpha=alpha
        )

    calibration.aggregate = _calibrate_aggregate(
        working, alpha=alpha, actual_column=actual_column, predicted_column=predicted_column
    )

    logger.info(
        "forecast.horizon_calibrated",
        buckets=len(calibration.buckets),
        pooled=calibration.pooled is not None,
        aggregate=calibration.aggregate is not None,
    )
    return calibration


def _thin_origins(frame: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Keep origins at least N days apart, for approximate independence."""
    spacing = config.intervals.calibration_origin_spacing_days
    if spacing <= 0 or ORIGIN_DATE not in frame.columns:
        return frame

    origins = np.sort(pd.to_datetime(frame[ORIGIN_DATE]).unique())
    kept: list[pd.Timestamp] = []
    for origin in origins:
        stamp = pd.Timestamp(origin)
        if not kept or (stamp - kept[-1]).days >= spacing:
            kept.append(stamp)

    return frame[pd.to_datetime(frame[ORIGIN_DATE]).isin(kept)]


def _calibrate_aggregate(
    frame: pd.DataFrame,
    *,
    alpha: float,
    actual_column: str,
    predicted_column: str,
) -> ConformalCalibration | None:
    """Calibrate on per-(series, origin) totals rather than daily rows.

    The residual scored here is ``|sum(y) - sum(yhat)| / max(sum(yhat), 1)`` over
    one origin's horizon - which is exactly the quantity the reported
    ``total_predicted_units`` interval needs to cover. Deriving it from the daily
    quantile instead would require assuming a correlation structure between
    daily errors that nobody has measured.
    """
    keys = [c for c in ("product_id", "store_id", ORIGIN_DATE) if c in frame.columns]
    if not keys:
        return None

    totals = frame.groupby(keys, observed=True).agg(
        actual=(actual_column, "sum"), predicted=(predicted_column, "sum")
    )
    if len(totals) < _MIN_BUCKET_POINTS:
        logger.info("forecast.aggregate_calibration_skipped", blocks=len(totals))
        return None

    return calibrate(totals["actual"], totals["predicted"], alpha=alpha)


def add_horizon_intervals(
    frame: pd.DataFrame,
    calibration: HorizonCalibration,
    config: ForecastConfig,
    *,
    predicted_column: str = "predicted_units",
) -> pd.DataFrame:
    """Attach per-row bounds using each row's own bucket calibration."""
    if frame.empty or HORIZON_STEP not in frame.columns:
        return frame

    result = frame.copy()
    predictions = pd.to_numeric(result[predicted_column], errors="coerce").to_numpy(dtype=float)
    widths = np.zeros(len(result), dtype=float)

    for index, step in enumerate(result[HORIZON_STEP].to_numpy()):
        bucket = calibration.for_step(int(step), config)
        if bucket is None:
            continue
        scale = max(predictions[index], 1.0) if bucket.scaled else 1.0
        widths[index] = bucket.quantile * scale

    result["lower_bound"] = np.clip(predictions - widths, 0.0, None)
    result["upper_bound"] = predictions + widths
    return result


def aggregate_interval(
    total: float, calibration: HorizonCalibration
) -> tuple[float | None, float | None]:
    """Bounds for a horizon total, from the aggregate calibration.

    Returns ``(None, None)`` when no aggregate calibration exists rather than
    falling back to summing daily bounds. An interval derived from the wrong
    correlation assumption is not a conservative interval - it is a wrong one
    that looks authoritative, and the caller is better served knowing it is
    absent.
    """
    if calibration.aggregate is None:
        return None, None

    scale = max(total, 1.0) if calibration.aggregate.scaled else 1.0
    width = calibration.aggregate.quantile * scale
    return max(total - width, 0.0), total + width


def measure_horizon_coverage(
    frame: pd.DataFrame,
    calibration: HorizonCalibration,
    config: ForecastConfig,
    *,
    actual_column: str,
    predicted_column: str = "predicted_units",
) -> dict[str, CoverageReport]:
    """Achieved coverage per bucket, measured on held-out data.

    Reported per bucket and never blended. A blended figure can sit at a healthy
    90% while the short horizons cover 98% and the long ones cover 71% - and it
    is the long ones a planner is least able to sanity-check for themselves.
    """
    if frame.empty:
        return {}

    with_intervals = add_horizon_intervals(
        frame, calibration, config, predicted_column=predicted_column
    )
    with_intervals = with_intervals.assign(
        _bucket=with_intervals[HORIZON_STEP].map(config.intervals.bucket_for)
    )

    reports: dict[str, CoverageReport] = {}
    for label in config.intervals.bucket_labels():
        block = with_intervals[with_intervals["_bucket"] == label]
        if block.empty:
            continue

        actual = pd.to_numeric(block[actual_column], errors="coerce").to_numpy(dtype=float)
        lower = block["lower_bound"].to_numpy(dtype=float)
        upper = block["upper_bound"].to_numpy(dtype=float)
        usable = np.isfinite(actual)
        if not usable.any():
            continue

        covered = (actual[usable] >= lower[usable]) & (actual[usable] <= upper[usable])
        widths = upper[usable] - lower[usable]
        mean_actual = float(np.mean(actual[usable])) or 1.0

        reports[label] = CoverageReport(
            nominal=calibration.nominal_coverage,
            empirical=float(np.mean(covered)),
            n=int(usable.sum()),
            mean_width=float(np.mean(widths)),
            relative_width=float(np.mean(widths) / mean_actual),
        )

    return reports
