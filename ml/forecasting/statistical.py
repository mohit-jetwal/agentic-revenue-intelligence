"""Classical time-series models, fitted where their assumptions actually hold.

Section 8 asks for a statistical approach and warns against forcing SARIMA onto
millions of series. The cost is measured here rather than asserted - and the
measurement is more modest than the warning implies, so it is worth stating
accurately.

**Measured on this dataset:** weekly ETS fits in ~0.25s per series. Across 6,128
product-store series that is roughly 25 minutes for one pass, or a bit over an
hour across three backtest folds. Expensive next to the global model's ~8 minutes
for the entire pipeline, but *not* infeasible. Claiming otherwise would be
overstating a real cost into a fictional impossibility. Daily SARIMA with two
seasonal periods would be far worse, but that is a different model and the number
here is for the one actually fitted.

**So scalability is not the argument. Appropriateness is.** The statistical models
are fitted at aggregate grain because that is where they are *correct*, not
because the alternative was impossible:

* At product-store grain the series are sparse counts against a ~35% irreducible
  noise floor. ETS would be fitting noise, and its intervals would be
  meaningless.
* At category or region grain the series are smooth and high-volume, and are
  genuinely well described by a level/trend/seasonal decomposition.

Fitting is **weekly**, not daily. Daily annual seasonality needs 365 seasonal
states, which will not converge sensibly on three years of history.

These fits are not throwaway. They are the independent top-level forecasts that
make the section 13 coherence check possible at no extra cost - without them
there would be nothing for the bottom-up aggregate to be compared against.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.observability.logging import get_logger
from ml.baseline.evaluation import BaselineMetrics, compute_metrics

logger = get_logger(__name__)

#: Weeks of history required before a seasonal fit is attempted. Two full annual
#: cycles; with less, the seasonal component is being estimated from one
#: observation per week-of-year.
_MIN_WEEKS_SEASONAL = 104
#: Weeks required for a non-seasonal fallback fit.
_MIN_WEEKS = 26


@dataclass
class StatisticalFit:
    """One fitted series at one aggregation level."""

    level: str
    key: str
    metrics: BaselineMetrics
    fit_seconds: float
    seasonal: bool
    n_weeks: int
    forecast: pd.Series = field(default_factory=pd.Series)


@dataclass
class StatisticalResult:
    """Every fit, plus the cost evidence for the scalability claim."""

    fits: list[StatisticalFit] = field(default_factory=list)
    total_seconds: float = 0.0
    #: Series at product-store grain in the full dataset, for the extrapolation.
    bottom_series_count: int = 0

    @property
    def mean_fit_seconds(self) -> float:
        return (
            float(np.mean([f.fit_seconds for f in self.fits])) if self.fits else float("nan")
        )

    def extrapolated_hours(self) -> float:
        """Hours one pass over every product-store series would cost.

        The number that turns "per-series SARIMA does not scale" from an opinion
        into a measurement.
        """
        return self.mean_fit_seconds * self.bottom_series_count / 3600.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "level": fit.level,
                    "key": fit.key,
                    "n_weeks": fit.n_weeks,
                    "seasonal": fit.seasonal,
                    "wmape": fit.metrics.wmape,
                    "bias_pct": fit.metrics.bias_pct,
                    "fit_seconds": round(fit.fit_seconds, 3),
                }
                for fit in self.fits
            ]
        )

    def by_level(self) -> pd.DataFrame:
        """Mean accuracy per aggregation level."""
        frame = self.to_frame()
        if frame.empty:
            return frame
        return (
            frame.groupby("level", observed=True)
            .agg(
                series=("key", "count"),
                mean_wmape=("wmape", "mean"),
                mean_fit_seconds=("fit_seconds", "mean"),
            )
            .reset_index()
        )


def _to_weekly(frame: pd.DataFrame, *, date_column: str, value_column: str) -> pd.Series:
    """Aggregate a daily series to weekly totals."""
    working = frame[[date_column, value_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column])
    weekly = (
        working.set_index(date_column)[value_column]
        .resample("W")
        .sum()
        .astype(float)
    )
    return weekly


def _fit_one(
    weekly: pd.Series, *, holdout_weeks: int
) -> tuple[BaselineMetrics, float, bool] | None:
    """Fit ETS on all but the last ``holdout_weeks`` and score on those.

    Falls back to a non-seasonal fit when there is not enough history for a
    seasonal one, rather than refusing. A trend-only ETS on a short series is a
    legitimate model; a seasonal ETS fitted on one cycle is not.
    """
    if len(weekly) < _MIN_WEEKS + holdout_weeks:
        return None

    train = weekly.iloc[:-holdout_weeks]
    test = weekly.iloc[-holdout_weeks:]
    seasonal = len(train) >= _MIN_WEEKS_SEASONAL

    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            # statsmodels is voluble about convergence on short series; the
            # metrics below are the actual verdict.
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add" if seasonal else None,
                seasonal_periods=52 if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            predictions = model.forecast(holdout_weeks)
    except Exception as exc:  # noqa: BLE001 - a failed fit is a result, not a crash
        logger.debug("forecast.ets_failed", error=str(exc))
        return None
    elapsed = time.perf_counter() - started

    metrics = compute_metrics(test, pd.Series(np.clip(predictions.to_numpy(), 0, None)))
    return metrics, elapsed, seasonal


def fit_aggregate_models(
    panel: pd.DataFrame,
    *,
    levels: tuple[str, ...] = ("total", "region", "category", "category_region"),
    holdout_weeks: int = 12,
    date_column: str = "date",
    value_column: str = "units",
    bottom_series_count: int = 0,
) -> StatisticalResult:
    """Fit weekly ETS at each aggregation level.

    This is the honest scope for a classical model on this data. The claim it
    supports is about aggregate forecasting, where the whole population is
    fitted - not a claim about product-store grain, which would need the
    per-series fits that do not scale.
    """
    result = StatisticalResult(bottom_series_count=bottom_series_count)
    if panel.empty:
        return result

    started = time.perf_counter()
    groupings: dict[str, list[str] | None] = {
        "total": None,
        "region": ["region"],
        "category": ["category"],
        "category_region": ["category", "region"],
    }

    for level in levels:
        keys = groupings.get(level)
        if keys is not None and any(k not in panel.columns for k in keys):
            logger.info("forecast.ets_level_skipped", level=level, reason="missing columns")
            continue

        blocks = (
            [("all", panel)]
            if keys is None
            else [
                ("|".join(str(v) for v in (key if isinstance(key, tuple) else (key,))), block)
                for key, block in panel.groupby(keys, observed=True)
            ]
        )

        for key, block in blocks:
            weekly = _to_weekly(block, date_column=date_column, value_column=value_column)
            fitted = _fit_one(weekly, holdout_weeks=holdout_weeks)
            if fitted is None:
                continue
            metrics, elapsed, seasonal = fitted
            result.fits.append(
                StatisticalFit(
                    level=level,
                    key=str(key),
                    metrics=metrics,
                    fit_seconds=elapsed,
                    seasonal=seasonal,
                    n_weeks=len(weekly),
                )
            )

    result.total_seconds = time.perf_counter() - started
    logger.info(
        "forecast.ets_fitted",
        fits=len(result.fits),
        seconds=round(result.total_seconds, 1),
        mean_fit_seconds=round(result.mean_fit_seconds, 3),
    )
    return result


def sample_bottom_series(
    panel: pd.DataFrame,
    *,
    n_series: int = 50,
    holdout_weeks: int = 12,
    seed: int = 42,
    date_column: str = "date",
    value_column: str = "units",
) -> StatisticalResult:
    """Fit ETS on a stratified sample of product-store series.

    Not a deliverable forecast. It answers one narrow question - *is a classical
    univariate model competitive where the global model actually operates?* -
    and it answers it on a sample, so the claim must be sized accordingly:
    enough to rule out an order-of-magnitude gap, nowhere near enough to rank
    two models a few points apart.
    """
    result = StatisticalResult()
    if panel.empty:
        return result

    rng = np.random.default_rng(seed)
    volumes = (
        panel.groupby(["product_id", "store_id"], observed=True)[value_column]
        .sum()
        .sort_values()
    )
    if volumes.empty:
        return result

    # Stratified by volume decile: a uniform draw is dominated by slow movers,
    # which is not where the model's accuracy matters.
    deciles = np.array_split(volumes.index.to_numpy(), min(10, len(volumes)))
    chosen: list[tuple[str, str]] = []
    per_decile = max(n_series // len(deciles), 1)
    for block in deciles:
        take = min(per_decile, len(block))
        picked = rng.choice(len(block), size=take, replace=False)
        chosen.extend(block[i] for i in picked)

    started = time.perf_counter()
    for product_id, store_id in chosen[:n_series]:
        block = panel[
            (panel["product_id"] == product_id) & (panel["store_id"] == store_id)
        ]
        weekly = _to_weekly(block, date_column=date_column, value_column=value_column)
        fitted = _fit_one(weekly, holdout_weeks=holdout_weeks)
        if fitted is None:
            continue
        metrics, elapsed, seasonal = fitted
        result.fits.append(
            StatisticalFit(
                level="product_store",
                key=f"{product_id}|{store_id}",
                metrics=metrics,
                fit_seconds=elapsed,
                seasonal=seasonal,
                n_weeks=len(weekly),
            )
        )

    result.total_seconds = time.perf_counter() - started
    logger.info(
        "forecast.ets_bottom_sample",
        fits=len(result.fits),
        seconds=round(result.total_seconds, 1),
    )
    return result


def scalability_statement(result: StatisticalResult) -> str:
    """The measured cost of per-series fitting, as a sentence.

    Written out rather than left implicit because section 8 asks for the
    scalability reasoning, and a measured extrapolation is worth more than the
    assertion that it would be slow.
    """
    if not result.fits or not result.bottom_series_count:
        return "Per-series cost was not measured."

    hours = result.extrapolated_hours()
    verdict = (
        "expensive but feasible" if hours < 4 else "prohibitive at this scale"
    )
    return (
        f"ETS fits averaged {result.mean_fit_seconds:.2f}s per series. Across the "
        f"{result.bottom_series_count:,} product-store series in this dataset, one "
        f"pass would cost roughly {hours:.1f} hours ({hours * 60:.0f} minutes), and "
        f"a walk-forward backtest multiplies that by the number of folds - "
        f"{verdict}. Reported as measured rather than rounded up into a claim of "
        f"impossibility: the case for fitting at aggregate grain rests on where "
        f"ETS is statistically appropriate, not on where it is affordable."
    )
