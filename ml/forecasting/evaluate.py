"""Forecast evaluation. Mostly composition of Step 4's metrics.

``ml.baseline.evaluation`` already computes MAE/RMSE/MAPE/WMAPE/Bias correctly,
handles the zero-actual problem for MAPE, and knows how to score against Step 2's
latent demand. None of that is rewritten here.

What is new is forecasting-specific and comes down to one idea: **a blended
number describes no decision anyone makes.** Forecast error grows with horizon by
nature, so a single WMAPE averaged over 1..90 days answers a question - "how
wrong are we, somewhere between tomorrow and three months?" - that nobody asks.
Everything below breaks results out by horizon bucket, aggregation level, or
benchmark comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import BaselineMetrics, compute_metrics
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import HORIZON_STEP, TARGET, TARGET_DATE

logger = get_logger(__name__)

PREDICTED = "predicted_units"


def metrics_by_horizon_bucket(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    actual_column: str = TARGET,
    predicted_column: str = PREDICTED,
) -> dict[str, BaselineMetrics]:
    """WMAPE and friends per horizon bucket."""
    if frame.empty or HORIZON_STEP not in frame.columns:
        return {}

    working = frame.assign(
        _bucket=frame[HORIZON_STEP].map(config.intervals.bucket_for)
    )
    results: dict[str, BaselineMetrics] = {}
    for label in config.intervals.bucket_labels():
        block = working[working["_bucket"] == label]
        if block.empty:
            continue
        results[label] = compute_metrics(block[actual_column], block[predicted_column])
    return results


def seasonal_naive_scale(
    train: pd.DataFrame,
    *,
    season_length: int = 7,
    actual_column: str = TARGET,
    date_column: str = TARGET_DATE,
    keys: tuple[str, ...] = ("product_id", "store_id"),
) -> float:
    """The MASE denominator: in-sample MAE of a seasonal naive.

    Computed on the **training fold only**, and that is the whole subtlety. MASE
    scales the model's error by the error a naive forecaster would have made *on
    data the model was fitted on*. Taking the denominator from the evaluation
    fold instead makes the metric partly self-referential - both numerator and
    denominator would then move with the same held-out noise, and a model could
    improve its MASE by getting worse in a period where the naive got worse
    faster.

    ``season_length=7`` because demand here is far more weekly than annual: the
    measured weekly peak-to-trough swing dominates the annual one, so the
    one-week-ago value is the natural "no effort" comparison. A yearly scale
    (364) would make almost any model look excellent, which is exactly the kind
    of flattering benchmark MASE exists to avoid.

    Returns ``nan`` when the scale cannot be computed, so callers report an
    absent MASE rather than dividing by an invented denominator.
    """
    if train.empty or actual_column not in train.columns:
        return float("nan")

    working = train[[*keys, date_column, actual_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column])
    working = working.sort_values([*keys, date_column])

    # Differences are taken within a series, never across the boundary between
    # two series - a cross-series difference is meaningless.
    previous = working.groupby(list(keys), observed=True)[actual_column].shift(season_length)
    differences = (working[actual_column] - previous).abs().dropna()

    if differences.empty:
        return float("nan")

    scale = float(differences.mean())
    return scale if scale > 0 else float("nan")


def mase(
    actual: pd.Series,
    predicted: pd.Series,
    scale: float,
) -> float:
    """Mean Absolute Scaled Error.

    ``MAE(model) / MAE(seasonal naive, in-sample)``. Below 1 means the model
    beats a naive forecaster; above 1 means it does not.

    Reported **alongside** WMAPE rather than instead of it, because the two
    answer different questions. WMAPE says how wrong the forecast is, weighted
    by volume - the question a planner holding inventory asks. MASE says whether
    the model is worth having at all, on a scale that is comparable across
    series with wildly different volumes. Neither substitutes for the other.
    """
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")

    y = pd.to_numeric(actual, errors="coerce")
    yhat = pd.to_numeric(predicted, errors="coerce")
    errors = (y - yhat).abs().dropna()
    if errors.empty:
        return float("nan")

    return float(errors.mean() / scale)


def horizon_error_grows(bucket_metrics: dict[str, BaselineMetrics]) -> bool:
    """Does error over long horizons exceed error over short ones?

    The strongest single diagnostic that the origin/target join is right. If
    forecasting 90 days out is as accurate as forecasting tomorrow, the model is
    not forecasting - it is reading information from the target date that it
    should not have, and every other metric will look excellent.

    Compares the **volume-weighted average over the short half** against the long
    half, rather than the single first and last buckets. Two reasons, both
    learned by watching this flip:

    * Individual buckets carry few rows and wobble. The endpoint comparison
      inverted between runs on noise alone, which would have made this a flaky
      test - and a flaky test gets weakened until it means nothing.
    * Strict monotonicity is not the claim being made. The claim is that
      forecasting further ahead is harder, which is a statement about the trend,
      not about every adjacent pair.
    """
    if len(bucket_metrics) < 2:
        return True

    buckets = list(bucket_metrics.values())
    midpoint = len(buckets) // 2
    short, long = buckets[:midpoint], buckets[midpoint:]

    def weighted(group: list[BaselineMetrics]) -> float:
        total = sum(m.n for m in group)
        if not total:
            return float("nan")
        return sum(m.wmape * m.n for m in group) / total

    return weighted(long) > weighted(short)


def fva_table(
    model_metrics: dict[str, BaselineMetrics],
    benchmark_metrics: dict[str, BaselineMetrics],
    *,
    model_name: str = "model",
    benchmark_name: str = "horizon_seasonal_naive",
) -> pd.DataFrame:
    """Forecast Value Added per horizon bucket, in WMAPE percentage points.

    ``FVA = WMAPE(benchmark) - WMAPE(model)``. Positive means the model beat what
    a planner would get unaided.

    Percentage points rather than a ratio: against a ~35% irreducible noise floor
    a ratio compresses every result into a narrow band, and a genuine four-point
    improvement reads as a rounding error.
    """
    rows = [
        {
            "bucket": bucket,
            f"{benchmark_name}_wmape": benchmark_metrics[bucket].wmape,
            f"{model_name}_wmape": metrics.wmape,
            "fva_pp": benchmark_metrics[bucket].wmape - metrics.wmape,
            "n": metrics.n,
        }
        for bucket, metrics in model_metrics.items()
        if bucket in benchmark_metrics
    ]
    return pd.DataFrame(rows)


def hierarchy_table(
    frame: pd.DataFrame,
    *,
    actual_column: str = TARGET,
    predicted_column: str = PREDICTED,
    levels: tuple[str, ...] = ("product_store", "product", "store", "region", "category", "total"),
) -> pd.DataFrame:
    """Accuracy at each level of the hierarchy, from one bottom-up forecast.

    Bottom-up aggregation is *exactly* coherent by construction - the regional
    number is the sum of its store numbers, always - so no reconciliation step is
    needed or performed.

    What this table shows is the price of that coherence. Aggregating averages
    out independent errors, so WMAPE falls sharply as you move up. That answers a
    question a planner actually asks - "should I trust the regional figure more
    than the SKU figure?" - with a magnitude rather than a shrug.
    """
    if frame.empty:
        return pd.DataFrame()

    groupings: dict[str, list[str]] = {
        "product_store": ["product_id", "store_id", TARGET_DATE],
        "product": ["product_id", TARGET_DATE],
        "store": ["store_id", TARGET_DATE],
        "region": ["region", TARGET_DATE],
        "category": ["category", TARGET_DATE],
        "total": [TARGET_DATE],
    }

    rows = []
    for level in levels:
        keys = groupings.get(level)
        if not keys or any(k not in frame.columns for k in keys):
            continue
        aggregated = frame.groupby(keys, observed=True).agg(
            actual=(actual_column, "sum"), predicted=(predicted_column, "sum")
        )
        metrics = compute_metrics(aggregated["actual"], aggregated["predicted"])
        rows.append(
            {
                "level": level,
                "series": len(aggregated),
                "wmape": metrics.wmape,
                "bias_pct": metrics.bias_pct,
                "mae": metrics.mae,
            }
        )

    return pd.DataFrame(rows)


def coherence_check(
    bottom_up: pd.DataFrame,
    independent: pd.DataFrame,
    *,
    level_column: str,
    predicted_column: str = PREDICTED,
) -> pd.DataFrame:
    """Compare bottom-up aggregates against independently fitted forecasts.

    Section 13 asks whether aggregated product-store forecasts match a separate
    forecast built at the aggregate level. They will not, and the interesting
    part is the sign and size of the gap rather than the fact of it.

    A systematic gap in one direction is evidence of a **level bias in the
    bottom model** - which is plausible here, because excluding stockout targets
    removes the high-demand tail. The check is diagnostic; nothing is reconciled,
    because reconciliation would hide exactly the signal being looked for.
    """
    if bottom_up.empty or independent.empty:
        return pd.DataFrame()

    left = bottom_up.groupby(level_column, observed=True)[predicted_column].sum()
    right = independent.groupby(level_column, observed=True)[predicted_column].sum()

    merged = pd.DataFrame({"bottom_up": left, "independent": right}).dropna()
    if merged.empty:
        return merged

    merged["gap"] = merged["bottom_up"] - merged["independent"]
    merged["gap_pct"] = merged["gap"] / merged["independent"].replace(0, np.nan)
    return merged.reset_index()


def revenue_impact(
    frame: pd.DataFrame,
    *,
    actual_column: str = TARGET,
    predicted_column: str = PREDICTED,
    price_column: str = "h_selling_price",
) -> dict[str, float]:
    """Translate unit error into money (brief section 11).

    Uses the **planned** price attached to the target date, which was knowable at
    forecast time. Section 11 is explicit that the realised price must not be
    used: a forecast made in June cannot be judged against a September price
    nobody knew in June, and doing so would mix pricing surprise into what is
    supposed to be a demand-accuracy figure.
    """
    if frame.empty or price_column not in frame.columns:
        return {}

    price = pd.to_numeric(frame[price_column], errors="coerce")
    actual = pd.to_numeric(frame[actual_column], errors="coerce")
    predicted = pd.to_numeric(frame[predicted_column], errors="coerce")
    usable = price.notna() & actual.notna() & predicted.notna()
    if not usable.any():
        return {}

    actual_revenue = float((actual[usable] * price[usable]).sum())
    predicted_revenue = float((predicted[usable] * price[usable]).sum())
    error = predicted_revenue - actual_revenue

    return {
        "actual_revenue": actual_revenue,
        "predicted_revenue": predicted_revenue,
        "revenue_error": error,
        "revenue_error_pct": error / actual_revenue if actual_revenue else float("nan"),
        "absolute_revenue_error": float(
            (np.abs(predicted[usable] - actual[usable]) * price[usable]).sum()
        ),
        "rows_priced": int(usable.sum()),
    }


def segment_errors(
    frame: pd.DataFrame,
    *,
    actual_column: str = TARGET,
    predicted_column: str = PREDICTED,
    segments: tuple[str, ...] = (
        "category", "brand", "region", "channel", "store_type",
        "h_season", "h_promotion_flag", "h_holiday_flag",
    ),
) -> pd.DataFrame:
    """Error by segment (brief section 24).

    Aggregate accuracy hides the case that matters: a model can look fine
    overall while being badly wrong for one category, and the manager who owns
    that category is the one being told a confident number.
    """
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for segment in segments:
        if segment not in frame.columns:
            continue
        for value, block in frame.groupby(segment, observed=True):
            if len(block) < 30:
                continue
            metrics = compute_metrics(block[actual_column], block[predicted_column])
            rows.append(
                {
                    "segment": segment,
                    "value": str(value),
                    "n": metrics.n,
                    "wmape": metrics.wmape,
                    "bias_pct": metrics.bias_pct,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("wmape", ascending=False).reset_index(drop=True)


def zero_demand_summary(frame: pd.DataFrame, *, actual_column: str = TARGET) -> dict[str, float]:
    """How much of the panel is zero or near-zero (brief section 26).

    Reported because it governs which metrics mean anything. MAPE is undefined at
    zero and unstable near it, which is precisely why WMAPE is the headline here
    and why MAPE is computed only over non-zero actuals with its exclusion count
    stated.
    """
    if frame.empty:
        return {}

    actual = pd.to_numeric(frame[actual_column], errors="coerce")
    return {
        "rows": len(actual),
        "zero_rows": int((actual == 0).sum()),
        "zero_share": float((actual == 0).mean()),
        "under_five_share": float((actual < 5).mean()),
        "mean_units": float(actual.mean()),
        "median_units": float(actual.median()),
    }


def format_bucket_table(bucket_metrics: dict[str, BaselineMetrics]) -> str:
    """Human-readable per-bucket metrics."""
    if not bucket_metrics:
        return "(no bucket metrics)"

    lines = [
        f"{'bucket':10s} {'n':>8s} {'WMAPE':>8s} {'MAE':>8s} {'bias':>8s}",
        "-" * 46,
    ]
    for bucket, metrics in bucket_metrics.items():
        lines.append(
            f"{bucket:10s} {metrics.n:>8,} {metrics.wmape:>7.1%} "
            f"{metrics.mae:>8.2f} {metrics.bias_pct:>+7.1%}"
        )
    return "\n".join(lines)
