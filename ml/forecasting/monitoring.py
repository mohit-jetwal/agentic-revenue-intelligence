"""Monitoring primitives (brief section 30).

Section 30 says not to build production monitoring yet, and this module takes
that seriously: it implements only the parts that are real *now*, and says
plainly what is missing.

**What is here** computes something meaningful from data that exists: feature
drift between the training window and a later one, forecast error once actuals
arrive, and the shape of the prediction distribution.

**What is not here, and why.** There is no alerting, no thresholds file, no
scheduler, no dashboard. Nothing is serving traffic yet, so an alert has no
recipient and a threshold would be invented rather than derived. Writing that
scaffolding now would produce code that looks like monitoring and monitors
nothing - and it would have to be rewritten anyway, because Stage 2 replaces
this with Databricks Lakehouse Monitoring, which computes drift natively over
Delta tables.

The functions here are therefore designed as *inputs* to that, not a substitute:
each returns a frame that a scheduled job could persist, and the migration note
in the docs maps each to its Databricks equivalent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import BaselineMetrics, compute_metrics
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import HORIZON_STEP, TARGET, TARGET_DATE

logger = get_logger(__name__)

#: Population Stability Index thresholds, from the conventional banking scale.
#: Reported rather than acted on - the labels are a reading aid, not a trigger.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """PSI between two distributions of one feature.

    Uses reference quantiles as the bin edges, so the reference is uniform by
    construction and any imbalance is genuinely the current window's. Bins with
    no mass get a small epsilon, because the log ratio is otherwise infinite and
    one empty bin would dominate the whole statistic.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < 50 or len(current) < 50:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    epsilon = 1e-6
    reference_share = np.histogram(reference, bins=edges)[0] / len(reference) + epsilon
    current_share = np.histogram(current, bins=edges)[0] / len(current) + epsilon

    ratio = np.log(current_share / reference_share)
    return float(np.sum((current_share - reference_share) * ratio))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """Feature drift between a reference window and a current one.

    Numeric features only. Categorical drift matters too, but PSI on a
    categorical needs a different construction, and shipping a numeric-only
    report that says so is better than shipping one that quietly skips columns.
    """
    if reference.empty or current.empty:
        return pd.DataFrame()

    rows = []
    for feature in feature_names[:top_n]:
        if feature not in reference.columns or feature not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[feature]):
            continue

        psi = population_stability_index(reference[feature], current[feature])
        if not np.isfinite(psi):
            continue

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "severity": (
                    "major" if psi >= PSI_MAJOR else "minor" if psi >= PSI_MINOR else "stable"
                ),
                "reference_mean": float(pd.to_numeric(reference[feature], errors="coerce").mean()),
                "current_mean": float(pd.to_numeric(current[feature], errors="coerce").mean()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    logger.info(
        "forecast.drift_computed",
        features=len(table),
        drifting=int((table["psi"] >= PSI_MINOR).sum()),
    )
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def forecast_vs_actual(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    config: ForecastConfig,
    *,
    predicted_column: str = "predicted_units",
    keys: tuple[str, ...] = ("product_id", "store_id"),
) -> dict[str, BaselineMetrics]:
    """Score past forecasts once the actuals have arrived.

    The check that would actually catch a model going stale in production, and
    the reason it belongs here rather than in ``evaluate.py``: this joins a
    *stored* forecast against outcomes that did not exist when it was made,
    which is a different operation from scoring a held-out fold.

    Returned per horizon bucket, since a model usually degrades at long range
    first and a blended number hides it.
    """
    if forecasts.empty or actuals.empty:
        return {}

    merged = forecasts.merge(
        actuals[[*keys, TARGET_DATE, TARGET]], on=[*keys, TARGET_DATE], how="inner"
    )
    if merged.empty:
        return {}

    merged["_bucket"] = merged[HORIZON_STEP].map(config.intervals.bucket_for)
    results: dict[str, BaselineMetrics] = {}
    for bucket in config.intervals.bucket_labels():
        block = merged[merged["_bucket"] == bucket]
        if block.empty:
            continue
        results[bucket] = compute_metrics(block[TARGET], block[predicted_column])

    logger.info("forecast.scored_against_actuals", buckets=len(results), rows=len(merged))
    return results


def prediction_distribution(
    frame: pd.DataFrame, *, predicted_column: str = "predicted_units"
) -> dict[str, float]:
    """Shape of the prediction distribution.

    Cheap and surprisingly diagnostic. A model that has quietly collapsed toward
    its global mean still produces plausible aggregate totals - the failure only
    shows up as a spread that has narrowed, which is exactly what this captures.
    """
    if frame.empty or predicted_column not in frame.columns:
        return {}

    values = pd.to_numeric(frame[predicted_column], errors="coerce").dropna()
    if values.empty:
        return {}

    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(values.quantile(0.10)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "zero_share": float((values <= 0).mean()),
        # Standard deviation over mean. A sharp fall between runs is the
        # signature of a model regressing toward its own average.
        "coefficient_of_variation": float(values.std() / values.mean())
        if values.mean()
        else float("nan"),
    }


def segment_performance(
    scored: pd.DataFrame,
    *,
    segment: str,
    predicted_column: str = "predicted_units",
    min_rows: int = 30,
) -> pd.DataFrame:
    """Error by segment, for spotting a subgroup degrading on its own."""
    if scored.empty or segment not in scored.columns:
        return pd.DataFrame()

    rows = []
    for value, block in scored.groupby(segment, observed=True):
        if len(block) < min_rows:
            continue
        metrics = compute_metrics(block[TARGET], block[predicted_column])
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
