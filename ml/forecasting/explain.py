"""Explainability (brief section 17).

Two things worth stating before the code.

**SHAP is not used, and that is a decision rather than an omission.** It is not
a project dependency, and at this scale it costs considerably more than it adds
for the question actually being asked - *what drives forecast demand?* -
which permutation importance answers directly by measuring the WMAPE degradation
when a feature is shuffled. Step 4 made the same call for the same reason.

**Importance is reported by horizon bucket, which is new here.** A single ranking
averaged over 1-90 days hides the most informative thing the model can tell you:
recent demand history should dominate at h=1 and decay toward irrelevance at
h=90, while calendar and planned-promotion features should hold their weight or
grow. If ``lag_1_units`` is still the top feature at ninety days, the
origin/target join is suspect - so this doubles as a leakage diagnostic.

Nothing here is causal. Permutation importance measures what the model *relies
on*, which is a statement about the fitted function, not about demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import compute_metrics
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import HORIZON_STEP, TARGET
from ml.forecasting.train import TrainedForecaster, _prepare

logger = get_logger(__name__)

#: Feature-name prefixes grouped into families for the summary table.
#:
#: **Recent history and the seasonal anchor are separate families, deliberately.**
#: Lumping them together as "demand history" hides the single most informative
#: contrast the model can show. They have opposite horizon profiles: yesterday's
#: sales decay toward irrelevance as the horizon grows, while units from 364 days
#: before the *target* are exactly as knowable at h=90 as at h=1 and become
#: relatively *more* important as the recent signal fades.
#:
#: Grouped together, the combined share rises with horizon - which reads as a
#: leakage alarm under the rule in :func:`recent_history_decay` and is in fact
#: the seasonal anchor doing its job.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("seasonal anchor", ("seasonal_reference", "lag_364")),
    ("recent history", ("lag_1_", "lag_7_", "lag_14_", "rolling_7_", "rolling_14_",
                        "demand_momentum")),
    ("medium history", ("lag_", "rolling_", "demand_")),
    ("calendar", ("h_day", "h_week", "h_month", "h_quarter", "h_dow", "h_doy",
                  "h_weekend", "h_holiday", "h_festival", "h_season", "h_financial",
                  "h_days_to_festival", "h_days_since_festival")),
    ("planned promotion", ("h_promotion", "h_display", "h_bundle", "h_days_into",
                           "h_days_until")),
    ("planned price", ("h_selling_price", "h_regular_price", "h_discount")),
    ("price position", ("selling_price", "regular_price", "discount_depth",
                        "price_", "historical_average_price")),
    ("competitor", ("competitor_", "cheaper_than")),
    ("promotion history", ("promotions_last", "days_since_promotion")),
    ("product/store", ("category", "subcategory", "brand", "pack_size", "unit_cost",
                       "product_", "store_", "channel", "region", "base_price",
                       "is_new_product")),
    ("horizon", (HORIZON_STEP,)),
)


def _family_of(feature: str) -> str:
    for family, prefixes in _FAMILIES:
        if any(feature.startswith(prefix) for prefix in prefixes):
            return family
    return "other"


def permutation_importance_by_horizon(
    trained: TrainedForecaster,
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    top_n: int | None = None,
    repeats: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """WMAPE degradation per feature, computed within each horizon bucket.

    Only the top ``top_n`` features by model gain are permuted. Shuffling all
    ~90 features across six buckets with three repeats each would be well over a
    thousand full prediction passes for a ranking whose tail nobody reads.
    """
    if frame.empty:
        return pd.DataFrame()

    top_n = top_n or config.explainability.top_n_features
    repeats = repeats or config.explainability.permutation_repeats

    gain = trained.estimator.feature_importance()
    if gain is not None and not gain.empty:
        candidates = [f for f in gain["feature"].head(top_n) if f in trained.feature_names]
    else:
        candidates = list(trained.feature_names)[:top_n]

    working = frame.assign(_bucket=frame[HORIZON_STEP].map(config.intervals.bucket_for))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for bucket in config.intervals.bucket_labels():
        block = working[working["_bucket"] == bucket]
        if len(block) < 50:
            continue

        prepared = _prepare(block, trained.feature_names, trained.categories)
        baseline = compute_metrics(
            block[TARGET], pd.Series(trained.estimator.predict(prepared))
        ).wmape

        for feature in candidates:
            degradations = []
            original_dtype = prepared[feature].dtype
            for _ in range(repeats):
                shuffled = prepared.copy()
                # Re-cast after shuffling. `.to_numpy()` on a categorical column
                # returns an object array, so assigning it back silently demotes
                # the dtype - and XGBoost then refuses the frame outright
                # ("DataFrame.dtypes for data must be int, float, bool or
                # category"). LightGBM accepts it and treats the column
                # differently instead, which is the quieter failure.
                values = (
                    shuffled[feature]
                    .sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
                    .to_numpy()
                )
                shuffled[feature] = pd.Series(
                    values, index=shuffled.index
                ).astype(original_dtype)
                permuted = compute_metrics(
                    block[TARGET], pd.Series(trained.estimator.predict(shuffled))
                ).wmape
                degradations.append(permuted - baseline)

            rows.append(
                {
                    "bucket": bucket,
                    "feature": feature,
                    "family": _family_of(feature),
                    "importance": float(np.mean(degradations)),
                    "std": float(np.std(degradations)),
                    "n": len(block),
                }
            )

    table = pd.DataFrame(rows)
    logger.info(
        "forecast.importance_computed",
        features=len(candidates),
        buckets=table["bucket"].nunique() if not table.empty else 0,
    )
    return table


def importance_by_family(table: pd.DataFrame) -> pd.DataFrame:
    """Roll feature importance up to families, per horizon bucket.

    The readable version. Ninety feature names tell a reviewer very little;
    "demand history matters most at a week and least at a quarter, while planned
    promotion holds steady" is the actual finding.
    """
    if table.empty:
        return table

    summary = (
        table.groupby(["bucket", "family"], observed=True)["importance"]
        .sum()
        .reset_index()
    )
    return summary.pivot(index="family", columns="bucket", values="importance").fillna(0.0)


def recent_history_decay(table: pd.DataFrame) -> pd.DataFrame:
    """How much the model leans on *recent* demand as the horizon grows.

    Expected to fall: yesterday's sales say a lot about tomorrow and little about
    three months out. A flat or rising profile is worth investigating, because it
    would mean recent history is somehow still informative at long range - which
    for a genuine forecast it should not be.

    **Only the recent family counts here.** An earlier version of this function
    included the 364-day seasonal anchor, and the combined share duly rose with
    horizon - which looked exactly like the leakage signal above and was nothing
    of the sort. The anchor is equally knowable at every horizon, so as the
    recent signal decays the anchor's *share* necessarily grows. Two features
    with opposite horizon profiles must not be averaged into one diagnostic.
    """
    if table.empty:
        return table

    totals = table.groupby("bucket", observed=True)["importance"].sum()
    recent = (
        table[table["family"] == "recent history"]
        .groupby("bucket", observed=True)["importance"]
        .sum()
        .reindex(totals.index)
        .fillna(0.0)
    )
    seasonal = (
        table[table["family"] == "seasonal anchor"]
        .groupby("bucket", observed=True)["importance"]
        .sum()
        .reindex(totals.index)
        .fillna(0.0)
    )

    return pd.DataFrame(
        {
            "recent_history_share": recent / totals.replace(0, np.nan),
            "seasonal_anchor_share": seasonal / totals.replace(0, np.nan),
        }
    ).reset_index()


def top_drivers(table: pd.DataFrame, *, bucket: str | None = None, n: int = 10) -> pd.DataFrame:
    """The n most-relied-on features, overall or within one bucket."""
    if table.empty:
        return table

    working = table if bucket is None else table[table["bucket"] == bucket]
    return (
        working.groupby(["feature", "family"], observed=True)["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
