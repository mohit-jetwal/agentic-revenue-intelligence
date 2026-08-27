"""Walk-forward validation over origins (brief section 7).

``ml.baseline.training.expanding_window_backtest`` could not be reused: it slices
folds on the row's own date, which for a horizon dataset is ambiguous - a row has
two dates - and it has no embargo, so training targets reach into the fold being
scored.

The fold containers themselves *are* reused. :class:`BacktestFold` and
:class:`BacktestResult` are horizon-agnostic and already know how to summarise a
sequence of folds, so only the fold construction is new.

Expanding rather than rolling: a forecaster should get better as history
accumulates, and a rolling window throws away the older seasons that make the
364-day lag meaningful in the first place.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import compute_metrics
from ml.baseline.models import BaselineEstimator
from ml.baseline.training import BacktestFold, BacktestResult
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import HORIZON_STEP, ORIGIN_DATE, TARGET, HorizonDataset
from ml.forecasting.train import _prepare, build_category_dtypes

logger = get_logger(__name__)


def expanding_origin_backtest(
    dataset: HorizonDataset,
    estimator_factory: Callable[[], BaselineEstimator],
    config: ForecastConfig,
    *,
    n_folds: int | None = None,
    fold_days: int = 90,
    min_train_days: int = 365,
) -> BacktestResult:
    """Refit on an expanding origin window, scoring each subsequent block.

    Every fold keeps the same embargo as the main split. Dropping it here while
    keeping it there would produce backtest numbers systematically better than
    the test number, which is the sort of inconsistency that gets explained away
    as "the backtest is optimistic" rather than fixed.
    """
    frame = dataset.frame
    if frame.empty:
        return BacktestResult(folds=[])

    folds_wanted = n_folds if n_folds is not None else config.validation.n_folds
    embargo = config.validation.embargo_days

    origins = pd.to_datetime(frame[ORIGIN_DATE]).dt.date
    first, last = origins.min(), origins.max()
    categories = build_category_dtypes(frame, dataset.feature_names)

    folds: list[BacktestFold] = []
    for index in range(folds_wanted):
        # Each fold scores a later block; training expands to everything before
        # it, minus the embargo.
        valid_end = last - timedelta(days=fold_days * (folds_wanted - index - 1))
        valid_start = valid_end - timedelta(days=fold_days - 1)
        train_end = valid_start - timedelta(days=embargo + 1)

        if (train_end - first).days < min_train_days:
            logger.info(
                "forecast.backtest_fold_skipped",
                fold=index,
                reason="insufficient training history after the embargo",
                train_days=(train_end - first).days,
            )
            continue

        train = frame[origins <= train_end]
        validation = frame[(origins >= valid_start) & (origins <= valid_end)]
        if train.empty or validation.empty:
            continue

        estimator = estimator_factory()
        estimator.fit(
            _prepare(train, dataset.feature_names, categories),
            train[TARGET],
        )
        predictions = estimator.predict(
            _prepare(validation, dataset.feature_names, categories)
        )
        metrics = compute_metrics(validation[TARGET], pd.Series(predictions))

        folds.append(
            BacktestFold(
                index=index,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                metrics=metrics,
                train_rows=len(train),
            )
        )
        logger.info(
            "forecast.backtest_fold",
            fold=index,
            wmape=round(metrics.wmape, 4),
            train_rows=len(train),
            valid_rows=len(validation),
        )

    return BacktestResult(folds=folds)


def backtest_by_horizon(
    dataset: HorizonDataset,
    estimator_factory: Callable[[], BaselineEstimator],
    config: ForecastConfig,
    *,
    n_folds: int | None = None,
    fold_days: int = 90,
) -> pd.DataFrame:
    """Fold-by-bucket stability.

    Stability at a blended level can hide the failure that matters: a model
    steady overall but wildly variable at long horizons is exactly the one that
    will embarrass a quarterly plan, and it is long-horizon numbers that nobody
    can sanity-check by eye.
    """
    frame = dataset.frame
    if frame.empty:
        return pd.DataFrame()

    folds_wanted = n_folds if n_folds is not None else config.validation.n_folds
    embargo = config.validation.embargo_days
    origins = pd.to_datetime(frame[ORIGIN_DATE]).dt.date
    first, last = origins.min(), origins.max()
    categories = build_category_dtypes(frame, dataset.feature_names)

    rows = []
    for index in range(folds_wanted):
        valid_end = last - timedelta(days=fold_days * (folds_wanted - index - 1))
        valid_start = valid_end - timedelta(days=fold_days - 1)
        train_end = valid_start - timedelta(days=embargo + 1)

        train = frame[origins <= train_end]
        validation = frame[(origins >= valid_start) & (origins <= valid_end)]
        if train.empty or validation.empty or (train_end - first).days < 365:
            continue

        estimator = estimator_factory()
        estimator.fit(_prepare(train, dataset.feature_names, categories), train[TARGET])
        scored = validation.assign(
            _pred=estimator.predict(_prepare(validation, dataset.feature_names, categories))
        )
        scored["_bucket"] = scored[HORIZON_STEP].map(config.intervals.bucket_for)

        for bucket, block in scored.groupby("_bucket", observed=True):
            metrics = compute_metrics(block[TARGET], block["_pred"])
            rows.append(
                {
                    "fold": index,
                    "bucket": bucket,
                    "n": metrics.n,
                    "wmape": metrics.wmape,
                    "bias_pct": metrics.bias_pct,
                }
            )

    return pd.DataFrame(rows)


def stability_by_bucket(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of WMAPE per bucket across folds."""
    if fold_table.empty:
        return fold_table

    summary = (
        fold_table.groupby("bucket", observed=True)["wmape"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Same rule Step 4 used: a coefficient of variation above 0.25 means a single
    # headline number is misleading about what the next quarter will look like.
    summary["stable"] = (summary["std"] / summary["mean"]) < 0.25
    return summary
