"""Training loop for one candidate over the horizon dataset.

Kept deliberately thin. Almost everything here is composition of parts that
already exist and are already tested - :class:`~ml.baseline.models.LightGBMBaseline`,
the metric functions, the conformal calibrator - so the module's job is to make
the *order* of operations visible in one place rather than implied across six.

``ml.baseline.training.train_baseline`` could not be reused: it hard-codes a
same-day target, applies promotion-approach row filtering that has no meaning
here, and calibrates a single scalar quantile where this needs one per horizon
bucket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import BaselineMetrics, compute_metrics
from ml.baseline.models import BaselineEstimator, LightGBMBaseline
from ml.forecasting.baselines import HorizonNaive, HorizonSeasonalNaive
from ml.forecasting.config import ForecastConfig
from ml.forecasting.conformal import HorizonCalibration, calibrate_by_horizon
from ml.forecasting.dataset import HORIZON_STEP, TARGET, HorizonDataset
from ml.forecasting.split import OriginSplit, slice_fold
from ml.forecasting.xgboost_model import XGBoostForecaster

logger = get_logger(__name__)

#: Estimator registry. Ordered simplest first, which is also the order the
#: comparison table reads best in.
ESTIMATORS: dict[str, type[BaselineEstimator]] = {
    "horizon_naive": HorizonNaive,
    "horizon_seasonal_naive": HorizonSeasonalNaive,
    "lightgbm": LightGBMBaseline,
    "xgboost": XGBoostForecaster,
}


def build_estimator(name: str, *, seed: int = 42) -> BaselineEstimator:
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](seed=seed)


@dataclass
class TrainedForecaster:
    """A fitted candidate and everything needed to judge it."""

    estimator: BaselineEstimator
    split: OriginSplit
    feature_names: list[str]
    metrics: dict[str, BaselineMetrics] = field(default_factory=dict)
    #: WMAPE per horizon bucket. The headline breakdown - a single blended
    #: number hides that a model may be excellent at h=1 and useless at h=90.
    bucket_metrics: dict[str, BaselineMetrics] = field(default_factory=dict)
    calibration: HorizonCalibration | None = None
    #: Frozen categorical encodings, so serving codes match training codes.
    categories: dict[str, pd.CategoricalDtype] = field(default_factory=dict)
    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    @property
    def name(self) -> str:
        return self.estimator.name

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict on a raw frame, applying the training encodings."""
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(
                f"frame is missing {len(missing)} training features, e.g. {missing[:5]}"
            )
        return self.estimator.predict(
            _prepare(frame, self.feature_names, self.categories)
        )

    def summary(self) -> str:
        test = self.metrics.get("test")
        return f"{self.name}: {test.summary() if test else 'not evaluated'}"


def build_category_dtypes(
    frame: pd.DataFrame, feature_names: list[str]
) -> dict[str, pd.CategoricalDtype]:
    """Freeze one categorical dtype per column, from the whole dataset.

    Necessary because ``astype("category")`` infers its levels from whatever
    frame it is handed. Fit on the training fold and predict on a later one, and
    a level that appears only later - a season, a promotion type - is either an
    unseen category (XGBoost raises) or silently remapped to a different integer
    code (LightGBM does not raise, which is worse).

    Found the hard way: XGBoost refused a calibration fold containing
    ``h_season="Autumn"`` when the training fold happened not to. LightGBM had
    been quietly mis-coding the same column all along.
    """
    dtypes: dict[str, pd.CategoricalDtype] = {}
    for column in feature_names:
        if column not in frame.columns:
            continue
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            dtypes[column] = series.dtype
        elif series.dtype == object:
            dtypes[column] = pd.CategoricalDtype(
                categories=sorted(series.dropna().astype(str).unique())
            )
    return dtypes


def _prepare(
    frame: pd.DataFrame,
    feature_names: list[str],
    categories: dict[str, pd.CategoricalDtype] | None = None,
) -> pd.DataFrame:
    """Project to the feature set with consistent categorical encodings."""
    working = frame[feature_names].copy()
    for column in working.columns:
        if categories is not None and column in categories:
            values = working[column]
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            working[column] = values.astype(categories[column])
        elif working[column].dtype == object:
            working[column] = working[column].astype("category")
    return working


def train_forecaster(
    dataset: HorizonDataset,
    estimator: BaselineEstimator,
    config: ForecastConfig,
    split: OriginSplit,
) -> TrainedForecaster:
    """Fit one candidate, calibrate it, and score it on the test fold.

    The sequence is the point: train on origins strictly before calibration,
    early-stop against validation, calibrate intervals on a fold the model never
    saw, and only then score on test. Any reordering makes one of the resulting
    numbers a self-report.
    """
    frame = dataset.frame
    features = dataset.feature_names

    train = slice_fold(frame, split.train_start, split.train_end)
    calibration = slice_fold(frame, split.calibration_start, split.calibration_end)
    validation = slice_fold(frame, split.valid_start, split.valid_end)
    test = slice_fold(frame, split.test_start, split.test_end)

    if train.empty:
        raise ValueError("training fold is empty; the split or the panel is wrong")

    # Derived from the whole dataset, not the training fold, so every fold and
    # every future prediction encodes the same level to the same code.
    categories = build_category_dtypes(frame, features)

    started = time.perf_counter()
    estimator.fit(
        _prepare(train, features, categories),
        train[TARGET],
        X_valid=_prepare(validation, features, categories) if not validation.empty else None,
        y_valid=validation[TARGET] if not validation.empty else None,
    )
    train_seconds = time.perf_counter() - started

    trained = TrainedForecaster(
        estimator=estimator,
        split=split,
        feature_names=list(features),
        categories=categories,
        train_seconds=train_seconds,
    )

    # -- intervals, on a fold the model has not seen -----------------------
    if not calibration.empty:
        calibration_predictions = trained.predict(calibration)
        trained.calibration = calibrate_by_horizon(
            calibration.assign(_pred=calibration_predictions),
            config,
            actual_column=TARGET,
            predicted_column="_pred",
        )

    # -- score --------------------------------------------------------------
    if not test.empty:
        started = time.perf_counter()
        predictions = trained.predict(test)
        trained.predict_seconds = time.perf_counter() - started

        trained.metrics["test"] = compute_metrics(test[TARGET], pd.Series(predictions))
        trained.metrics["train"] = compute_metrics(
            train[TARGET], pd.Series(trained.predict(train))
        )
        trained.bucket_metrics = _metrics_by_bucket(
            test.assign(_pred=predictions), config, predicted_column="_pred"
        )

    logger.info(
        "forecast.candidate_trained",
        model=estimator.name,
        train_rows=len(train),
        test_rows=len(test),
        wmape=round(trained.metrics["test"].wmape, 4) if "test" in trained.metrics else None,
        train_seconds=round(train_seconds, 2),
    )
    return trained


def _metrics_by_bucket(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    predicted_column: str,
    actual_column: str = TARGET,
) -> dict[str, BaselineMetrics]:
    """Metrics per horizon bucket.

    Reported instead of a blended figure, never alongside it as an afterthought.
    Forecast error grows with horizon by nature, so one number averaged over
    1..90 days describes no decision anyone actually makes: nobody forecasts
    "somewhere between tomorrow and three months".
    """
    if frame.empty or HORIZON_STEP not in frame.columns:
        return {}

    working = frame.copy()
    working["_bucket"] = working[HORIZON_STEP].map(config.intervals.bucket_for)

    results: dict[str, BaselineMetrics] = {}
    for label in config.intervals.bucket_labels():
        block = working[working["_bucket"] == label]
        if block.empty:
            continue
        results[label] = compute_metrics(block[actual_column], block[predicted_column])
    return results


def forecast_value_added(
    model_metrics: dict[str, BaselineMetrics],
    benchmark_metrics: dict[str, BaselineMetrics],
) -> dict[str, float]:
    """FVA in WMAPE percentage points, per horizon bucket.

    ``FVA = WMAPE(benchmark) - WMAPE(model)``. Positive means the model reduced
    error over what a planner would get unaided.

    Reported in percentage points rather than as a ratio, deliberately: against a
    35% irreducible noise floor a ratio compresses everything into a narrow band
    and makes a genuine 4-point improvement look like a rounding error.

    Broken out by bucket because that is where the decision lives. A model that
    adds 6 points at h=1-7 and 0 at h=57-90 should be used for the short horizon
    and replaced by the seasonal naive for the long one - which a blended figure
    would never reveal.
    """
    return {
        bucket: benchmark_metrics[bucket].wmape - metrics.wmape
        for bucket, metrics in model_metrics.items()
        if bucket in benchmark_metrics
    }


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result
