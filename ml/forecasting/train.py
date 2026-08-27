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
from ml.forecasting.exceptions import (
    FeatureGenerationError,
    InsufficientHistoryError,
    InvalidForecastRequestError,
)
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


def build_estimator(
    name: str, *, seed: int = 42, params: dict[str, Any] | None = None
) -> BaselineEstimator:
    """Construct one candidate.

    ``params`` overrides the estimator's defaults and is how tuned
    hyperparameters reach the model. Silently ignored by estimators that take
    none - the naive benchmarks have nothing to tune, and refusing the argument
    would force every caller to special-case them.
    """
    if name not in ESTIMATORS:
        raise InvalidForecastRequestError(
            f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}",
            requested=name,
            available=sorted(ESTIMATORS),
        )

    estimator_class = ESTIMATORS[name]
    if params:
        try:
            return estimator_class(seed=seed, params=params)  # type: ignore[call-arg]
        except TypeError:
            logger.debug("forecast.params_ignored", model=name)

    return estimator_class(seed=seed)


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
            raise FeatureGenerationError(
                f"frame is missing {len(missing)} training features, e.g. {missing[:5]}",
                stage="predict",
                missing_count=len(missing),
                missing_sample=missing[:5],
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
        if not isinstance(series.dtype, pd.CategoricalDtype) and series.dtype != object:
            # Numeric and boolean columns are left alone; trees handle them
            # directly and categorising a bool gains nothing.
            continue
        # Levels are always normalised to sorted strings, even when the column
        # already carries a CategoricalDtype. Preserving the source dtype looks
        # harmless and is not: a category index of Python booleans is rejected
        # outright by XGBoost ("Category index must contain only values of the
        # same type"), and mixed-type levels sort unpredictably, which silently
        # changes the integer codes between frames.
        dtypes[column] = pd.CategoricalDtype(
            categories=sorted(series.dropna().astype(str).unique())
        )
    return dtypes


def _prepare(
    frame: pd.DataFrame,
    feature_names: list[str],
    categories: dict[str, pd.CategoricalDtype] | None = None,
) -> pd.DataFrame:
    """Project to the feature set and coerce every column to its training dtype.

    Coercing *all* columns, not just the categorical ones, is deliberate. The
    serving scaffold is built from different tables than the training panel, so a
    column that is float there can arrive as object here - most easily when every
    value in a short forecast window happens to be missing. XGBoost rejects that
    outright ("the data type doesn't match the one used in the training
    dataset"); LightGBM accepts it and quietly treats the column differently.

    Pinning the schema turns a whole class of train/serve skew into an
    impossibility rather than something to be discovered later.
    """
    working = frame.reindex(columns=feature_names).copy()

    for column in feature_names:
        target_dtype = (categories or {}).get(column)
        if target_dtype is not None:
            # Via string, to match how the levels were built. Casting a
            # categorical straight to another CategoricalDtype maps by value,
            # and a bool True would not match the string "True".
            values = working[column].astype(object)
            values = values.where(values.isna(), values.astype(str))
            working[column] = values.astype(target_dtype)
        elif working[column].dtype == object:
            # Not a known categorical, so it must be numeric in training.
            working[column] = pd.to_numeric(working[column], errors="coerce")

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
        raise InsufficientHistoryError(
            "training fold is empty; the split or the panel is wrong",
            available_days=0,
        )

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
