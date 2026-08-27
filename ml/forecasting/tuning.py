"""Controlled hyperparameter search (brief section 13).

Section 13 says explicitly: do **not** perform an excessive search. That is the
right instruction here for a reason worth stating, because it is easy to read as
laziness.

The model already sits at roughly **1.25x the irreducible noise floor** measured
in Step 4 - 43.8% WMAPE against a floor of 35.0%. That leaves about nine
percentage points of learnable signal *in total*, and hyperparameters compete for
a fraction of it. A thousand-trial sweep would spend hours to move a number whose
own fold-to-fold standard deviation is around one point. It would also overfit the
validation fold: with enough trials, the best score is the luckiest one.

So: **a small, seeded, fully-recorded random search.** Twenty trials by default,
every one logged, and the full trial table returned so that a marginal winner is
visibly marginal rather than presented as a discovery.

Two design choices carry the honesty of the result:

* **Random search, not grid.** For the same budget, random search explores each
  individual parameter at more distinct values, and most of the parameters here
  matter far less than one or two of them. A grid spends its budget proving that
  the unimportant ones are unimportant.
* **Scored on the validation fold, never test.** Tuning against test would make
  every subsequently reported number a self-report. The test fold stays untouched
  until selection is finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import compute_metrics
from ml.baseline.models import BaselineEstimator
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import TARGET, HorizonDataset
from ml.forecasting.split import OriginSplit, slice_fold
from ml.forecasting.train import _prepare, build_category_dtypes

logger = get_logger(__name__)


@dataclass
class Trial:
    """One sampled configuration and what it scored."""

    index: int
    params: dict[str, Any]
    wmape: float
    mae: float
    bias_pct: float
    seconds: float

    def to_row(self) -> dict[str, Any]:
        return {
            "trial": self.index,
            "wmape": self.wmape,
            "mae": self.mae,
            "bias_pct": self.bias_pct,
            "seconds": round(self.seconds, 2),
            **{f"hp_{k}": v for k, v in self.params.items()},
        }


@dataclass
class TuningResult:
    """Every trial, plus the winner and how much it actually won by."""

    trials: list[Trial] = field(default_factory=list)
    best: Trial | None = None
    baseline_wmape: float = float("nan")
    total_seconds: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([t.to_row() for t in self.trials])
        return frame.sort_values("wmape").reset_index(drop=True) if not frame.empty else frame

    @property
    def improvement_pp(self) -> float:
        """Percentage points of WMAPE gained over the default parameters.

        The number that decides whether tuning was worth doing. Reported even -
        especially - when it is negligible.
        """
        if self.best is None or not np.isfinite(self.baseline_wmape):
            return float("nan")
        return self.baseline_wmape - self.best.wmape

    def is_material(self, threshold_pp: float = 0.005) -> bool:
        """Whether the gain exceeds fold-to-fold noise.

        Default threshold is half a percentage point. Backtest standard deviation
        on this data runs 0.3-1.8 points, so anything under that is
        indistinguishable from which fold you happened to look at.
        """
        gain = self.improvement_pp
        return bool(np.isfinite(gain) and gain > threshold_pp)

    def summary(self) -> str:
        if self.best is None:
            return "no trials completed"
        verdict = (
            "material - adopt the tuned parameters"
            if self.is_material()
            else "within fold-to-fold noise - keep the defaults"
        )
        return (
            f"{len(self.trials)} trials in {self.total_seconds:.0f}s. "
            f"Best WMAPE {self.best.wmape:.2%} vs default {self.baseline_wmape:.2%} "
            f"({self.improvement_pp:+.2%} points) - {verdict}."
        )


def sample_params(
    space: dict[str, list[Any]], rng: np.random.Generator
) -> dict[str, Any]:
    """Draw one configuration from the declared space."""
    return {name: values[int(rng.integers(len(values)))] for name, values in space.items()}


def tune(
    dataset: HorizonDataset,
    split: OriginSplit,
    config: ForecastConfig,
    estimator_factory: Callable[[dict[str, Any]], BaselineEstimator],
    *,
    space: dict[str, list[Any]] | None = None,
    n_trials: int | None = None,
    seed: int | None = None,
) -> TuningResult:
    """Random search over ``space``, scored on the validation fold.

    ``estimator_factory`` takes a parameter dict and returns an unfitted
    estimator, so this function never needs to know which library is underneath.
    """
    tuning = config.tuning
    space = space or tuning.space
    n_trials = n_trials if n_trials is not None else tuning.n_trials
    rng = np.random.default_rng(seed if seed is not None else config.sampling.seed)

    frame = dataset.frame
    train = slice_fold(frame, split.train_start, split.train_end)
    validation = slice_fold(frame, split.valid_start, split.valid_end)

    if train.empty or validation.empty:
        logger.warning("forecast.tuning_skipped", reason="empty train or validation fold")
        return TuningResult()

    categories = build_category_dtypes(frame, dataset.feature_names)
    X_train = _prepare(train, dataset.feature_names, categories)
    X_valid = _prepare(validation, dataset.feature_names, categories)
    y_train, y_valid = train[TARGET], validation[TARGET]

    def score(params: dict[str, Any]) -> tuple[float, float, float, float]:
        started = time.perf_counter()
        estimator = estimator_factory(params)
        estimator.fit(X_train, y_train)
        metrics = compute_metrics(y_valid, pd.Series(estimator.predict(X_valid)))
        return metrics.wmape, metrics.mae, metrics.bias_pct, time.perf_counter() - started

    started = time.perf_counter()
    result = TuningResult()

    # The default configuration is scored first and separately. Without it there
    # is nothing to say whether the search found anything - only which of its own
    # samples was least bad.
    baseline_wmape, _, _, baseline_seconds = score({})
    result.baseline_wmape = baseline_wmape
    logger.info(
        "forecast.tuning_baseline",
        wmape=round(baseline_wmape, 4),
        seconds=round(baseline_seconds, 1),
    )

    seen: set[tuple[Any, ...]] = set()
    for index in range(n_trials):
        params = sample_params(space, rng)
        # Re-scoring an identical draw costs a fit and tells you nothing.
        signature = tuple(sorted(params.items()))
        if signature in seen:
            continue
        seen.add(signature)

        wmape, mae, bias_pct, seconds = score(params)
        trial = Trial(
            index=index, params=params, wmape=wmape, mae=mae,
            bias_pct=bias_pct, seconds=seconds,
        )
        result.trials.append(trial)

        if result.best is None or wmape < result.best.wmape:
            result.best = trial

        logger.info(
            "forecast.tuning_trial",
            trial=index,
            wmape=round(wmape, 4),
            best=round(result.best.wmape, 4),
        )

    result.total_seconds = time.perf_counter() - started

    # A search that cannot beat its own starting point is a result, not a bug.
    if result.best is not None and result.best.wmape >= baseline_wmape:
        logger.info(
            "forecast.tuning_no_gain",
            best=round(result.best.wmape, 4),
            baseline=round(baseline_wmape, 4),
            note="defaults retained",
        )

    logger.info(
        "forecast.tuning_completed",
        trials=len(result.trials),
        seconds=round(result.total_seconds, 1),
        improvement_pp=round(result.improvement_pp, 4),
        material=result.is_material(),
    )
    return result


def best_params(result: TuningResult, *, threshold_pp: float = 0.005) -> dict[str, Any]:
    """The parameters to actually use.

    Returns ``{}`` - meaning "keep the defaults" - unless the gain clears
    fold-to-fold noise. Adopting a configuration that won by less than the
    variance between folds is adopting noise, and it makes the next run's
    comparison harder to interpret rather than easier.
    """
    if result.best is None or not result.is_material(threshold_pp):
        return {}
    return dict(result.best.params)
