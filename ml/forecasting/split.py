"""Temporal splitting for horizon data, which needs one thing Step 4's did not.

Step 4's :func:`ml.baseline.training.build_temporal_split` carves folds from the
back of history and is correct for a nowcast, where a row's features and its
target share a date. Here they do not: a row at origin *t* is scored against the
outcome at *t + h*, up to 90 days later.

So a training origin sitting just before the fold boundary has its **target
inside the evaluation window**. The model is fitted on the very outcomes it is
about to be scored on, the test metric improves, and nothing anywhere raises.

The fix is an **embargo**: a gap of ``max_horizon`` days between the last usable
training origin and the first evaluation origin. It costs real training data -
90 days of origins per boundary - and that cost is the point. Without it the
numbers are not measuring generalisation.

Everything splits on the **origin date**, never the target date and never the
row. Rows sharing an origin are not independent observations: they see the same
history and differ only in ``horizon_step``, so splitting by row would scatter
one origin's rows across train and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.observability.logging import get_logger
from ml.forecasting.config import ForecastConfig
from ml.forecasting.dataset import ORIGIN_DATE, TARGET_DATE
from ml.forecasting.exceptions import InsufficientHistoryError

logger = get_logger(__name__)


@dataclass(frozen=True)
class OriginSplit:
    """Chronological origin-date boundaries, with embargo gaps between folds.

    Four folds for the same reason Step 4 used four: conformal calibration needs
    data the model did not train on *and* that is not the test set, or the
    measured coverage is the calibration set's own quantile reported back to
    itself.
    """

    train_start: date
    train_end: date
    calibration_start: date
    calibration_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date
    embargo_days: int

    def describe(self) -> str:
        return (
            f"train {self.train_start}..{self.train_end} | "
            f"calib {self.calibration_start}..{self.calibration_end} | "
            f"valid {self.valid_start}..{self.valid_end} | "
            f"test {self.test_start}..{self.test_end} "
            f"(embargo {self.embargo_days}d)"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "train_start": str(self.train_start), "train_end": str(self.train_end),
            "calibration_start": str(self.calibration_start),
            "calibration_end": str(self.calibration_end),
            "valid_start": str(self.valid_start), "valid_end": str(self.valid_end),
            "test_start": str(self.test_start), "test_end": str(self.test_end),
            "embargo_days": str(self.embargo_days),
        }


def build_origin_split(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    origin_column: str = ORIGIN_DATE,
) -> OriginSplit:
    """Carve origin-date folds from the back of history, separated by embargoes.

    Ordered train -> calibration -> validation -> test, oldest to newest, with an
    embargo before each evaluation fold. Calibration sits before validation
    deliberately: validation drives early stopping, so it must be the fold
    closest to test for the stopping point to reflect the most recent regime.
    """
    origins = pd.to_datetime(frame[origin_column]).dt.date
    first, last = origins.min(), origins.max()

    embargo = config.validation.embargo_days
    test_days = config.validation.test_days
    valid_days = config.validation.valid_days
    calibration_days = config.validation.calibration_days

    # Three embargo gaps, one before each of calibration, validation and test.
    required = test_days + valid_days + calibration_days + (3 * embargo) + 120
    available = (last - first).days
    if available < required:
        raise InsufficientHistoryError(
            f"only {available} days of origins but the split needs {required} "
            f"(test {test_days} + valid {valid_days} + calibration "
            f"{calibration_days} + 3 embargo gaps of {embargo} + at least 120 "
            f"for training). The embargo is what makes the evaluation honest at "
            f"a {config.max_horizon}-day horizon; shortening it would be "
            f"cheaper and wrong.",
            available_days=available,
            required_days=required,
        )

    test_end = last
    test_start = test_end - timedelta(days=test_days - 1)

    valid_end = test_start - timedelta(days=embargo + 1)
    valid_start = valid_end - timedelta(days=valid_days - 1)

    calibration_end = valid_start - timedelta(days=embargo + 1)
    calibration_start = calibration_end - timedelta(days=calibration_days - 1)

    train_end = calibration_start - timedelta(days=embargo + 1)

    split = OriginSplit(
        train_start=first, train_end=train_end,
        calibration_start=calibration_start, calibration_end=calibration_end,
        valid_start=valid_start, valid_end=valid_end,
        test_start=test_start, test_end=test_end,
        embargo_days=embargo,
    )
    logger.info("forecast.split_built", **split.to_dict())
    return split


def slice_fold(
    frame: pd.DataFrame,
    start: date,
    end: date,
    *,
    origin_column: str = ORIGIN_DATE,
) -> pd.DataFrame:
    """Rows whose **origin** falls inside the window."""
    origins = pd.to_datetime(frame[origin_column]).dt.date
    return frame[(origins >= start) & (origins <= end)]


def worst_case_gap_days(split: OriginSplit, max_horizon: int) -> int:
    """Slack between the furthest possible training target and the *nearest*
    evaluation fold.

    The structural guarantee, as opposed to the realised one. A training origin
    at ``train_end`` drawing the longest horizon lands on
    ``train_end + max_horizon``; if that reaches any evaluation fold, the design
    permits leakage regardless of whether this particular random draw happened
    to produce it.

    Two details worth stating, because both are easy to get wrong:

    * It is the **minimum** across all three evaluation folds. Measuring against
      the test fold alone is far too lenient - calibration and validation sit
      between train and test, so the test window looks safe even with no embargo
      at all, and the check silently passes on an unsound split.
    * It uses the configured maximum rather than the realised maximum. With few
      horizons drawn per origin the worst case is often simply not sampled, so a
      realised check passes on sampling luck rather than on design.
    """
    return min(
        (fold_start - split.train_end).days - max_horizon
        for fold_start in (split.calibration_start, split.valid_start, split.test_start)
    )


def leakage_gap_days(train: pd.DataFrame, evaluation: pd.DataFrame) -> int:
    """Days between the latest training *target* and the earliest eval origin.

    The number the embargo exists to keep positive, exposed so tests and the
    training report can assert on it rather than trusting the construction.
    Negative means a training row's outcome lies inside the evaluation window.
    """
    if train.empty or evaluation.empty:
        return 0
    latest_target = pd.to_datetime(train[TARGET_DATE]).max()
    earliest_origin = pd.to_datetime(evaluation[ORIGIN_DATE]).min()
    return int((earliest_origin - latest_target).days)
