"""Temporal splitting, and the embargo that makes it honest at long horizons.

The ordering tests are the same shape as Step 4's. The embargo tests (T7) are
new and are the reason this module exists at all: without a gap of
``max_horizon`` days, a training origin near the fold boundary has its *target*
inside the evaluation window, and the model is scored on outcomes it was fitted
on. That inflates the test metric silently - there is no error, no warning, and
the resulting number looks like a good result.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from ml.forecasting.dataset import ORIGIN_DATE, TARGET_DATE
from ml.forecasting.split import (
    build_origin_split,
    leakage_gap_days,
    slice_fold,
    worst_case_gap_days,
)

pytestmark = pytest.mark.models


class TestFoldOrdering:
    def test_folds_are_ordered_and_non_overlapping(
        self, horizon_dataset, forecast_config
    ) -> None:
        split = build_origin_split(horizon_dataset.frame, forecast_config)

        assert split.train_start <= split.train_end
        assert split.train_end < split.calibration_start
        assert split.calibration_end < split.valid_start
        assert split.valid_end < split.test_start
        assert split.test_start <= split.test_end

    def test_test_fold_ends_at_the_last_origin(
        self, horizon_dataset, forecast_config
    ) -> None:
        split = build_origin_split(horizon_dataset.frame, forecast_config)
        last = pd.to_datetime(horizon_dataset.frame[ORIGIN_DATE]).dt.date.max()

        assert split.test_end == last

    def test_requested_fold_widths_are_honoured(
        self, horizon_dataset, forecast_config
    ) -> None:
        split = build_origin_split(horizon_dataset.frame, forecast_config)
        validation = forecast_config.validation

        assert (split.test_end - split.test_start).days + 1 == validation.test_days
        assert (split.valid_end - split.valid_start).days + 1 == validation.valid_days
        assert (
            split.calibration_end - split.calibration_start
        ).days + 1 == validation.calibration_days

    def test_insufficient_history_raises_rather_than_shrinking_the_embargo(
        self, horizon_dataset, forecast_config
    ) -> None:
        """Refusing is the right failure mode.

        The tempting alternative - quietly shortening the embargo to make the
        split fit - produces folds that look fine and metrics that are wrong.
        The error message says so explicitly, because that is the fix someone
        under time pressure would otherwise reach for.
        """
        frame = horizon_dataset.frame
        origins = pd.to_datetime(frame[ORIGIN_DATE]).dt.date
        short = frame[origins < origins.min() + timedelta(days=200)]

        with pytest.raises(ValueError, match="embargo"):
            build_origin_split(short, forecast_config)


class TestEmbargo:
    """T7: no training row's outcome may fall inside an evaluation window."""

    def test_embargo_gap_separates_calibration_from_train(
        self, horizon_dataset, forecast_config
    ) -> None:
        split = build_origin_split(horizon_dataset.frame, forecast_config)

        gap = (split.calibration_start - split.train_end).days

        assert gap > forecast_config.validation.embargo_days

    def test_embargo_gap_separates_validation_and_test(
        self, horizon_dataset, forecast_config
    ) -> None:
        split = build_origin_split(horizon_dataset.frame, forecast_config)

        assert (split.valid_start - split.calibration_end).days > \
            forecast_config.validation.embargo_days
        assert (split.test_start - split.valid_end).days > \
            forecast_config.validation.embargo_days

    def test_no_training_target_lands_inside_the_test_window(
        self, horizon_dataset, forecast_config
    ) -> None:
        """The assertion the embargo exists for, stated directly.

        Checks actual rows rather than the boundary arithmetic, so it would
        catch a split that is correct on paper but applied wrongly.
        """
        frame = horizon_dataset.frame
        split = build_origin_split(frame, forecast_config)

        train = slice_fold(frame, split.train_start, split.train_end)
        test = slice_fold(frame, split.test_start, split.test_end)

        latest_training_target = pd.to_datetime(train[TARGET_DATE]).max()
        earliest_test_origin = pd.to_datetime(test[ORIGIN_DATE]).min()

        assert latest_training_target < earliest_test_origin, (
            f"a training row's outcome ({latest_training_target.date()}) falls "
            f"at or after the first test origin ({earliest_test_origin.date()}) "
            f"- the model is being scored on data it was fitted on"
        )

    def test_leakage_gap_is_positive_for_every_evaluation_fold(
        self, horizon_dataset, forecast_config
    ) -> None:
        frame = horizon_dataset.frame
        split = build_origin_split(frame, forecast_config)
        train = slice_fold(frame, split.train_start, split.train_end)

        for name, (start, end) in {
            "calibration": (split.calibration_start, split.calibration_end),
            "validation": (split.valid_start, split.valid_end),
            "test": (split.test_start, split.test_end),
        }.items():
            fold = slice_fold(frame, start, end)

            assert leakage_gap_days(train, fold) > 0, f"{name} fold overlaps training targets"

    def test_the_structural_worst_case_is_safe(
        self, horizon_dataset, forecast_config
    ) -> None:
        """A training origin drawing the *longest* horizon must still land clear
        of the test window.

        Stronger than checking realised targets, which depend on which horizons
        the random draw happened to produce. At smoke scale only a handful of
        horizons are drawn per origin, so the worst case is frequently not
        sampled - a realised-only check would pass on an unsound design.
        """
        split = build_origin_split(horizon_dataset.frame, forecast_config)

        assert worst_case_gap_days(split, forecast_config.max_horizon) > 0

    def test_removing_the_embargo_would_break_the_guarantee(
        self, horizon_dataset, forecast_config
    ) -> None:
        """Proves the embargo is doing work rather than being decorative.

        With the gap set to zero the split still builds and still looks
        reasonable, but a training origin at the boundary can reach 90 days into
        the test window. A suite that only checked the embargoed case could not
        distinguish a working embargo from an ignored one.
        """
        frame = horizon_dataset.frame
        no_embargo = forecast_config.model_copy(
            update={
                "validation": forecast_config.validation.model_copy(
                    update={"embargo_days": 0}
                )
            }
        )
        # Bypass the config validator, which rightly refuses embargo < max_horizon.
        object.__setattr__(no_embargo.validation, "embargo_days", 0)

        split = build_origin_split(frame, no_embargo)

        assert worst_case_gap_days(split, no_embargo.max_horizon) <= 0, (
            "with no embargo a boundary training origin should be able to reach "
            "into the test window; if it cannot, the embargo tests above prove "
            "nothing"
        )


class TestFoldSlicing:
    def test_slicing_selects_on_origin_not_target(self, horizon_dataset) -> None:
        """Rows sharing an origin see the same history and differ only in
        horizon step, so they are not independent. Splitting by row would
        scatter one origin's rows across train and test."""
        frame = horizon_dataset.frame
        origins = pd.to_datetime(frame[ORIGIN_DATE]).dt.date
        start, end = origins.min(), origins.min() + timedelta(days=60)

        fold = slice_fold(frame, start, end)
        fold_origins = pd.to_datetime(fold[ORIGIN_DATE]).dt.date

        assert fold_origins.min() >= start
        assert fold_origins.max() <= end
        # Targets legitimately reach past the window - that is what a horizon is.
        assert pd.to_datetime(fold[TARGET_DATE]).dt.date.max() > end

    def test_every_row_of_an_origin_stays_together(self, horizon_dataset) -> None:
        frame = horizon_dataset.frame
        origins = pd.to_datetime(frame[ORIGIN_DATE]).dt.date
        cutoff = origins.min() + timedelta(days=60)

        early = set(pd.to_datetime(slice_fold(frame, origins.min(), cutoff)[ORIGIN_DATE]))
        late = set(
            pd.to_datetime(
                slice_fold(frame, cutoff + timedelta(days=1), origins.max())[ORIGIN_DATE]
            )
        )

        assert not (early & late)
