"""The horizon dataset: origin/target arithmetic and feature placement.

These are the tests written immediately after ``dataset.py`` and before anything
was built on top of it, because an off-by-``h`` here does not raise. It produces
a dataset that trains fine, scores well, and is measuring the wrong thing - the
model would be reading tomorrow's calendar as today's, or sourcing "history"
from the future.

The arithmetic tests reconstruct the expected value **by hand from the source
panel** rather than by re-running the same helper that built the column. A test
that calls the implementation to compute its own expectation cannot detect a bug
in that implementation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.baseline.training import SUPPLY_FEATURES
from ml.forecasting.dataset import (
    FORECAST_EXCLUDED,
    HORIZON_STEP,
    ORIGIN_DATE,
    TARGET,
    TARGET_DATE,
    TARGET_PREFIX,
    build_horizon_dataset,
)

pytestmark = pytest.mark.models


class TestHorizonArithmetic:
    """The four assertions that catch a mis-joined self-join."""

    def test_target_date_is_origin_plus_horizon(self, horizon_dataset) -> None:
        frame = horizon_dataset.frame

        expected = frame[ORIGIN_DATE] + pd.to_timedelta(frame[HORIZON_STEP], unit="D")

        assert (frame[TARGET_DATE] == expected).all()

    def test_target_is_always_strictly_after_origin(self, horizon_dataset) -> None:
        """h >= 1 always. A row with h=0 would be a nowcast wearing a
        forecast's clothes, and would flatter every metric."""
        frame = horizon_dataset.frame

        assert (frame[TARGET_DATE] > frame[ORIGIN_DATE]).all()
        assert frame[HORIZON_STEP].min() >= 1

    def test_horizon_never_exceeds_the_configured_maximum(
        self, horizon_dataset, forecast_config
    ) -> None:
        assert horizon_dataset.frame[HORIZON_STEP].max() <= forecast_config.max_horizon

    def test_target_value_matches_the_source_panel_at_the_target_date(
        self, horizon_dataset, forecast_history
    ) -> None:
        """`y` is units at (series, origin + h), reconstructed by hand.

        The single most important assertion in the file. If the join were on the
        origin date instead, this fails immediately; every other test in the
        suite would keep passing.
        """
        panel = forecast_history.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        lookup = panel.set_index(["product_id", "store_id", "date"])[TARGET]

        sample = horizon_dataset.frame.sample(n=200, random_state=7)
        for row in sample.itertuples():
            key = (row.product_id, row.store_id, getattr(row, TARGET_DATE))
            expected = lookup.get(key)

            assert getattr(row, TARGET) == pytest.approx(expected), (
                f"target for {row.product_id}/{row.store_id} at h={getattr(row, HORIZON_STEP)} "
                f"does not match the panel at {getattr(row, TARGET_DATE)}"
            )

    def test_lag_features_are_measured_from_the_origin_not_the_target(
        self, horizon_dataset, forecast_history
    ) -> None:
        """``lag_7_units`` must be units at ``origin - 7``.

        The signature bug of this design is sourcing origin-side features from
        the target row, which would make this ``origin + h - 7`` instead. That
        version of the dataset produces a model that appears to forecast
        brilliantly at every horizon - which is exactly what the horizon
        monotonicity test elsewhere is designed to catch behaviourally.
        """
        panel = forecast_history.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        lookup = panel.set_index(["product_id", "store_id", "date"])[TARGET]

        sample = horizon_dataset.frame.sample(n=200, random_state=11)
        checked = 0
        for row in sample.itertuples():
            lag_value = getattr(row, "lag_7_units", None)
            if lag_value is None or pd.isna(lag_value):
                continue
            source_date = getattr(row, ORIGIN_DATE) - pd.Timedelta(days=7)
            expected = lookup.get((row.product_id, row.store_id, source_date))
            if expected is None or pd.isna(expected):
                continue

            assert lag_value == pytest.approx(expected), (
                f"lag_7_units is measured from the wrong date for "
                f"{row.product_id}/{row.store_id} at h={getattr(row, HORIZON_STEP)}"
            )
            checked += 1

        assert checked > 50, "too few comparable rows to conclude anything"

    def test_target_side_calendar_comes_from_the_target_date(
        self, horizon_dataset, forecast_view
    ) -> None:
        """``h_holiday_flag`` must describe ``origin + h``, not the origin.

        Getting this backwards is subtle and expensive: the model would learn to
        raise its forecast on the day a promotion *starts being planned* rather
        than the day it runs.
        """
        calendar = forecast_view.get_calendar()
        calendar["date"] = pd.to_datetime(calendar["date"])
        holidays = calendar.set_index("date")["holiday_flag"].astype(bool)

        column = f"{TARGET_PREFIX}holiday_flag"
        frame = horizon_dataset.frame
        assert column in frame.columns

        sample = frame.sample(n=300, random_state=13)
        mismatches = 0
        for row in sample.itertuples():
            expected = holidays.get(getattr(row, TARGET_DATE))
            if expected is None:
                continue
            if bool(getattr(row, column)) != bool(expected):
                mismatches += 1

        assert mismatches == 0, (
            f"{mismatches} rows carry a holiday flag that does not match their "
            f"target date - the target-side join is using the wrong date"
        )


class TestFeaturePlacement:
    def test_no_supply_feature_reaches_the_model(self, horizon_dataset) -> None:
        """Step 4 measured this: with inventory features, the model recovered
        only 0.30 of true demand during stockouts because it had learned that
        low stock predicts low sales. The forecast is a *demand* forecast, so
        the same exclusion holds."""
        leaked = set(horizon_dataset.feature_names) & SUPPLY_FEATURES

        assert not leaked, f"supply columns leaked into the feature set: {leaked}"

    def test_the_target_is_never_a_feature(self, horizon_dataset) -> None:
        assert TARGET not in horizon_dataset.feature_names

    def test_excluded_columns_are_absent(self, horizon_dataset) -> None:
        leaked = set(horizon_dataset.feature_names) & FORECAST_EXCLUDED

        assert not leaked, f"excluded columns present as features: {leaked}"

    def test_hive_partition_key_is_not_a_feature(self, horizon_dataset) -> None:
        """`part` is a storage-layout artifact, not data.

        A tree splits on it happily as a coarse date proxy, which leaks calendar
        position into the origin side - and it cannot be reproduced for a future
        date, which has no partition yet.
        """
        assert "part" not in horizon_dataset.feature_names

    def test_time_index_is_excluded(self, horizon_dataset) -> None:
        """`time_index` is anchored to the frame's own minimum, so the same
        calendar date gets a different value at training and serving time. No
        leakage test on the training frame would ever reveal that."""
        assert "time_index" not in horizon_dataset.feature_names
        assert f"{TARGET_PREFIX}time_index" not in horizon_dataset.feature_names

    def test_year_is_excluded_on_both_sides(self, horizon_dataset) -> None:
        """A year is either one the model has seen, and it overfits, or one it
        has not, and it cannot place it. Neither is useful."""
        for column in ("year", "financial_year"):
            assert column not in horizon_dataset.feature_names
            assert f"{TARGET_PREFIX}{column}" not in horizon_dataset.feature_names

    def test_both_sides_are_populated(self, horizon_dataset) -> None:
        """The design depends on having both; an empty side means a join failed
        silently and left the model with half the information."""
        target_side = [c for c in horizon_dataset.feature_names if c.startswith(TARGET_PREFIX)]
        origin_side = [
            c for c in horizon_dataset.feature_names if not c.startswith(TARGET_PREFIX)
        ]

        assert len(target_side) >= 10
        assert len(origin_side) >= 20

    def test_horizon_step_is_a_feature(self, horizon_dataset) -> None:
        """The whole reason one model can span every horizon."""
        assert HORIZON_STEP in horizon_dataset.feature_names

    def test_demand_history_is_present(self, horizon_dataset) -> None:
        names = set(horizon_dataset.feature_names)

        assert {"lag_7_units", "lag_364_units", "rolling_28_units"} <= names

    def test_planned_promotion_is_on_the_target_side(self, horizon_dataset) -> None:
        """Promotions are KNOWN_IN_ADVANCE, so the schedule for the day being
        forecast is legitimately available and is a major demand driver."""
        assert f"{TARGET_PREFIX}promotion_flag" in horizon_dataset.feature_names


class TestCensoredTargets:
    def test_stockout_targets_are_excluded(self, horizon_dataset) -> None:
        """Approach A. On a stockout day the target records what was available
        to sell, not what customers wanted."""
        assert horizon_dataset.excluded.get("stockout_targets", 0) > 0

    def test_exclusion_counts_are_reported(self, horizon_dataset) -> None:
        """The cost of every filter must be visible, not inferred."""
        excluded = horizon_dataset.excluded

        assert excluded["panel_rows"] > 0
        assert excluded["retained"] > 0
        assert excluded["retained"] <= excluded["panel_rows"]

    def test_stockout_origins_are_kept(
        self, forecast_history, forecast_view, forecast_config, forecast_sample
    ) -> None:
        """A stockout at the *origin* is a legitimate knowable state.

        Dropping those origins would bias the feature distribution for no gain -
        the target is not corrupted there, only the origin's own sales are, and
        those enter as history rather than as the thing being predicted.
        """
        assert forecast_config.target_handling.exclude_stockout_origins is False

        dataset = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample
        )
        origins_kept = dataset.excluded["origins"]

        stricter = forecast_config.model_copy(
            update={
                "target_handling": forecast_config.target_handling.model_copy(
                    update={"exclude_stockout_origins": True}
                )
            }
        )
        fewer = build_horizon_dataset(
            forecast_history, forecast_view, stricter, forecast_sample
        )

        assert fewer.excluded["origins"] < origins_kept


class TestDeterminism:
    def test_same_seed_produces_the_same_dataset(
        self, forecast_history, forecast_view, forecast_config, forecast_sample
    ) -> None:
        """Horizon steps are drawn at random, so reproducibility is not free.

        Without this, two runs of the comparison are not comparable and a 0.4%
        gap between candidates means nothing.
        """
        first = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=99
        )
        second = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=99
        )

        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_different_seeds_draw_different_horizons(
        self, forecast_history, forecast_view, forecast_config, forecast_sample
    ) -> None:
        first = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=1
        )
        second = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=2
        )

        assert not first.frame[HORIZON_STEP].equals(second.frame[HORIZON_STEP])

    def test_horizon_steps_span_the_full_range(self, horizon_dataset) -> None:
        """Random draws rather than a fixed grid.

        A fixed grid makes the model's splits on ``horizon_step``
        piecewise-constant, which shows up as a visible staircase in the daily
        forecast path - and that path is a deliverable, not an internal detail.
        """
        distinct = horizon_dataset.frame[HORIZON_STEP].nunique()

        assert distinct > 50, (
            f"only {distinct} distinct horizon steps; the model will interpolate "
            f"between them as a step function"
        )
