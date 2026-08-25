"""Split-conformal prediction intervals.

The reason this module exists at all is that a confidence number is the easiest
thing in the system to fabricate. A field called ``confidence: 0.92`` costs
nothing to emit and means nothing, and by Step 17 an agent will be putting
intervals in front of a user who will act on them.

Conformal prediction earns the number instead: it has a distribution-free
finite-sample coverage guarantee, and - the part these tests enforce - the
*achieved* coverage is measured on held-out data and reported whatever it turns
out to be. An interval that claims 90% and delivers 71% is a finding, not a
detail to smooth over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline.conformal import add_intervals, calibrate, measure_coverage

pytestmark = pytest.mark.models


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


@pytest.fixture
def calibration_data(rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    """Predictions and actuals with a known, homoscedastic error scale."""
    predictions = pd.Series(rng.uniform(20, 120, size=4000))
    actuals = predictions + pd.Series(rng.normal(0, 10, size=4000))
    return actuals.clip(lower=0), predictions


class TestCalibration:
    def test_produces_a_positive_width(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        actuals, predictions = calibration_data

        calibration = calibrate(actuals, predictions, alpha=0.1)

        assert calibration.quantile > 0

    def test_lower_alpha_gives_a_wider_interval(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """A 99% interval must be wider than a 90% one. Monotonicity is the
        most basic property an interval can have, and getting the quantile
        direction backwards is an easy mistake that nothing else would catch."""
        actuals, predictions = calibration_data

        wide = calibrate(actuals, predictions, alpha=0.01)
        narrow = calibrate(actuals, predictions, alpha=0.1)

        assert wide.quantile > narrow.quantile

    def test_records_the_nominal_coverage_it_was_built_for(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        actuals, predictions = calibration_data

        calibration = calibrate(actuals, predictions, alpha=0.1)

        assert calibration.nominal_coverage == pytest.approx(0.9)

    def test_empty_calibration_set_raises(self) -> None:
        """Better than returning a zero-width interval that claims 90%."""
        with pytest.raises(ValueError):
            calibrate(pd.Series(dtype=float), pd.Series(dtype=float), alpha=0.1)


class TestCoverage:
    def test_measured_coverage_is_near_nominal(
        self, rng: np.random.Generator, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """The guarantee, verified on data the calibration never saw.

        Exchangeable by construction here - the test draw comes from the same
        distribution as the calibration draw - so this is the case where
        conformal should work essentially perfectly. If it fails here it is
        broken, not merely challenged by a distribution shift.
        """
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)

        test_predictions = pd.Series(rng.uniform(20, 120, size=4000))
        test_actuals = (test_predictions + rng.normal(0, 10, size=4000)).clip(lower=0)

        report = measure_coverage(test_actuals, test_predictions, calibration)

        assert report.empirical == pytest.approx(0.9, abs=0.03)
        assert report.is_calibrated

    def test_under_coverage_is_reported_rather_than_hidden(
        self, rng: np.random.Generator, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """The test that gives the whole module its value.

        Calibrate on quiet data, then evaluate on data three times noisier - a
        regime change the interval cannot possibly cover. The requirement is not
        that coverage holds; it is that the shortfall is *visible*. A system that
        silently reported 90% here would be lying to the user.
        """
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)

        test_predictions = pd.Series(rng.uniform(20, 120, size=4000))
        test_actuals = (test_predictions + rng.normal(0, 30, size=4000)).clip(lower=0)

        report = measure_coverage(test_actuals, test_predictions, calibration)

        assert report.empirical < 0.85
        assert not report.is_calibrated
        assert "%" in report.summary()

    def test_summary_states_the_achieved_number(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)

        report = measure_coverage(actuals, predictions, calibration)

        assert f"{report.empirical:.1%}" in report.summary()


class TestIntervalApplication:
    def test_bounds_bracket_the_point_prediction(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)
        frame = pd.DataFrame({"baseline_units": [10.0, 50.0, 200.0]})

        result = add_intervals(frame, calibration, actual_column=None)

        assert (result["baseline_lower"] <= result["baseline_units"]).all()
        assert (result["baseline_upper"] >= result["baseline_units"]).all()

    def test_lower_bound_is_never_negative(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """Units sold cannot be negative, so a lower bound below zero is not a
        wider interval - it is an invalid one that makes the interval look more
        informative than it is."""
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)
        frame = pd.DataFrame({"baseline_units": [0.0, 1.0, 2.0]})

        result = add_intervals(frame, calibration, actual_column=None)

        assert (result["baseline_lower"] >= 0).all()

    def test_does_not_mutate_the_input(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)
        frame = pd.DataFrame({"baseline_units": [10.0, 50.0]})
        before = frame.copy()

        add_intervals(frame, calibration, actual_column=None)

        pd.testing.assert_frame_equal(frame, before)

    def test_significance_flags_gaps_outside_the_interval(
        self, calibration_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """``is_significant`` must mean "outside the interval", nothing looser.

        This is the flag Step 17's root-cause agent will use to decide whether a
        decline is real. If it were set by any gap at all, every ordinary day
        would look like an incident.
        """
        actuals, predictions = calibration_data
        calibration = calibrate(actuals, predictions, alpha=0.1)
        frame = pd.DataFrame(
            {
                "baseline_units": [100.0, 100.0],
                # One actual sits on the baseline, one far below any plausible
                # interval width.
                "actual_units": [100.0, 1.0],
            }
        )

        result = add_intervals(frame, calibration)

        assert not bool(result["is_significant"].iloc[0])
        assert bool(result["is_significant"].iloc[1])
