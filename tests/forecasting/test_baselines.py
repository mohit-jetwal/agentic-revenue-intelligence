"""The naive benchmarks, and the correction that made one of them usable.

These matter more than their simplicity suggests. Forecast Value Added is defined
*against* the seasonal naive, so a weak benchmark makes every model look good and
turns the comparison table into flattery. The brief is explicit that a LightGBM
losing to a seasonal naive must be reported as such - which only means something
if the naive is a fair fight.

The central test here is that the seasonal reference reads from
``target_date - 364``, not ``origin - 364``. Step 4's estimator does the latter,
which for a row at horizon *h* lands ``364 + h`` days before the date being
forecast - wrong weekday, wrong point in the season, and worse as *h* grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.forecasting.baselines import (
    SEASONAL_LAG_DAYS,
    HorizonNaive,
    HorizonSeasonalNaive,
    attach_seasonal_reference,
)

pytestmark = pytest.mark.models


class TestSeasonalReference:
    def test_reference_is_read_from_the_target_date_not_the_origin(self) -> None:
        """The correction over Step 4's estimator, asserted directly.

        Reconstructed by hand from the source frame rather than by re-running
        the helper, so a bug in the helper cannot hide inside its own test.
        """
        history = pd.DataFrame(
            {
                "product_id": "P1",
                "store_id": "S1",
                "date": pd.date_range("2023-01-01", periods=500, freq="D"),
                "units": np.arange(500, dtype=float),
            }
        )
        frame = pd.DataFrame(
            {
                "product_id": ["P1"],
                "store_id": ["S1"],
                "target_date": [pd.Timestamp("2024-03-01")],
            }
        )

        attached = attach_seasonal_reference(frame, history)

        expected_date = pd.Timestamp("2024-03-01") - pd.Timedelta(days=SEASONAL_LAG_DAYS)
        expected = history.loc[history["date"] == expected_date, "units"].iloc[0]

        assert attached["seasonal_reference"].iloc[0] == pytest.approx(expected)

    def test_the_lag_is_364_so_the_weekday_matches(self) -> None:
        """364, not 365. Demand here is far more weekly than annual, so an
        off-by-one-day comparison against a different weekday is worse than
        useless."""
        assert SEASONAL_LAG_DAYS == 364
        assert SEASONAL_LAG_DAYS % 7 == 0

        target = pd.Timestamp("2025-06-15")
        reference = target - pd.Timedelta(days=SEASONAL_LAG_DAYS)
        assert target.day_name() == reference.day_name()

    def test_missing_reference_is_left_null_not_invented(self) -> None:
        """A series with under a year of history has no reference. Filling it
        with a guess would make the benchmark look better than it is, which is
        exactly the failure this benchmark exists to avoid."""
        history = pd.DataFrame(
            {
                "product_id": "P1",
                "store_id": "S1",
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "units": 10.0,
            }
        )
        frame = pd.DataFrame(
            {
                "product_id": ["P1"],
                "store_id": ["S1"],
                "target_date": [pd.Timestamp("2024-02-01")],
            }
        )

        attached = attach_seasonal_reference(frame, history)

        assert attached["seasonal_reference"].isna().all()

    def test_does_not_mutate_the_input(self) -> None:
        history = pd.DataFrame(
            {
                "product_id": "P1", "store_id": "S1",
                "date": pd.date_range("2023-01-01", periods=400, freq="D"),
                "units": 5.0,
            }
        )
        frame = pd.DataFrame(
            {
                "product_id": ["P1"], "store_id": ["S1"],
                "target_date": [pd.Timestamp("2024-01-15")],
            }
        )
        before = frame.copy()

        attach_seasonal_reference(frame, history)

        pd.testing.assert_frame_equal(frame, before)


class TestHorizonNaive:
    def test_carries_the_last_observed_value_forward(self) -> None:
        X = pd.DataFrame({"lag_1_units": [42.0, 17.0]})
        estimator = HorizonNaive().fit(X, pd.Series([42.0, 17.0]))

        np.testing.assert_allclose(estimator.predict(X), [42.0, 17.0])

    def test_falls_back_down_the_chain(self) -> None:
        """A cold-start series has no lag at all. Returning NaN would drop those
        rows from every metric silently, which is how a benchmark comes to look
        better than it is."""
        X = pd.DataFrame({"lag_1_units": [np.nan], "rolling_7_units": [30.0]})
        estimator = HorizonNaive().fit(X, pd.Series([30.0]))

        prediction = estimator.predict(X)[0]

        assert np.isfinite(prediction)
        assert prediction == pytest.approx(30.0)

    def test_falls_back_to_the_global_mean_when_nothing_is_available(self) -> None:
        train = pd.DataFrame({"lag_1_units": [10.0, 30.0]})
        estimator = HorizonNaive().fit(train, pd.Series([10.0, 30.0]))

        cold = pd.DataFrame({"lag_1_units": [np.nan]})
        prediction = estimator.predict(cold)[0]

        assert np.isfinite(prediction)
        assert prediction == pytest.approx(20.0)

    def test_predictions_are_never_negative(self) -> None:
        X = pd.DataFrame({"lag_1_units": [-5.0]})
        estimator = HorizonNaive().fit(X, pd.Series([0.0]))

        assert (estimator.predict(X) >= 0).all()


class TestHorizonSeasonalNaive:
    def test_blends_the_seasonal_reference_with_the_recent_level(self) -> None:
        """Pure seasonal naive inherits the full random error of one day a year
        ago. Blending with a rolling mean is both standard and a materially
        stronger benchmark - which is the point, since FVA is measured against
        it."""
        X = pd.DataFrame({"seasonal_reference": [100.0], "rolling_28_units": [50.0]})
        estimator = HorizonSeasonalNaive(seasonal_weight=0.5).fit(X, pd.Series([75.0]))

        assert estimator.predict(X)[0] == pytest.approx(75.0)

    def test_degrades_to_the_recent_level_without_a_reference(self) -> None:
        """A benchmark that raises is a benchmark nobody runs."""
        X = pd.DataFrame({"rolling_28_units": [60.0]})
        estimator = HorizonSeasonalNaive().fit(X, pd.Series([60.0]))

        prediction = estimator.predict(X)[0]

        assert np.isfinite(prediction)
        assert prediction == pytest.approx(60.0)

    def test_rows_without_a_reference_still_get_a_value(self) -> None:
        X = pd.DataFrame(
            {
                "seasonal_reference": [100.0, np.nan],
                "rolling_28_units": [50.0, 40.0],
            }
        )
        estimator = HorizonSeasonalNaive(seasonal_weight=0.5).fit(
            X, pd.Series([75.0, 40.0])
        )

        predictions = estimator.predict(X)

        assert np.isfinite(predictions).all()
        assert predictions[1] == pytest.approx(40.0)

    def test_seasonal_weight_is_recorded_in_params(self) -> None:
        """Reproducibility: the weight changes the benchmark, so it has to reach
        MLflow with everything else."""
        estimator = HorizonSeasonalNaive(seasonal_weight=0.7)

        assert estimator.get_params()["seasonal_weight"] == 0.7

    def test_beats_the_plain_naive_on_seasonal_data(self, benchmark_dataset,
                                                    forecast_config, forecast_split) -> None:
        """The benchmark has to be worth beating.

        If the seasonal naive were no better than carrying the last value
        forward, FVA measured against it would be meaningless.
        """
        from ml.forecasting.train import build_estimator, train_forecaster

        naive = train_forecaster(
            benchmark_dataset, build_estimator("horizon_naive", seed=42),
            forecast_config, forecast_split,
        )
        seasonal = train_forecaster(
            benchmark_dataset, build_estimator("horizon_seasonal_naive", seed=42),
            forecast_config, forecast_split,
        )

        assert seasonal.metrics["test"].wmape < naive.metrics["test"].wmape
