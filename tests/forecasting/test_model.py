"""Model-level behaviour: training, prediction, shape and determinism.

These were previously covered only indirectly - a training failure would fail
the service tests, but nothing asserted the model's own contract. That is a real
gap: an indirect test tells you *something* broke, not what, and it cannot
distinguish "the estimator is wrong" from "the service wiring is wrong".

Determinism gets the most attention here because the whole comparison table rests
on it. A 0.4-point gap between two candidates means nothing if re-running would
reorder them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.forecasting.exceptions import InvalidForecastRequestError
from ml.forecasting.split import slice_fold
from ml.forecasting.train import build_estimator, train_forecaster

pytestmark = pytest.mark.models

CANDIDATES = ("horizon_naive", "horizon_seasonal_naive", "lightgbm", "xgboost")


class TestTraining:
    @pytest.mark.parametrize("name", CANDIDATES)
    def test_every_candidate_trains(
        self, name, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        trained = train_forecaster(
            benchmark_dataset, build_estimator(name, seed=42), forecast_config, forecast_split
        )

        assert trained.estimator.is_fitted
        assert trained.name == name

    @pytest.mark.parametrize("name", CANDIDATES)
    def test_every_candidate_records_its_metrics(
        self, name, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        trained = train_forecaster(
            benchmark_dataset, build_estimator(name, seed=42), forecast_config, forecast_split
        )

        assert "test" in trained.metrics
        assert trained.metrics["test"].n > 0
        assert trained.bucket_metrics

    def test_unknown_estimator_is_refused_with_the_available_set(self) -> None:
        """Naming what *is* available turns a typo into a one-line fix."""
        with pytest.raises(InvalidForecastRequestError) as caught:
            build_estimator("not_a_model")

        assert "available" in caught.value.detail

    def test_tuned_parameters_reach_the_estimator(self) -> None:
        estimator = build_estimator("lightgbm", seed=42, params={"num_leaves": 17})

        assert estimator.params["num_leaves"] == 17

    def test_parameters_are_ignored_by_estimators_that_take_none(self) -> None:
        """The naive benchmarks have nothing to tune.

        Refusing the argument would force every caller to special-case them,
        which is how a tuning loop grows an `if name == ...` ladder.
        """
        estimator = build_estimator("horizon_naive", seed=42, params={"num_leaves": 17})

        assert estimator.is_fitted is False


class TestPrediction:
    @pytest.fixture(scope="class")
    def fitted(self, benchmark_dataset, forecast_config, forecast_split):
        return train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=42),
            forecast_config,
            forecast_split,
        )

    def test_output_length_equals_input_rows(
        self, fitted, benchmark_dataset, forecast_split
    ) -> None:
        """The assertion that catches a silent row drop.

        A merge that loses rows inside the predict path would otherwise show up
        only as a confusing length mismatch much further downstream.
        """
        rows = slice_fold(
            benchmark_dataset.frame, forecast_split.test_start, forecast_split.test_end
        ).head(500)

        predictions = fitted.predict(rows)

        assert len(predictions) == len(rows)

    def test_predictions_are_never_negative(
        self, fitted, benchmark_dataset, forecast_split
    ) -> None:
        """Demand cannot be negative, and a negative forecast propagates into
        nonsense downstream rather than failing loudly."""
        rows = slice_fold(
            benchmark_dataset.frame, forecast_split.test_start, forecast_split.test_end
        )

        assert (fitted.predict(rows) >= 0).all()

    def test_predictions_are_finite(self, fitted, benchmark_dataset, forecast_split) -> None:
        rows = slice_fold(
            benchmark_dataset.frame, forecast_split.test_start, forecast_split.test_end
        )

        assert np.isfinite(fitted.predict(rows)).all()

    def test_predictions_vary(self, fitted, benchmark_dataset, forecast_split) -> None:
        """A model that has collapsed to its global mean still produces
        plausible aggregate totals. The only visible symptom is a spread that
        has vanished."""
        rows = slice_fold(
            benchmark_dataset.frame, forecast_split.test_start, forecast_split.test_end
        )

        assert pd.Series(fitted.predict(rows)).std() > 1.0

    def test_a_missing_feature_is_refused_not_guessed(self, fitted, benchmark_dataset) -> None:
        """Silently filling a missing feature would hand back a number computed
        from something other than what the model learned."""
        from ml.forecasting.exceptions import FeatureGenerationError

        rows = benchmark_dataset.frame.head(20).drop(columns=["lag_7_units"])

        with pytest.raises(FeatureGenerationError):
            fitted.predict(rows)


class TestDeterminism:
    def test_the_same_seed_gives_identical_predictions(
        self, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        """The property the whole comparison table rests on.

        Without it, a 0.4-point gap between two candidates is indistinguishable
        from run-to-run noise, and the selection is arbitrary.
        """
        rows = slice_fold(
            benchmark_dataset.frame, forecast_split.test_start, forecast_split.test_end
        ).head(300)

        first = train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=7),
            forecast_config,
            forecast_split,
        ).predict(rows)
        second = train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=7),
            forecast_config,
            forecast_split,
        ).predict(rows)

        np.testing.assert_allclose(first, second)

    def test_categorical_encodings_are_frozen_at_fit_time(
        self, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        """Regression test for a real defect.

        Inferring categorical dtypes per frame meant a level appearing only in a
        later fold became an unseen category (XGBoost raised) or a silently
        different integer code (LightGBM did not, which is worse).
        """
        trained = train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=42),
            forecast_config,
            forecast_split,
        )

        assert trained.categories
        for dtype in trained.categories.values():
            # Levels are normalised to strings; a boolean category index is
            # rejected outright by XGBoost.
            assert all(isinstance(level, str) for level in dtype.categories)


class TestIntervals:
    def test_a_calibration_is_produced(
        self, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        trained = train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=42),
            forecast_config,
            forecast_split,
        )

        assert trained.calibration is not None
        assert trained.calibration.nominal_coverage == pytest.approx(
            1 - forecast_config.intervals.alpha
        )

    def test_bucket_or_pooled_calibration_covers_every_horizon(
        self, benchmark_dataset, forecast_config, forecast_split
    ) -> None:
        """Every horizon step must resolve to *some* calibration.

        A bucket with too few points falls back to the pooled quantile rather
        than returning nothing - an uncalibrated interval is worse than a
        slightly mis-bucketed one.
        """
        trained = train_forecaster(
            benchmark_dataset,
            build_estimator("lightgbm", seed=42),
            forecast_config,
            forecast_split,
        )

        for step in (1, 7, 28, 30, 60, 90):
            assert trained.calibration.for_step(step, forecast_config) is not None
