"""Hyperparameter search.

Two properties matter more than whether the search finds anything.

**It must never read the test fold.** Tuning against test makes every number
reported afterwards a self-report, and the failure is undetectable from the
outside - the metrics simply come out better.

**It must decline to adopt a marginal winner.** The model sits near the
irreducible noise floor, so a search will usually find something nominally
better than the defaults purely by chance. Adopting it means adopting noise, and
makes the next run's comparison harder to interpret rather than easier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.forecasting.split import slice_fold
from ml.forecasting.train import build_estimator
from ml.forecasting.tuning import Trial, TuningResult, best_params, sample_params, tune

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def small_space() -> dict[str, list]:
    """A deliberately tiny space - these tests check mechanics, not search."""
    return {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}


class TestSampling:
    def test_a_draw_takes_one_value_per_parameter(self, small_space) -> None:
        params = sample_params(small_space, np.random.default_rng(0))

        assert set(params) == set(small_space)
        for name, value in params.items():
            assert value in small_space[name]

    def test_the_same_seed_draws_the_same_configuration(self, small_space) -> None:
        first = sample_params(small_space, np.random.default_rng(11))
        second = sample_params(small_space, np.random.default_rng(11))

        assert first == second


class TestSearch:
    @pytest.fixture(scope="class")
    def result(self, benchmark_dataset, forecast_config, forecast_split, small_space):
        return tune(
            benchmark_dataset,
            forecast_split,
            forecast_config,
            lambda params: build_estimator("lightgbm", seed=42, params=params or None),
            space=small_space,
            n_trials=3,
            seed=5,
        )

    def test_records_every_trial(self, result) -> None:
        assert result.trials
        assert all(np.isfinite(t.wmape) for t in result.trials)

    def test_scores_the_defaults_as_a_reference_point(self, result) -> None:
        """Without it there is nothing to say whether the search found anything -
        only which of its own samples was least bad."""
        assert np.isfinite(result.baseline_wmape)

    def test_the_best_trial_is_the_lowest_wmape(self, result) -> None:
        assert result.best is not None
        assert result.best.wmape == min(t.wmape for t in result.trials)

    def test_the_trial_table_is_sorted_and_carries_the_parameters(self, result) -> None:
        frame = result.to_frame()

        assert list(frame["wmape"]) == sorted(frame["wmape"])
        assert any(c.startswith("hp_") for c in frame.columns)

    def test_the_search_is_reproducible(
        self, benchmark_dataset, forecast_config, forecast_split, small_space
    ) -> None:
        """A search nobody can reproduce is a search nobody can review."""
        runs = [
            tune(
                benchmark_dataset, forecast_split, forecast_config,
                lambda params: build_estimator("lightgbm", seed=42, params=params or None),
                space=small_space, n_trials=2, seed=3,
            )
            for _ in range(2)
        ]

        assert [t.params for t in runs[0].trials] == [t.params for t in runs[1].trials]
        assert [t.wmape for t in runs[0].trials] == [t.wmape for t in runs[1].trials]


class TestTestFoldIsUntouched:
    """The property that keeps every subsequent number honest."""

    def test_tuning_scores_on_validation_not_test(
        self, benchmark_dataset, forecast_config, forecast_split, small_space
    ) -> None:
        """Asserted by construction rather than by inspection.

        A trial's score must be reproducible from the validation fold alone. If
        tuning had scored on test, refitting on train and scoring on validation
        would give a different number.
        """
        result = tune(
            benchmark_dataset, forecast_split, forecast_config,
            lambda params: build_estimator("lightgbm", seed=42, params=params or None),
            space=small_space, n_trials=1, seed=9,
        )
        assert result.best is not None

        from ml.baseline.evaluation import compute_metrics
        from ml.forecasting.dataset import TARGET
        from ml.forecasting.train import _prepare, build_category_dtypes

        frame = benchmark_dataset.frame
        categories = build_category_dtypes(frame, benchmark_dataset.feature_names)
        train = slice_fold(frame, forecast_split.train_start, forecast_split.train_end)
        validation = slice_fold(frame, forecast_split.valid_start, forecast_split.valid_end)

        estimator = build_estimator("lightgbm", seed=42, params=result.best.params)
        estimator.fit(_prepare(train, benchmark_dataset.feature_names, categories), train[TARGET])
        reproduced = compute_metrics(
            validation[TARGET],
            pd.Series(
                estimator.predict(
                    _prepare(validation, benchmark_dataset.feature_names, categories)
                )
            ),
        ).wmape

        assert reproduced == pytest.approx(result.best.wmape, abs=1e-9)


class TestAdoption:
    def test_a_marginal_gain_is_not_adopted(self) -> None:
        """Below fold-to-fold noise, keep the defaults.

        Backtest standard deviation on this data runs 0.3-1.8 points, so a
        smaller "win" is indistinguishable from which fold you happened to look
        at.
        """
        result = TuningResult(
            trials=[Trial(0, {"num_leaves": 31}, wmape=0.4399, mae=1, bias_pct=0, seconds=1)],
            baseline_wmape=0.4400,
        )
        result.best = result.trials[0]

        assert not result.is_material()
        assert best_params(result) == {}

    def test_a_material_gain_is_adopted(self) -> None:
        result = TuningResult(
            trials=[Trial(0, {"num_leaves": 127}, wmape=0.40, mae=1, bias_pct=0, seconds=1)],
            baseline_wmape=0.44,
        )
        result.best = result.trials[0]

        assert result.is_material()
        assert best_params(result) == {"num_leaves": 127}

    def test_a_search_that_loses_to_the_defaults_adopts_nothing(self) -> None:
        """A search finding nothing is a result, not a bug - and it is the
        expected outcome near the noise floor."""
        result = TuningResult(
            trials=[Trial(0, {"num_leaves": 15}, wmape=0.46, mae=1, bias_pct=0, seconds=1)],
            baseline_wmape=0.44,
        )
        result.best = result.trials[0]

        assert result.improvement_pp < 0
        assert not result.is_material()
        assert best_params(result) == {}

    def test_summary_states_the_verdict_either_way(self) -> None:
        result = TuningResult(
            trials=[Trial(0, {}, wmape=0.46, mae=1, bias_pct=0, seconds=1)],
            baseline_wmape=0.44,
        )
        result.best = result.trials[0]

        assert "keep the defaults" in result.summary()

    def test_an_empty_search_is_handled(self) -> None:
        assert best_params(TuningResult()) == {}
        assert "no trials" in TuningResult().summary()
