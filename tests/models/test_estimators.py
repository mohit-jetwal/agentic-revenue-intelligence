"""The three candidate estimators.

Each implements the same small interface, so most of these tests run against all
three via parametrisation. That is deliberate: the pipeline swaps them freely,
and a contract that only one of them honours is not a contract.

The seasonal naive gets extra attention despite being the simplest. It is the
benchmark the others must beat to justify themselves, so if it is quietly broken
the comparison becomes a walkover and the step's central claim - that complexity
earned its place - would be unfounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline.models import (
    ESTIMATORS,
    LightGBMBaseline,
    SeasonalNaiveBaseline,
    build_estimator,
    permutation_importance,
)
from ml.baseline.training import (
    PromotionApproach,
    build_temporal_split,
    prepare_training_rows,
    select_features,
)

pytestmark = pytest.mark.models

ALL_ESTIMATORS = sorted(ESTIMATORS)


@pytest.fixture(scope="module")
def training_frame(feature_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean training rows and target, prepared once for every estimator."""
    split = build_temporal_split(feature_panel)
    dates = pd.to_datetime(feature_panel["date"]).dt.date
    train = feature_panel[dates <= split.train_end]

    rows, _ = prepare_training_rows(train, approach=PromotionApproach.EXCLUDE)
    features = select_features(feature_panel, approach=PromotionApproach.EXCLUDE)
    return rows[features], rows["units"]


class TestEstimatorContract:
    """Properties every estimator must satisfy, whatever its internals."""

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_fit_returns_self_for_chaining(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = training_frame
        estimator = build_estimator(name, seed=7)

        assert estimator.fit(X, y) is estimator

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_predict_returns_one_value_per_row(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = training_frame
        estimator = build_estimator(name, seed=7).fit(X, y)

        assert estimator.predict(X.head(100)).shape == (100,)

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_predictions_are_never_negative(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Every estimator clips at zero.

        Ridge in particular will happily extrapolate below zero on an unusual
        row, and a negative baseline propagates into a nonsensical negative
        uplift rather than failing loudly.
        """
        X, y = training_frame
        estimator = build_estimator(name, seed=7).fit(X, y)

        assert (estimator.predict(X) >= 0).all()

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_predictions_are_finite(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = training_frame
        estimator = build_estimator(name, seed=7).fit(X, y)

        assert np.isfinite(estimator.predict(X)).all()

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_is_deterministic_under_a_fixed_seed(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Two runs with the same seed must agree exactly.

        Without this, the model comparison cannot be trusted: a 0.4% gap between
        two candidates means nothing if re-running would reorder them.
        """
        X, y = training_frame

        first = build_estimator(name, seed=7).fit(X, y).predict(X.head(500))
        second = build_estimator(name, seed=7).fit(X, y).predict(X.head(500))

        np.testing.assert_allclose(first, second)

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_beats_a_global_mean(
        self, name: str, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """The floor no candidate is allowed to fall below.

        Predicting the training mean for every row is the least informative
        model that still runs. An estimator that cannot beat it is not
        contributing signal, and would be worse than useless in a comparison
        table where it might win on stability.
        """
        X, y = training_frame
        estimator = build_estimator(name, seed=7).fit(X, y)

        model_error = np.abs(y - estimator.predict(X)).mean()
        mean_error = np.abs(y - y.mean()).mean()

        assert model_error < mean_error

    @pytest.mark.parametrize("name", ALL_ESTIMATORS)
    def test_unknown_estimator_name_raises(self, name: str) -> None:
        with pytest.raises(KeyError):
            build_estimator("not_a_real_model", seed=7)


class TestSeasonalNaive:
    def test_uses_last_years_same_weekday_when_available(self) -> None:
        """The benchmark's core idea, isolated from everything else.

        Given a clean ``lag_364`` and a matching rolling mean, the prediction
        should be a blend of the two rather than either extreme - which is what
        makes it a fair benchmark rather than a straw man.
        """
        X = pd.DataFrame({"lag_364_units": [100.0], "rolling_28_units": [100.0]})
        estimator = SeasonalNaiveBaseline().fit(
            X, pd.Series([100.0])
        )

        assert estimator.predict(X)[0] == pytest.approx(100.0, rel=0.05)

    def test_falls_back_when_the_seasonal_lag_is_missing(self) -> None:
        """Cold start must degrade, not crash.

        A product launched six months ago has no value 364 days back. The
        benchmark has to produce something sensible from the rolling mean
        instead, or the comparison would be run on a subset of rows.
        """
        X = pd.DataFrame({"lag_364_units": [np.nan], "rolling_28_units": [80.0]})
        estimator = SeasonalNaiveBaseline().fit(X, pd.Series([80.0]))

        prediction = estimator.predict(X)[0]

        assert np.isfinite(prediction)
        assert prediction == pytest.approx(80.0, rel=0.15)

    def test_falls_back_to_the_global_mean_when_everything_is_missing(self) -> None:
        """The last link in the fallback chain.

        A brand-new product-store pair has no history at all. Returning NaN here
        would propagate into the metrics and silently drop rows.
        """
        train = pd.DataFrame(
            {"lag_364_units": [50.0, 70.0], "rolling_28_units": [50.0, 70.0]}
        )
        estimator = SeasonalNaiveBaseline().fit(train, pd.Series([50.0, 70.0]))

        cold = pd.DataFrame({"lag_364_units": [np.nan], "rolling_28_units": [np.nan]})
        prediction = estimator.predict(cold)[0]

        assert np.isfinite(prediction)
        assert prediction > 0


class TestLightGBM:
    def test_uses_a_poisson_objective(self) -> None:
        """Not a cosmetic choice.

        The alternative - fitting on ``log1p`` and back-transforming with
        ``expm1`` - introduces retransformation bias: by Jensen's inequality
        ``E[exp(X)] != exp(E[X])``, so the back-transformed mean is
        systematically *low*. A baseline biased low manufactures uplift on every
        promotion measured against it, which is precisely the number this
        platform exists to get right.
        """
        assert LightGBMBaseline().params["objective"] == "poisson"

    def test_reports_feature_importance(
        self, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = training_frame
        estimator = LightGBMBaseline(seed=7).fit(X, y)

        importance = estimator.feature_importance()

        assert importance is not None
        assert not importance.empty
        assert (importance["importance"] >= 0).all()

    def test_recovers_the_weekend_effect_the_fixture_built_in(
        self, feature_panel: pd.DataFrame, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """A real recovery test rather than a smoke test.

        The fixture applies a 35% weekend uplift. If the model has learned
        anything about demand at all, its weekend predictions must exceed its
        weekday predictions - and this would catch a feature matrix that had
        silently lost its calendar columns.
        """
        X, y = training_frame
        estimator = LightGBMBaseline(seed=7).fit(X, y)

        predictions = pd.Series(estimator.predict(X), index=X.index)
        weekend = predictions[X["is_weekend"].astype(bool)].mean()
        weekday = predictions[~X["is_weekend"].astype(bool)].mean()

        assert weekend > weekday


class TestPermutationImportance:
    def test_ranks_a_known_driver_above_a_noise_column(
        self, training_frame: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Validates the attribution method itself, not just that it runs.

        A column of pure noise is added and must rank below the rolling mean,
        which the fixture guarantees is genuinely predictive. Chosen over SHAP
        deliberately: SHAP is not a dependency, and at panel scale it costs far
        more than it adds for the question "what drives baseline demand".
        """
        X, y = training_frame
        rng = np.random.default_rng(3)
        X = X.copy()
        X["pure_noise"] = rng.normal(size=len(X))

        estimator = LightGBMBaseline(seed=7).fit(X, y)
        importance = permutation_importance(estimator, X.head(2000), y.head(2000), seed=7)

        ranking = importance.set_index("feature")["importance"]

        assert ranking["rolling_28_units"] > ranking["pure_noise"]
