"""Leakage tests for the model's own feature matrix (extends Step 3's suite).

Step 3 proved the *feature engineer* cannot see the future. This module proves
the property survives the trip into the model - which is a separate claim, and
the one that actually matters, because leakage introduced between the panel and
the fitted estimator would be invisible to Step 3's tests and would produce
excellent metrics.

Leakage is the failure mode with the worst signal-to-noise in all of applied ML:
it never raises, never warns, and makes every number look *better*. The only
defence is to assert the absence of something, repeatedly, from several angles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline.evaluation import compute_metrics
from ml.baseline.models import build_estimator
from ml.baseline.training import (
    EXCLUDED_FROM_FEATURES,
    TARGET,
    PromotionApproach,
    build_temporal_split,
    prepare_training_rows,
    select_features,
    train_baseline,
)

pytestmark = [pytest.mark.models, pytest.mark.leakage]


class TestFeatureMatrixHygiene:
    def test_target_derived_columns_are_excluded(self) -> None:
        """``revenue = units x price`` recovers the target exactly.

        This is the leak that looks most innocent in a column list and is most
        catastrophic in practice - a model given revenue achieves near-perfect
        accuracy and has learned arithmetic, not demand.
        """
        for column in ("revenue", "cost", "gross_profit", "units_uncensored"):
            assert column in EXCLUDED_FROM_FEATURES

    def test_the_target_itself_is_excluded(self) -> None:
        assert TARGET in EXCLUDED_FROM_FEATURES

    def test_no_feature_correlates_almost_perfectly_with_the_target(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """A blanket check that does not depend on knowing the column names.

        The named exclusions above only catch leaks someone anticipated. This
        catches a *new* column added in a later step that happens to encode the
        target, which is how leaks actually get introduced.

        Lagged features legitimately correlate strongly with today's sales, so
        the threshold is set above that - only a near-identity relationship
        fails.
        """
        features = select_features(feature_panel, approach=PromotionApproach.CONTROL)
        numeric = feature_panel[features].select_dtypes(include="number")
        target = feature_panel[TARGET]

        correlations = numeric.corrwith(target).abs().dropna()
        suspicious = correlations[correlations > 0.98]

        assert suspicious.empty, (
            f"these features are nearly identical to the target and probably "
            f"leak it: {dict(suspicious)}"
        )


class TestTemporalIsolation:
    def test_no_training_row_falls_inside_an_evaluation_fold(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """The split is only as good as what the trainer actually honours."""
        split = build_temporal_split(feature_panel)
        dates = pd.to_datetime(feature_panel["date"]).dt.date

        train = feature_panel[dates <= split.train_end]
        train_dates = pd.to_datetime(train["date"]).dt.date

        assert train_dates.max() < split.calibration_start
        assert train_dates.max() < split.valid_start
        assert train_dates.max() < split.test_start

    def test_lag_features_never_reference_a_future_row(
        self, synthetic_panel: pd.DataFrame
    ) -> None:
        """Verified directly against the source series.

        For a sampled product-store pair, the value of ``lag_7_units`` on day D
        must equal the units recorded on day D-7. Reconstructing it by hand is
        the only way to be sure the shift ran within the group and in the right
        direction - an off-by-one or a cross-series bleed would still produce a
        plausible-looking column.
        """
        series = synthetic_panel[
            (synthetic_panel["product_id"] == "P001")
            & (synthetic_panel["store_id"] == "S001")
        ].sort_values("date").reset_index(drop=True)

        for position in (400, 500, 600):
            expected = series.loc[position - 7, "units"]
            actual = series.loc[position, "lag_7_units"]

            assert actual == pytest.approx(expected), (
                f"lag_7 at row {position} is {actual}, but units 7 rows earlier "
                f"were {expected}"
            )

    def test_rolling_means_exclude_the_current_row(
        self, synthetic_panel: pd.DataFrame
    ) -> None:
        """The subtlest leak in the whole feature set.

        A rolling mean that includes today mixes a fraction of the target into
        a feature. With a 28-day window that is only 1/28th of the answer - large
        enough to inflate every metric, small enough that nobody notices.
        """
        series = synthetic_panel[
            (synthetic_panel["product_id"] == "P002")
            & (synthetic_panel["store_id"] == "S000")
        ].sort_values("date").reset_index(drop=True)

        for position in (400, 500, 600):
            window = series.loc[position - 28 : position - 1, "units"]
            expected = window.mean()
            actual = series.loc[position, "rolling_28_units"]

            assert actual == pytest.approx(expected, rel=1e-6)

    def test_early_rows_have_no_seasonal_lag(
        self, synthetic_panel: pd.DataFrame
    ) -> None:
        """The first year cannot have a value from the year before it.

        A fully populated ``lag_364`` at the start of history would mean the
        column was back-filled or wrapped - both of which import future data.
        """
        series = synthetic_panel[
            (synthetic_panel["product_id"] == "P000")
            & (synthetic_panel["store_id"] == "S000")
        ].sort_values("date").reset_index(drop=True)

        assert series.loc[:363, "lag_364_units"].isna().all()
        assert series.loc[364:, "lag_364_units"].notna().all()


class TestLeakageDetection:
    def test_a_planted_leak_is_caught_by_the_accuracy_check(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """Proves the suite can actually detect leakage.

        A test that only ever passes is indistinguishable from a test that does
        nothing. Here a leaking column is deliberately added and the model
        becomes near-perfect - confirming both that leakage produces exactly the
        suspiciously-good metrics described, and that this suite would notice.
        """
        split = build_temporal_split(feature_panel)
        leaky = feature_panel.copy()
        leaky["disguised_revenue"] = leaky["units"] * 4.5

        trained = train_baseline(
            leaky,
            build_estimator("lightgbm", seed=7),
            approach=PromotionApproach.EXCLUDE,
            split=split,
        )

        dates = pd.to_datetime(leaky["date"]).dt.date
        test_rows = leaky[(dates >= split.test_start) & (dates <= split.test_end)]
        clean, _ = prepare_training_rows(test_rows, approach=PromotionApproach.EXCLUDE)
        metrics = compute_metrics(
            clean["units"], pd.Series(trained.predict_baseline(clean), index=clean.index)
        )

        assert metrics.wmape < 0.05, (
            "a planted leak did not produce implausible accuracy, so this "
            "suite's ability to detect leakage is unproven"
        )

    def test_the_honest_model_is_nowhere_near_that_accurate(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """The other half of the comparison above.

        The fixture adds 10% multiplicative noise, so a legitimate model cannot
        approach the leaking one. If this ever starts passing at leak-like
        accuracy, something has gone wrong with the feature set.
        """
        split = build_temporal_split(feature_panel)
        trained = train_baseline(
            feature_panel,
            build_estimator("lightgbm", seed=7),
            approach=PromotionApproach.EXCLUDE,
            split=split,
        )

        dates = pd.to_datetime(feature_panel["date"]).dt.date
        test_rows = feature_panel[(dates >= split.test_start) & (dates <= split.test_end)]
        clean, _ = prepare_training_rows(test_rows, approach=PromotionApproach.EXCLUDE)
        metrics = compute_metrics(
            clean["units"], pd.Series(trained.predict_baseline(clean), index=clean.index)
        )

        assert metrics.wmape > 0.05, (
            f"WMAPE of {metrics.wmape:.1%} on noisy data is too good to be "
            f"honest - suspect leakage"
        )


class TestPredictionIndependence:
    def test_shuffling_row_order_does_not_change_predictions(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """Each prediction must depend only on its own row.

        If order matters, some state is leaking between rows - and in a panel
        sorted by date that state comes from the future. This also guards a real
        deployment concern: batch composition must not change a result.
        """
        split = build_temporal_split(feature_panel)
        trained = train_baseline(
            feature_panel,
            build_estimator("lightgbm", seed=7),
            approach=PromotionApproach.EXCLUDE,
            split=split,
        )

        dates = pd.to_datetime(feature_panel["date"]).dt.date
        sample = feature_panel[dates >= split.test_start].head(500)
        shuffled = sample.sample(frac=1.0, random_state=3)

        ordered = pd.Series(trained.predict_baseline(sample), index=sample.index)
        reordered = pd.Series(trained.predict_baseline(shuffled), index=shuffled.index)

        np.testing.assert_allclose(
            ordered.to_numpy(), reordered.reindex(sample.index).to_numpy(), rtol=1e-9
        )
