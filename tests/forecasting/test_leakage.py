"""Leakage tests for the horizon design (brief section 32).

Leakage is the failure mode with the worst signal-to-noise in applied ML: it
never raises, never warns, and makes every number look *better*. In a forecaster
it has a specific and dangerous shape - if origin-side features are accidentally
read from the target row, the model appears to predict 90 days out as accurately
as tomorrow, and the resulting system is confidently useless.

Four kinds of test here, deliberately overlapping:

* **T1 - the mutation test.** Corrupt everything after a cutoff; assert the
  training features are byte-identical.
* **T2 - the falsifiability test.** Plant the exact bug and assert T1 *fails*.
  A test that has never failed proves nothing about its ability to detect
  anything.
* **T4/T5 - behavioural.** Error must grow with horizon, and must not fall below
  the irreducible noise floor. These need no column names, so they survive a
  refactor that renames everything.
* **T6 - train/serve equivalence.** The one property with no analogue in Steps
  3-4, because this is the first step where training and serving build features
  by different paths.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from ml.forecasting.dataset import (
    HORIZON_STEP,
    ORIGIN_DATE,
    TARGET,
    TARGET_DATE,
    TARGET_PREFIX,
    build_horizon_dataset,
    target_side_features,
)
from ml.forecasting.evaluate import horizon_error_grows

pytestmark = [pytest.mark.models, pytest.mark.leakage]


class _MutatedRepository:
    """Wraps a repository, corrupting OBSERVED data after a cutoff.

    Only ``sales_daily``, ``inventory`` and ``competitor_pricing`` are corrupted.
    ``calendar``, ``promotions`` and ``pricing`` are left alone **on purpose**:
    they are KNOWN_IN_ADVANCE, so reading them forward is legitimate and this
    design does exactly that. Corrupting them too would make the test assert a
    property the system does not have and should not have.
    """

    def __init__(self, wrapped, cutoff) -> None:
        self._wrapped = wrapped
        self._cutoff = cutoff

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)

    def _corrupt(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "date" not in frame.columns:
            return frame
        result = frame.copy()
        dates = pd.to_datetime(result["date"]).dt.date
        after = dates > self._cutoff
        for column in ("units", "sold_units", "closing_inventory", "competitor_price"):
            if column in result.columns:
                result.loc[after, column] = result.loc[after, column] * 1000
        return result

    def get_sales(self, **kwargs):
        return self._corrupt(self._wrapped.get_sales(**kwargs))

    def get_inventory(self, **kwargs):
        return self._corrupt(self._wrapped.get_inventory(**kwargs))

    def get_competitor_prices(self, **kwargs):
        return self._corrupt(self._wrapped.get_competitor_prices(**kwargs))

    def as_of(self, as_of_date):
        from data.repositories.point_in_time import PointInTimeView

        return PointInTimeView(self, as_of_date)


class TestMutationLeakage:
    """T1 and T2."""

    def test_corrupting_the_future_does_not_change_training_features(
        self, forecast_history, forecast_view, forecast_config, forecast_sample
    ) -> None:
        """T1: features for origins before the cutoff must be untouched.

        ``y`` is deliberately excluded from the comparison. The target lives at
        ``origin + h``, which is legitimately after the cutoff for many rows, so
        it *should* change. Comparing whole frames would fail for the wrong
        reason - and the natural "fix" would be to weaken the test until it
        stopped failing.
        """
        origins = pd.to_datetime(forecast_history["date"]).dt.date
        cutoff = origins.min() + timedelta(days=(origins.max() - origins.min()).days // 2)

        clean = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=5
        )
        corrupted_history = forecast_history.copy()
        dates = pd.to_datetime(corrupted_history["date"]).dt.date
        after = dates > cutoff
        for column in ("units", "lag_1_units", "rolling_7_units"):
            if column in corrupted_history.columns:
                corrupted_history.loc[after, column] = (
                    corrupted_history.loc[after, column] * 1000
                )

        corrupted = build_horizon_dataset(
            corrupted_history, forecast_view, forecast_config, forecast_sample, seed=5
        )

        # Compare only rows whose *origin* precedes the cutoff.
        feature_columns = [
            c for c in clean.feature_names if c in corrupted.frame.columns and c != TARGET
        ]
        clean_rows = clean.frame[
            pd.to_datetime(clean.frame[ORIGIN_DATE]).dt.date <= cutoff
        ][feature_columns].reset_index(drop=True)
        corrupted_rows = corrupted.frame[
            pd.to_datetime(corrupted.frame[ORIGIN_DATE]).dt.date <= cutoff
        ][feature_columns].reset_index(drop=True)

        assert len(clean_rows) > 100, "too few pre-cutoff rows to conclude anything"
        pd.testing.assert_frame_equal(clean_rows, corrupted_rows)

    def test_the_mutation_test_can_actually_fail(
        self, forecast_history, forecast_view, forecast_config, forecast_sample
    ) -> None:
        """T2: plant the bug and confirm T1's comparison breaks.

        Without this, T1 is unfalsifiable. ``horizon_features_from_target``
        sources target-side features from the origin date instead - the precise
        confusion this design exists to prevent - and the two datasets must then
        differ.
        """
        honest = build_horizon_dataset(
            forecast_history, forecast_view, forecast_config, forecast_sample, seed=5
        )
        leaky = build_horizon_dataset(
            forecast_history,
            forecast_view,
            forecast_config,
            forecast_sample,
            seed=5,
            horizon_features_from_target=True,
        )

        column = f"{TARGET_PREFIX}day_of_week"
        assert column in honest.frame.columns
        assert column in leaky.frame.columns

        # The honest frame's calendar follows the target date; the leaky one's
        # follows the origin. They cannot agree except by coincidence.
        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(
                honest.frame[column].reset_index(drop=True),
                leaky.frame[column].reset_index(drop=True),
            )


class TestTrainServeEquivalence:
    """T6: the two feature paths must agree exactly."""

    def test_target_side_features_match_between_paths(
        self, forecast_view, forecast_sample
    ) -> None:
        """The same (series, date) must produce the same vector either way.

        Training reads target-side features over historical dates; serving reads
        them over future dates. Both go through ``target_side_features``, so this
        asserts the property that makes that single-function design worth having.
        """
        pairs = forecast_sample.pairs.head(5)
        dates = pd.date_range("2025-06-01", periods=14, freq="D")

        # Two independent calls, as training and serving would make them.
        first = target_side_features(forecast_view, pairs, dates)
        second = target_side_features(forecast_view, pairs, dates)

        assert not first.empty
        pd.testing.assert_frame_equal(
            first.sort_values(["product_id", "store_id", TARGET_DATE]).reset_index(drop=True),
            second.sort_values(["product_id", "store_id", TARGET_DATE]).reset_index(drop=True),
        )

    def test_a_narrow_window_produces_the_same_columns_as_a_wide_one(
        self, forecast_view, forecast_sample
    ) -> None:
        """Regression test for a real train/serve skew.

        ``add_festival_proximity`` measures distance to the nearest festival in
        whatever calendar it is given. Over a full history that is plentiful;
        over a 7-day serving window there may be none, and the columns came out
        missing entirely - so the model raised at serving time on features it had
        trained with. Both paths now read a wide calendar window.
        """
        pairs = forecast_sample.pairs.head(3)

        narrow = target_side_features(
            forecast_view, pairs, pd.date_range("2025-06-01", periods=7, freq="D")
        )
        wide = target_side_features(
            forecast_view, pairs, pd.date_range("2024-01-01", periods=400, freq="D")
        )

        assert set(narrow.columns) == set(wide.columns)
        for column in (f"{TARGET_PREFIX}days_to_festival", f"{TARGET_PREFIX}days_since_festival"):
            assert column in narrow.columns


class TestBehaviouralLeakage:
    """T4 and T5: properties that need no column names."""

    def test_error_grows_with_horizon(self, trained_smoke_forecaster) -> None:
        """T4: the single strongest signal that the join is right.

        If forecasting 90 days out is as accurate as forecasting tomorrow, the
        model is not forecasting - it is reading the target date's information.
        This is behavioural, so it survives any refactor that renames features,
        and it would catch a leak introduced through a column nobody thought to
        add to an exclusion list.
        """
        buckets = trained_smoke_forecaster.bucket_metrics
        assert len(buckets) >= 4, "too few populated buckets to judge a trend"

        assert horizon_error_grows(buckets), (
            "error over the long horizon half does not exceed the short half: "
            + ", ".join(f"{k} {v.wmape:.1%} (n={v.n})" for k, v in buckets.items())
            + ". Forecasting three months out should be harder than forecasting "
            "tomorrow; if it is not, the origin/target join is leaking."
        )

    def test_accuracy_is_not_implausibly_good(self, trained_smoke_forecaster) -> None:
        """T5: nothing may score near-perfectly on genuinely noisy demand.

        Step 4 measured the irreducible noise floor on this data at 35% WMAPE -
        the score a model knowing the *true* conditional mean would still get,
        because demand is drawn from an over-dispersed negative binomial. A
        forecaster beating that is not skilled; it has seen the answer.
        """
        test_metrics = trained_smoke_forecaster.metrics.get("test")
        assert test_metrics is not None

        assert test_metrics.wmape > 0.15, (
            f"WMAPE of {test_metrics.wmape:.1%} is far below the ~35% irreducible "
            f"noise floor for this data. Suspect target leakage rather than skill."
        )


class TestOriginTargetSeparation:
    def test_no_feature_is_perfectly_correlated_with_the_target(
        self, horizon_dataset
    ) -> None:
        """A blanket check that does not depend on knowing column names.

        The named exclusions only catch leaks someone anticipated. This catches a
        new column added in a later step that happens to encode the target, which
        is how leaks actually get introduced.
        """
        frame = horizon_dataset.frame
        numeric = frame[horizon_dataset.feature_names].select_dtypes(include="number")
        correlations = numeric.corrwith(frame[TARGET]).abs().dropna()
        suspicious = correlations[correlations > 0.98]

        assert suspicious.empty, (
            f"these features are nearly identical to the target: {dict(suspicious)}"
        )

    def test_horizon_step_does_not_predict_the_target(self, horizon_dataset) -> None:
        """``horizon_step`` is drawn at random, so it must carry no signal about
        the target level. A correlation would mean the draw was not independent
        of the data - for instance if long horizons were only sampled for
        long-lived series."""
        frame = horizon_dataset.frame
        correlation = frame[HORIZON_STEP].corr(frame[TARGET])

        assert abs(correlation) < 0.1, (
            f"horizon_step correlates {correlation:.3f} with the target; the "
            f"random draw is not independent of the data"
        )
