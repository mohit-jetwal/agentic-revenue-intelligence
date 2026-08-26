"""Training mechanics: splits, row filtering and promotion neutralisation.

These are the tests that protect the *setup* rather than the fit. A model can be
perfectly specified and still be worthless because the split leaked, the
censored rows stayed in, or the promotion features were not actually neutralised
at prediction time - and none of those show up as an error, only as metrics that
look better than they should.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from ml.baseline.training import (
    EXCLUDED_FROM_FEATURES,
    PROMOTION_FEATURES,
    SUPPLY_FEATURES,
    TARGET,
    PromotionApproach,
    build_temporal_split,
    neutralise_promotions,
    prepare_training_rows,
    select_features,
)

pytestmark = pytest.mark.models


class TestTemporalSplit:
    def test_folds_are_ordered_and_non_overlapping(self, feature_panel: pd.DataFrame) -> None:
        """Every fold strictly follows the one before it.

        The single property that makes the evaluation meaningful: if any fold
        overlaps or precedes its predecessor, the model is being scored on data
        it could have learned from.
        """
        split = build_temporal_split(feature_panel)

        assert split.train_start <= split.train_end
        assert split.train_end < split.calibration_start
        assert split.calibration_end < split.valid_start
        assert split.valid_end < split.test_start
        assert split.test_start <= split.test_end

    def test_folds_are_contiguous(self, feature_panel: pd.DataFrame) -> None:
        """No gap between folds - a gap would silently discard history."""
        split = build_temporal_split(feature_panel)

        assert split.calibration_start == split.train_end + timedelta(days=1)
        assert split.valid_start == split.calibration_end + timedelta(days=1)
        assert split.test_start == split.valid_end + timedelta(days=1)

    def test_test_fold_ends_at_the_last_available_date(
        self, feature_panel: pd.DataFrame
    ) -> None:
        split = build_temporal_split(feature_panel)
        last = pd.to_datetime(feature_panel["date"]).dt.date.max()

        assert split.test_end == last

    def test_requested_fold_widths_are_honoured(self, feature_panel: pd.DataFrame) -> None:
        split = build_temporal_split(
            feature_panel, test_days=100, valid_days=80, calibration_days=50
        )

        assert (split.test_end - split.test_start).days + 1 == 100
        assert (split.valid_end - split.valid_start).days + 1 == 80
        assert (split.calibration_end - split.calibration_start).days + 1 == 50

    def test_insufficient_history_raises_rather_than_producing_a_tiny_train_fold(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """Refusing is the right failure mode here.

        Silently carving a 3-day training fold out of short history produces a
        model that trains, predicts, and is meaningless. An exception naming the
        shortfall is far more useful than a fitted object nobody can trust.
        """
        dates = pd.to_datetime(feature_panel["date"]).dt.date
        short = feature_panel[dates < dates.min() + timedelta(days=200)]

        with pytest.raises(ValueError, match="days of history"):
            build_temporal_split(short)


class TestFeatureSelection:
    def test_target_is_never_a_feature(self, feature_panel: pd.DataFrame) -> None:
        for approach in PromotionApproach:
            assert TARGET not in select_features(feature_panel, approach=approach)

    def test_identifiers_are_never_features(self, feature_panel: pd.DataFrame) -> None:
        """Identifiers memorise rather than generalise.

        A tree given ``store_id`` can encode a per-store constant, which flatters
        the test metric on known stores and collapses on a new one.
        """
        features = select_features(feature_panel, approach=PromotionApproach.CONTROL)

        for identifier in ("date", "product_id", "store_id"):
            assert identifier not in features

    def test_exclude_approach_drops_every_promotion_feature(
        self, feature_panel: pd.DataFrame
    ) -> None:
        features = set(select_features(feature_panel, approach=PromotionApproach.EXCLUDE))

        assert not (features & PROMOTION_FEATURES)

    def test_control_approach_keeps_promotion_features(
        self, feature_panel: pd.DataFrame
    ) -> None:
        features = set(select_features(feature_panel, approach=PromotionApproach.CONTROL))

        assert features & PROMOTION_FEATURES

    def test_supply_columns_are_never_features(self) -> None:
        """A baseline must not condition on inventory, under either approach.

        Regression test for a measured defect. With inventory columns present,
        ``closing_inventory_lag_1`` became the most important feature in the
        LightGBM baseline and the model recovered only 0.30 of true demand
        during stockouts - it had learned that low stock predicts low sales, and
        would therefore report a supply failure as a demand collapse.

        Excluding stockout *rows* does not prevent this: the relationship is
        learned from partially-depleted rows just below the threshold and then
        extrapolated to zero stock. The columns themselves have to go.
        """
        panel = pd.DataFrame(
            {
                "units": [1.0],
                "rolling_28_units": [1.0],
                **{column: [0.0] for column in SUPPLY_FEATURES},
            }
        )

        for approach in PromotionApproach:
            features = set(select_features(panel, approach=approach))

            assert not (features & SUPPLY_FEATURES), (
                f"supply columns leaked into the {approach.value} feature set: "
                f"{features & SUPPLY_FEATURES}"
            )

    def test_supply_exclusion_survives_the_real_panel(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """The exclusion must hold against actual panel columns, not just a stub."""
        features = set(select_features(feature_panel, approach=PromotionApproach.CONTROL))

        assert not (features & SUPPLY_FEATURES)

    def test_ground_truth_columns_never_become_features(
        self, synthetic_panel: pd.DataFrame
    ) -> None:
        """The fixture's hidden truth must not leak into a feature matrix.

        ``latent_units`` and ``true_baseline`` are the test analogue of Step 2's
        unpublished parameters. They are not in ``EXCLUDED_FROM_FEATURES``,
        because production never sees them - so this asserts the discipline that
        tests drop them, and would catch a fixture change that stopped doing so.
        """
        features = select_features(synthetic_panel, approach=PromotionApproach.CONTROL)

        # Documents the real reason these are safe: the fixture strips them, not
        # the exclusion list.
        assert "latent_units" not in EXCLUDED_FROM_FEATURES
        assert set(features) & {"latent_units", "true_baseline", "true_mean"}, (
            "select_features does not know about the fixture's ground-truth "
            "columns - tests must use the `feature_panel` fixture, which drops "
            "them, rather than `synthetic_panel`"
        )


class TestTrainingRowFiltering:
    def test_stockout_rows_are_always_excluded(self, feature_panel: pd.DataFrame) -> None:
        """Under both approaches, because the target is censored either way.

        This is the single most consequential filter in the step. Leaving these
        rows in teaches the model that a supply failure predicts low demand,
        which inverts every downstream root-cause conclusion.
        """
        for approach in PromotionApproach:
            rows, _ = prepare_training_rows(feature_panel, approach=approach)

            assert not rows["stockout_flag"].astype(bool).any()

    def test_exclude_approach_drops_promotional_rows(
        self, feature_panel: pd.DataFrame
    ) -> None:
        rows, _ = prepare_training_rows(
            feature_panel, approach=PromotionApproach.EXCLUDE
        )

        assert not rows["promotion_flag"].astype(bool).any()

    def test_control_approach_retains_promotional_rows(
        self, feature_panel: pd.DataFrame
    ) -> None:
        rows, _ = prepare_training_rows(
            feature_panel, approach=PromotionApproach.CONTROL
        )

        assert rows["promotion_flag"].astype(bool).any()

    def test_exclusion_counts_are_reported(self, feature_panel: pd.DataFrame) -> None:
        """The caller must be able to see how much data a filter removed."""
        _, excluded = prepare_training_rows(
            feature_panel, approach=PromotionApproach.EXCLUDE
        )

        assert excluded["total"] == len(feature_panel)
        assert excluded["stockout"] > 0
        assert excluded["promotional"] > 0

    def test_control_keeps_strictly_more_rows_than_exclude(
        self, feature_panel: pd.DataFrame
    ) -> None:
        control, _ = prepare_training_rows(
            feature_panel, approach=PromotionApproach.CONTROL
        )
        exclude, _ = prepare_training_rows(
            feature_panel, approach=PromotionApproach.EXCLUDE
        )

        assert len(control) > len(exclude)


class TestPromotionNeutralisation:
    def test_promotion_flag_is_zeroed(self, feature_panel: pd.DataFrame) -> None:
        """The whole point of Approach B's prediction step.

        Without this the 'baseline' would include the promotion the question is
        trying to measure, and uplift would come out near zero.
        """
        promoted = feature_panel[feature_panel["promotion_flag"].astype(bool)].head(200)

        neutral = neutralise_promotions(promoted)

        assert not neutral["promotion_flag"].astype(bool).any()

    def test_promotion_discount_is_zeroed(self, feature_panel: pd.DataFrame) -> None:
        promoted = feature_panel[feature_panel["promotion_flag"].astype(bool)].head(200)

        neutral = neutralise_promotions(promoted)

        assert (neutral["promotion_discount"] == 0).all()

    def test_non_promotion_features_are_untouched(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """Neutralisation must change the promotion state and nothing else.

        A version that also reset price or seasonality would answer a different
        counterfactual question than the one being asked.
        """
        sample = feature_panel.head(200)

        neutral = neutralise_promotions(sample)

        untouched = [c for c in sample.columns if c not in PROMOTION_FEATURES]
        pd.testing.assert_frame_equal(neutral[untouched], sample[untouched])

    def test_does_not_mutate_the_input(self, feature_panel: pd.DataFrame) -> None:
        """A silent in-place edit here would corrupt the shared panel.

        The session-scoped fixture is reused by every other test, so mutation
        would produce failures in unrelated tests that look nothing like the
        cause.
        """
        sample = feature_panel[feature_panel["promotion_flag"].astype(bool)].head(200)
        before = sample.copy()

        neutralise_promotions(sample)

        pd.testing.assert_frame_equal(sample, before)

    def test_promotion_type_becomes_missing_not_a_new_label(self) -> None:
        """Regression test for a real bug.

        Setting a categorical to the string ``"none"`` introduces a level the
        encoder never saw during training, which either raises or silently maps
        to an unknown bucket. Missing is what an unpromoted row actually looked
        like in training, so that is what neutralisation must produce.
        """
        frame = pd.DataFrame(
            {
                "promotion_flag": [True, True],
                "promotion_type": pd.Categorical(
                    ["bogo", "discount"], categories=["bogo", "discount"]
                ),
            }
        )

        neutral = neutralise_promotions(frame)

        assert neutral["promotion_type"].isna().all()
        assert list(neutral["promotion_type"].cat.categories) == ["bogo", "discount"]
