"""Pre-treatment covariates and the leakage they exist to prevent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.promo_uplift.features import (
    POST_TREATMENT_FEATURES,
    CovariateFrame,
    anchor_dates,
)
from ml.promo_uplift.treatment import AnalysisFrame, RowRole

pytestmark = [pytest.mark.leakage, pytest.mark.models]


class TestAnchoring:
    def test_treated_rows_share_one_anchor_per_event(
        self, covariates: CovariateFrame
    ) -> None:
        """Every row of a promotion must see the same pre-treatment information.

        On day five of a promotion, a covariate anchored at the row's own date
        would contain four days of the effect being estimated.
        """
        treated = covariates.frame[covariates.frame["role"] == RowRole.TREATED]
        per_event = treated.groupby("promotion_id", observed=True)["_anchor_date"].nunique()
        assert (per_event == 1).all()

    def test_trailing_covariates_are_constant_within_an_event(
        self, covariates: CovariateFrame
    ) -> None:
        treated = covariates.frame[covariates.frame["role"] == RowRole.TREATED]
        long_events = treated.groupby("promotion_id", observed=True).filter(
            lambda g: len(g) >= 5
        )
        if long_events.empty:
            pytest.skip("no event long enough to test")

        spread = long_events.groupby("promotion_id", observed=True)[
            "demand_mean_7"
        ].nunique()
        assert (spread == 1).all()

    def test_anchor_is_the_event_start_not_the_day_before(
        self, analysis: AnalysisFrame
    ) -> None:
        """The exclusion of the current day lives in the trailing statistics.

        Subtracting a day here as well would shift the window twice and discard
        the most recent - and most informative - day of history.
        """
        anchors = anchor_dates(analysis.frame, analysis.events)
        treated = analysis.frame["role"] == RowRole.TREATED
        starts = pd.to_datetime(
            analysis.events.set_index("promotion_id")["start_date"]
        )
        expected = analysis.frame.loc[treated, "promotion_id"].map(starts)
        assert anchors[treated].equals(expected)

    def test_control_rows_anchor_at_their_own_date(
        self, analysis: AnalysisFrame
    ) -> None:
        anchors = anchor_dates(analysis.frame, analysis.events)
        control = analysis.frame["role"] == RowRole.CONTROL
        assert anchors[control].equals(pd.to_datetime(analysis.frame.loc[control, "date"]))

    def test_trailing_mean_matches_a_hand_computation(
        self, analysis: AnalysisFrame, covariates: CovariateFrame
    ) -> None:
        """Reconstructed by hand from the source panel, not from another feature."""
        treated = covariates.frame[covariates.frame["role"] == RowRole.TREATED]
        long_events = analysis.events[analysis.events["duration_days"] >= 6]
        if long_events.empty or treated.empty:
            pytest.skip("no suitable event")

        for event in long_events.head(3).to_dict("records"):
            rows = treated[treated["promotion_id"] == event["promotion_id"]]
            if rows.empty:
                continue
            history = analysis.frame[
                (analysis.frame["product_id"] == event["product_id"])
                & (analysis.frame["store_id"] == event["store_id"])
                & (analysis.frame["date"] < event["start_date"])
            ].sort_values("date")
            if len(history) < 7:
                continue
            expected = history["units"].tail(7).mean()
            assert float(rows["demand_mean_7"].iloc[0]) == pytest.approx(expected)
            return
        pytest.skip("no event with enough history")


class TestPostTreatmentExclusion:
    def test_no_post_treatment_column_reaches_the_adjustment_set(
        self, covariates: CovariateFrame
    ) -> None:
        assert not set(covariates.feature_names) & POST_TREATMENT_FEATURES

    def test_the_mediators_are_excluded_by_name(self) -> None:
        """Discount and selling price are consequences of treatment.

        Conditioning on them holds the price cut fixed across arms, which
        removes the largest channel a promotion works through - and the
        estimate that survives is the mechanic alone, reported as the whole
        effect.
        """
        for mediator in ("selling_price", "discount_percentage", "promotion_flag"):
            assert mediator in POST_TREATMENT_FEATURES

    def test_the_collider_is_excluded(self) -> None:
        """Stockout is caused by treatment AND correlated with demand.

        Conditioning on it opens a path that was closed. It is used to filter
        rows but must never enter the adjustment set.
        """
        assert "stockout_flag" in POST_TREATMENT_FEATURES

    def test_outcome_and_its_arithmetic_are_excluded(self) -> None:
        for column in ("units", "revenue", "cost", "gross_profit"):
            assert column in POST_TREATMENT_FEATURES

    def test_the_guard_fires_on_a_planted_name(self) -> None:
        """Falsifiability, tested directly against the guard.

        Note what this does *not* do: patch the module constant and rebuild the
        covariates. That can never fail, because `build_covariates` filters the
        feature list with the same set the assertion checks - so a name added to
        the set is removed from the features before the assertion sees it. A
        test written that way passes while proving nothing, which is worse than
        no test. The guard is exercised directly instead.
        """
        from ml.promo_uplift.features import _assert_no_post_treatment

        _assert_no_post_treatment(("demand_mean_7", "category"))
        with pytest.raises(AssertionError, match="post-treatment"):
            _assert_no_post_treatment(("demand_mean_7", "units"))

    def test_features_come_only_from_the_declared_groups(
        self, covariates: CovariateFrame
    ) -> None:
        """The real protection is an allow-list, not the deny-list above.

        A column cannot become a covariate by appearing in the panel - it has to
        be constructed by one of the three feature builders. That is what stops
        a new post-treatment column nobody thought to exclude from silently
        entering the adjustment set.
        """
        declared = {name for group in covariates.groups.values() for name in group}
        assert set(covariates.feature_names) <= declared


class TestCovariateContent:
    def test_the_seasonal_confounder_is_present(
        self, covariates: CovariateFrame
    ) -> None:
        """Promotion timing is targeted at each category's seasonal peak.

        Omitting the seasonal position would leave the main back-door path
        wide open, however good the rest of the adjustment set is.
        """
        assert "season_sin_1" in covariates.feature_names
        assert "season_cos_1" in covariates.feature_names
        assert "category" in covariates.feature_names

    def test_prior_promotion_intensity_is_present(
        self, covariates: CovariateFrame
    ) -> None:
        assert "promo_share_28" in covariates.feature_names

    def test_demand_history_is_present(self, covariates: CovariateFrame) -> None:
        for name in ("demand_lag_1", "demand_mean_28", "demand_mean_56"):
            assert name in covariates.feature_names

    def test_identifiers_are_carried_but_not_fitted_on(
        self, covariates: CovariateFrame
    ) -> None:
        """A model given raw ids memorises listings instead of learning what
        makes them promotable."""
        for identifier in ("product_id", "store_id", "date"):
            assert identifier in covariates.frame.columns
            assert identifier not in covariates.feature_names

    def test_no_covariate_is_all_null(self, covariates: CovariateFrame) -> None:
        nulls = covariates.X.isna().all()
        assert not nulls.any(), f"all-null covariates: {list(nulls[nulls].index)}"

    def test_rows_without_complete_history_are_dropped_not_imputed(
        self, covariates: CovariateFrame
    ) -> None:
        """Imputing a trailing mean asserts the listing runs at the average
        rate, which is a claim nobody made."""
        required = [
            n for n in covariates.feature_names if n.startswith(("demand_", "promo_share"))
        ]
        assert not covariates.X[required].isna().any().any()

    def test_both_arms_survive(self, covariates: CovariateFrame) -> None:
        assert covariates.t.sum() > 0
        assert (~covariates.t).sum() > 0


class TestAccessors:
    def test_design_matrix_outcome_and_treatment_are_aligned(
        self, covariates: CovariateFrame
    ) -> None:
        assert len(covariates.X) == len(covariates.y) == len(covariates.t)

    def test_numeric_names_exclude_categoricals(
        self, covariates: CovariateFrame
    ) -> None:
        assert not set(covariates.numeric_names()) & set(covariates.categorical_names)

    def test_numeric_covariates_are_finite_or_null(
        self, covariates: CovariateFrame
    ) -> None:
        numeric = covariates.X[list(covariates.numeric_names())].to_numpy(dtype=float)
        assert not np.isinf(numeric).any()
