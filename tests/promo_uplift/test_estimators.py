"""Estimator arithmetic, cross-fitting and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.promo_uplift.config import PromoUpliftConfig
from ml.promo_uplift.estimators import (
    AIPWEstimator,
    DRLearner,
    EffectEstimate,
    IPWEstimator,
    NuisanceFit,
    assign_folds,
)
from ml.promo_uplift.exceptions import EstimationError
from ml.promo_uplift.features import CovariateFrame
from ml.promo_uplift.propensity import att_weights, effective_sample_size

pytestmark = pytest.mark.models


class TestFoldAssignment:
    def test_series_scheme_keeps_a_listing_whole(self) -> None:
        """Nothing about a listing may inform its own counterfactual."""
        series = pd.Series(["A|1"] * 10 + ["B|2"] * 10 + ["C|3"] * 10)
        anchor = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))

        folds = assign_folds(series, anchor, 3, scheme="series", seed=1)
        for listing in series.unique():
            assert len(set(folds[series == listing])) == 1

    def test_series_scheme_is_deterministic(self) -> None:
        series = pd.Series([f"L{i % 12}" for i in range(60)])
        anchor = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))

        a = assign_folds(series, anchor, 4, scheme="series", seed=7)
        b = assign_folds(series, anchor, 4, scheme="series", seed=7)
        assert (a == b).all()

    def test_time_blocks_are_contiguous(self) -> None:
        series = pd.Series([f"L{i}" for i in range(50)])
        anchor = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))

        folds = assign_folds(series, anchor, 5, scheme="time_blocks")
        assert (np.diff(folds) >= 0).all()

    def test_unknown_scheme_is_refused(self) -> None:
        series = pd.Series(["A"])
        anchor = pd.Series(pd.to_datetime(["2024-01-01"]))
        with pytest.raises(EstimationError, match="unknown cross-fitting scheme"):
            assign_folds(series, anchor, 2, scheme="random")


class TestAIPWArithmetic:
    def test_the_estimate_matches_the_formula_by_hand(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """Reconstructed from the definition, not from another code path."""
        estimator = AIPWEstimator(config=confounded_config).fit(covariates, nuisance)
        estimate = estimator.estimate_ate()

        t = covariates.t
        y = covariates.y
        odds = att_weights(
            nuisance.propensity,
            t,
            stabilise_at=confounded_config.propensity.stabilise_weights_at,
        )
        residual = y - nuisance.mu0
        expected = (
            residual[t].sum() - (odds[~t] * residual[~t]).sum()
        ) / int(t.sum())

        assert estimate.ate == pytest.approx(expected)

    def test_standard_error_is_cluster_robust(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """Clustered on the listing, reconstructed by hand from the definition.

        Not `sd(psi)/sqrt(n)`. That formula assumes independent rows, and rows
        within a product-store are strongly serially correlated - measured, it
        produced intervals that covered the known truth in only four of six
        synthetic scenarios while every point estimate was within 2-5 points.
        """
        estimator = AIPWEstimator(config=confounded_config).fit(covariates, nuisance)
        estimate = estimator.estimate_ate()
        psi = estimator.influence()

        listing = covariates.frame[["product_id", "store_id"]].agg("|".join, axis=1)
        totals = pd.Series(psi).groupby(listing.to_numpy()).sum().to_numpy()
        n_clusters = len(totals)
        expected = float(
            np.sqrt(n_clusters / (n_clusters - 1) * np.sum(totals**2)) / len(psi)
        )

        assert estimate.standard_error == pytest.approx(expected)

    def test_clustered_error_exceeds_the_iid_formula(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """The correction must widen the interval, not narrow it.

        Positive within-cluster correlation is what makes the i.i.d. formula too
        small; if clustering ever produced a *smaller* standard error here, the
        clustering key would be wrong.
        """
        estimator = AIPWEstimator(config=confounded_config).fit(covariates, nuisance)
        psi = estimator.influence()
        iid = float(psi.std(ddof=1) / np.sqrt(len(psi)))

        assert estimator.estimate_ate().standard_error > iid

    def test_influence_function_is_mean_zero(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """A defining property. If it does not hold, the SE is meaningless."""
        estimator = AIPWEstimator(config=confounded_config).fit(covariates, nuisance)
        psi = estimator.influence()
        assert psi.mean() == pytest.approx(0.0, abs=1e-6 * max(abs(psi).max(), 1.0))

    def test_baseline_is_the_counterfactual_on_treated_rows(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """Not the raw control mean - that is a different population, and
        dividing by it would make the percentage incomparable to the effect."""
        estimate = (
            AIPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )
        assert estimate.baseline_units == pytest.approx(
            nuisance.mu0[covariates.t].mean()
        )

    def test_interval_brackets_the_point_estimate(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        estimate = (
            AIPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )
        assert estimate.ci_lower is not None
        assert estimate.ci_lower < estimate.ate < estimate.ci_upper

    def test_calibration_warning_fires_on_bad_propensities(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """Since E[(1-T)e/(1-e)] = P(T=1), the control weights must sum to about
        the treated count. This check is what would have caught the fold-structure
        bug that once turned a true +65% into -424%.
        """
        broken = NuisanceFit(
            mu0=nuisance.mu0,
            mu1=nuisance.mu1,
            # Every row scored as very likely treated: control odds explode.
            propensity=np.full_like(nuisance.propensity, 0.95),
            folds=nuisance.folds,
            mu0_r2=nuisance.mu0_r2,
        )
        estimate = (
            AIPWEstimator(config=confounded_config)
            .fit(covariates, broken)
            .estimate_ate()
        )
        assert any("calibration" in w for w in estimate.warnings)


class TestIPW:
    def test_uses_a_self_normalised_weighted_mean(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """Hajek rather than Horvitz-Thompson: the unnormalised form can produce
        a weighted mean outside the range of the data."""
        estimate = (
            IPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )
        y = covariates.y
        assert y.min() <= estimate.baseline_units <= y.max()

    def test_reports_effective_sample_size(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        estimate = (
            IPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )
        assert 0 < estimate.diagnostics["effective_sample_fraction"] <= 1.0

    def test_has_no_analytic_interval(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """An SE ignoring propensity estimation would look tighter than AIPW's,
        which is backwards. Absent is the honest answer."""
        estimate = (
            IPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )
        assert estimate.standard_error is None
        assert not estimate.has_interval

    def test_cate_is_refused(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        estimator = IPWEstimator(config=confounded_config).fit(covariates, nuisance)
        with pytest.raises(EstimationError, match="DR-learner"):
            estimator.estimate_cate(covariates.X)


class TestDRLearner:
    def test_pseudo_outcome_is_winsorised(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        learner = DRLearner(config=confounded_config).fit(covariates, nuisance)
        estimate = learner.estimate_ate()
        assert estimate.diagnostics["winsorised_share"] > 0

    def test_cate_has_one_value_per_row(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        learner = DRLearner(config=confounded_config).fit(covariates, nuisance)
        assert len(learner.estimate_cate(covariates.X)) == len(covariates.X)

    def test_sparse_segments_return_null_not_a_number(
        self, covariates: CovariateFrame, nuisance: NuisanceFit,
        confounded_config: PromoUpliftConfig,
    ) -> None:
        """A segment uplift computed from eight promotions is a rounding error
        with a label on it, and Step 8 would allocate budget against it."""
        learner = DRLearner(config=confounded_config).fit(covariates, nuisance)
        segments = learner.segment_effects("category", min_treated=10_000)
        assert segments["uplift_pct"].isna().all()
        assert not segments["estimable"].any()

    def test_unfitted_estimator_refuses(
        self, covariates: CovariateFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        learner = DRLearner(config=confounded_config)
        with pytest.raises(EstimationError, match="not fitted"):
            learner.estimate_cate(covariates.X)


class TestEffectEstimate:
    def test_significance_requires_an_interval(self) -> None:
        """An estimate whose uncertainty was never established is not
        'significant by default'."""
        estimate = EffectEstimate(
            method="x", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        assert not estimate.significant
        assert not estimate.has_interval

    def test_interval_excluding_zero_is_significant(self) -> None:
        estimate = EffectEstimate(
            method="x", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400, ci_lower=1.0, ci_upper=9.0,
        )
        assert estimate.significant

    def test_interval_spanning_zero_is_not(self) -> None:
        estimate = EffectEstimate(
            method="x", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400, ci_lower=-1.0, ci_upper=11.0,
        )
        assert not estimate.significant

    def test_percentage_interval_scales_by_the_baseline(self) -> None:
        estimate = EffectEstimate(
            method="x", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400, ci_lower=2.5, ci_upper=7.5,
        )
        interval = estimate.interval_pct()
        assert interval == pytest.approx((0.1, 0.3))


class TestWeights:
    def test_treated_rows_weigh_one(self) -> None:
        """The treated group is the population of interest, so it is already
        correctly represented."""
        scores = np.array([0.2, 0.5, 0.8])
        t = np.array([True, False, True])
        assert att_weights(scores, t)[t].tolist() == [1.0, 1.0]

    def test_control_weight_is_the_odds(self) -> None:
        scores = np.array([0.25])
        t = np.array([False])
        assert att_weights(scores, t)[0] == pytest.approx(0.25 / 0.75)

    def test_stabilisation_caps_the_extremes(self) -> None:
        """One row at e=0.999 would otherwise carry a weight of 999 and *be* the
        weighted control mean."""
        scores = np.append(np.full(99, 0.2), 0.999)
        t = np.zeros(100, dtype=bool)

        raw = att_weights(scores, t)
        capped = att_weights(scores, t, stabilise_at=99.0)

        assert raw.max() == pytest.approx(0.999 / 0.001, rel=1e-3)
        assert capped.max() < raw.max() / 50
        # Everything below the cap is untouched.
        assert capped[:99].tolist() == raw[:99].tolist()

    def test_effective_sample_size_falls_with_concentration(self) -> None:
        even = np.ones(100)
        skewed = np.append(np.full(99, 0.01), 100.0)
        assert effective_sample_size(even) == pytest.approx(100.0)
        assert effective_sample_size(skewed) < 5
