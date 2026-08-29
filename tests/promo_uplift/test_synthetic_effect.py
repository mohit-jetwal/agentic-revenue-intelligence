"""Recovery of known treatment effects (brief section 32).

These are the tests that establish the estimator is *correct*, as opposed to
merely runnable. Nothing else in the suite can: on real data the counterfactual
is missing, so a method can score perfectly on every predictive metric while
being wrong about the effect by any margin at all.

Here the effect is applied by hand and recorded, so recovery is checkable
against a number rather than against a plausibility judgement.

The tolerance scales with the size of the true effect. A fixed three points
would be a strict test at +65% and an impossible one at 0%, so the same
threshold would silently mean two different things across the scenario set.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.promo_uplift.config import PromoUpliftConfig
from ml.promo_uplift.controls import build_control_pool
from ml.promo_uplift.estimators import AIPWEstimator, DRLearner, IPWEstimator, fit_nuisances
from ml.promo_uplift.features import build_covariates
from ml.promo_uplift.synthetic import (
    GROUND_TRUTH_COLUMNS,
    SCENARIOS,
    SyntheticPanel,
    generate,
    scenario_config,
)
from ml.promo_uplift.treatment import build_analysis_frame

pytestmark = [pytest.mark.causal, pytest.mark.models]

SERIES = 60
DAYS = 300


def _estimate(
    name: str, base: PromoUpliftConfig, *, seed: int = 11
) -> tuple[SyntheticPanel, float, float, float]:
    """Run the shipped pipeline on one scenario.

    Returns the panel, the naive estimate, the AIPW estimate and the tolerance.
    """
    config = scenario_config(name, base)
    panel = generate(name, config=config, n_series=SERIES, n_days=DAYS, seed=seed)

    analysis = build_analysis_frame(panel.observable(), config=config)
    pool = build_control_pool(analysis, config=config)
    covariates = build_covariates(
        pool.frame, analysis.events, config=config, history=analysis.frame
    )
    nuisance = fit_nuisances(covariates, config=config)
    estimate = AIPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()

    y, t = covariates.y, covariates.t
    naive = y[t].mean() / y[~t].mean() - 1.0

    tolerance = 0.06 * max(abs(panel.true_att_pct) / 0.15, 1.0)
    return panel, naive, estimate.ate_pct, tolerance


class TestGroundTruthIsSound:
    """The generator itself must be right before anything is tested against it."""

    def test_true_effect_is_exact_not_estimated(self, base_config: PromoUpliftConfig) -> None:
        panel = generate("positive", config=base_config, n_series=30, n_days=200)
        frame = panel.frame
        treated = frame["treatment"]

        # true_effect_units is lambda_treated - lambda_untreated, both recorded
        # at generation time. The ATT is their mean over treated rows, computed
        # as a ratio of totals rather than a mean of ratios.
        expected = frame.loc[treated, "true_effect_units"].mean()
        assert panel.true_att_units == pytest.approx(expected)

        baseline = frame.loc[treated, "true_lambda_untreated"].sum()
        incremental = frame.loc[treated, "true_effect_units"].sum()
        assert panel.true_att_pct == pytest.approx(incremental / baseline)

    def test_control_rows_have_no_effect_by_construction(
        self, base_config: PromoUpliftConfig
    ) -> None:
        panel = generate("positive", config=base_config, n_series=30, n_days=200)
        control = panel.frame[~panel.frame["treatment"]]
        assert (control["true_effect_units"].abs() < 1e-9).all()

    def test_observable_frame_hides_every_ground_truth_column(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """A leaked truth column makes every result look excellent, invisibly."""
        panel = generate("positive", config=base_config, n_series=20, n_days=150)
        assert not GROUND_TRUTH_COLUMNS & set(panel.observable().columns)

    def test_generation_is_deterministic(self, base_config: PromoUpliftConfig) -> None:
        a = generate("confounded", config=base_config, n_series=20, n_days=150, seed=3)
        b = generate("confounded", config=base_config, n_series=20, n_days=150, seed=3)
        assert a.true_att_pct == b.true_att_pct
        assert a.frame["units"].tolist() == b.frame["units"].tolist()

    def test_null_scenarios_have_exactly_zero_effect(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """A discounted 'null' promotion would still move volume via elasticity.

        The null scenarios use mechanic-only promotions with no price cut, which
        is what makes a genuinely zero true effect possible.
        """
        for name in ("null", "confounded_null"):
            panel = generate(name, config=base_config, n_series=30, n_days=200)
            assert panel.true_att_pct == pytest.approx(0.0, abs=1e-9)
            assert (panel.frame["discount_percentage"] == 0).all()


class TestEffectRecovery:
    def test_positive_effect_is_recovered(self, base_config: PromoUpliftConfig) -> None:
        panel, _, aipw, tolerance = _estimate("positive", base_config)
        assert aipw == pytest.approx(panel.true_att_pct, abs=tolerance)

    def test_negative_effect_stays_negative(self, base_config: PromoUpliftConfig) -> None:
        """Nothing anywhere floors the estimate at zero.

        A promotion that destroys volume is a real finding and the one Step 8
        most needs to be able to act on. An estimator that cannot return it is
        useless for allocation.
        """
        panel, _, aipw, tolerance = _estimate("negative", base_config)
        assert panel.true_att_pct < 0
        assert aipw < 0
        assert aipw == pytest.approx(panel.true_att_pct, abs=tolerance)

    def test_null_effect_is_near_zero(self, base_config: PromoUpliftConfig) -> None:
        _, _, aipw, _ = _estimate("null", base_config)
        assert abs(aipw) < 0.06


class TestConfounding:
    """The scenarios that justify the whole apparatus."""

    def test_naive_is_biased_upward_under_confounding(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """Promotions are scheduled into strong demand, so the naive comparison
        counts the season as if the promotion caused it."""
        panel, naive, _, _ = _estimate("confounded", base_config)
        assert naive > panel.true_att_pct + 0.15

    def test_adjustment_removes_most_of_that_bias(
        self, base_config: PromoUpliftConfig
    ) -> None:
        panel, naive, aipw, tolerance = _estimate("confounded", base_config)

        naive_error = abs(naive - panel.true_att_pct)
        aipw_error = abs(aipw - panel.true_att_pct)
        assert aipw_error < naive_error / 2
        assert aipw == pytest.approx(panel.true_att_pct, abs=tolerance)

    def test_confounded_null_is_the_sharpest_test(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """Targeted promotions that do nothing at all.

        The naive method finds a large, confident, entirely spurious uplift. Any
        method that reported anything other than zero here would, on real data,
        invent effects for promotions that did nothing - which is the specific
        failure this capability exists to prevent.
        """
        _, naive, aipw, _ = _estimate("confounded_null", base_config)
        assert naive > 0.15, "the scenario must produce a spurious naive effect"
        assert abs(aipw) < 0.08
        assert abs(aipw) < naive / 3


class TestHeterogeneity:
    def test_cate_ranks_segments_correctly(self, base_config: PromoUpliftConfig) -> None:
        """The DR-learner's actual job: which segments respond best.

        The absolute values include the price channel, which is common across
        segments, so they sit above the mechanic values in the generator. The
        *ordering* is what a budget allocation depends on.
        """
        config = scenario_config("heterogeneous", base_config)
        panel = generate(
            "heterogeneous", config=config, n_series=80, n_days=DAYS, seed=11
        )
        analysis = build_analysis_frame(panel.observable(), config=config)
        pool = build_control_pool(analysis, config=config)
        covariates = build_covariates(
            pool.frame, analysis.events, config=config, history=analysis.frame
        )
        nuisance = fit_nuisances(covariates, config=config)

        learner = DRLearner(config=config).fit(covariates, nuisance)
        segments = learner.segment_effects("store_segment", min_treated=20)
        ranked = segments.sort_values("uplift_pct", ascending=False)["segment"].tolist()

        assert ranked == ["A", "B", "C"]

    def test_aggregate_hides_the_negative_segment(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """Segment C has a negative mechanic while the average is strongly
        positive - which is exactly why an aggregate ATT is not enough to
        allocate budget with."""
        panel = generate(
            "heterogeneous", config=base_config, n_series=60, n_days=250, seed=11
        )
        truth = panel.frame[panel.frame["treatment"]]
        by_segment = truth.groupby("store_segment", observed=True)[
            "true_segment_uplift"
        ].mean()

        assert by_segment["C"] < 0
        assert panel.true_att_pct > 0


class TestEstimatorAgreement:
    def test_estimators_agree_within_tolerance(
        self, base_config: PromoUpliftConfig
    ) -> None:
        """Agreement is evidence; it is not proof, and the test says which.

        IPW and AIPW rest on different failure modes, so a large divergence
        localises a problem. They are not expected to be identical - AIPW is the
        efficient estimator and IPW is not.
        """
        config = scenario_config("confounded", base_config)
        panel = generate("confounded", config=config, n_series=SERIES, n_days=DAYS, seed=11)
        analysis = build_analysis_frame(panel.observable(), config=config)
        pool = build_control_pool(analysis, config=config)
        covariates = build_covariates(
            pool.frame, analysis.events, config=config, history=analysis.frame
        )
        nuisance = fit_nuisances(covariates, config=config)

        ipw = IPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()
        aipw = AIPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()

        assert abs(ipw.ate_pct - aipw.ate_pct) < 0.20
        assert np.sign(ipw.ate_pct) == np.sign(aipw.ate_pct)

    def test_every_scenario_is_covered_by_a_test(self) -> None:
        """A scenario nobody asserts on is decoration.

        Guards against a scenario being added to the generator and never
        checked, which would leave the suite looking more thorough than it is.
        """
        asserted = {
            "positive",
            "negative",
            "null",
            "confounded",
            "confounded_null",
            "heterogeneous",
        }
        assert set(SCENARIOS) == asserted
