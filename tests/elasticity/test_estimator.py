"""Own-price elasticity: recovery, bias direction, and the instrument guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.price_elasticity.estimator import (
    METHOD_PREFERENCE,
    NOT_SELECTABLE,
    check_identification,
    comparison_table,
    estimate_all,
    instrument_diagnostics,
    iv_two_stage,
    naive_ols,
    panel_fixed_effects,
    prepare_panel,
    randomised_subset,
    select_estimate,
)
from tests.elasticity.conftest import TRUE_ELASTICITY, make_panel

pytestmark = pytest.mark.models


class TestPanelPreparation:
    def test_builds_log_columns(self, clean_panel: pd.DataFrame) -> None:
        frame = prepare_panel(clean_panel)
        assert "log_units" in frame.columns
        assert "log_price" in frame.columns
        assert np.isfinite(frame["log_units"]).all()

    def test_drops_zero_unit_rows(self) -> None:
        """log(0) is undefined, and log(q+1) would change the quantity being
        estimated into something with no clean interpretation."""
        panel = make_panel(n_stores=3, n_days=100)
        panel.loc[panel.index[:20], "units"] = 0
        frame = prepare_panel(panel)
        assert (frame["units"] > 0).all()

    def test_drops_promoted_rows_by_default(self) -> None:
        """A promotion moves price and applies a mechanic at once, so a promoted
        row attributes the mechanic's lift to the price cut."""
        panel = make_panel(n_stores=3, n_days=100)
        panel.loc[panel.index[:50], "promotion_flag"] = True
        assert len(prepare_panel(panel)) == len(panel) - 50

    def test_keeps_promoted_rows_when_asked(self) -> None:
        """Cross-price needs them: a candidate's promotional price cut is the
        largest source of the variation that identifies the effect."""
        panel = make_panel(n_stores=3, n_days=100)
        panel.loc[panel.index[:50], "promotion_flag"] = True
        assert len(prepare_panel(panel, drop_promotions=False)) == len(panel)

    def test_drops_censored_rows(self) -> None:
        panel = make_panel(n_stores=3, n_days=100)
        panel.loc[panel.index[:30], "stockout_flag"] = True
        assert len(prepare_panel(panel)) == len(panel) - 30

    def test_joins_the_cost_instrument(
        self, clean_panel: pd.DataFrame, cost_index: pd.DataFrame
    ) -> None:
        frame = prepare_panel(clean_panel, costs=cost_index)
        assert "log_cost" in frame.columns
        assert frame["log_cost"].notna().all()


class TestRecovery:
    def test_naive_ols_recovers_truth_when_price_is_exogenous(
        self, clean_panel: pd.DataFrame
    ) -> None:
        """With no endogeneity there is nothing to correct for, so even the
        naive estimator is unbiased. That is what makes the endogenous case
        below attributable to endogeneity rather than to the estimator."""
        estimate = naive_ols(prepare_panel(clean_panel))
        assert estimate.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.15)

    def test_panel_fe_recovers_truth(self, clean_panel: pd.DataFrame) -> None:
        estimate = panel_fixed_effects(prepare_panel(clean_panel))
        assert estimate.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.15)

    def test_randomised_subset_recovers_truth(self, clean_panel: pd.DataFrame) -> None:
        estimate = randomised_subset(prepare_panel(clean_panel))
        assert estimate.elasticity == pytest.approx(TRUE_ELASTICITY, abs=0.30)

    def test_interval_covers_truth(self, clean_panel: pd.DataFrame) -> None:
        estimate = panel_fixed_effects(prepare_panel(clean_panel))
        assert estimate.confidence_interval is not None
        low, high = estimate.confidence_interval
        assert low <= TRUE_ELASTICITY <= high


class TestEndogeneity:
    """Whether fixed effects rescue the estimate depends on *which* signal the
    pricing manager responded to. That distinction is the whole story."""

    def test_naive_is_attenuated_toward_zero(
        self, endogenous_panel: pd.DataFrame
    ) -> None:
        """The headline failure this module exists to prevent.

        A manager who raises price into strong demand makes the product look
        less price-sensitive than it is - and acting on that number means
        cutting price when you should hold it.
        """
        naive = naive_ols(prepare_panel(endogenous_panel))

        assert naive.elasticity > TRUE_ELASTICITY, "expected attenuation toward zero"
        assert abs(naive.elasticity) < abs(TRUE_ELASTICITY)

    def test_fixed_effects_fix_seasonal_endogeneity(
        self, endogenous_panel: pd.DataFrame
    ) -> None:
        """The confounder lives entirely in the date dimension, so time fixed
        effects absorb it. This is the case the platform generator creates, and
        why panel_fe recovers the real elasticities at r=0.99.
        """
        frame = prepare_panel(endogenous_panel)
        naive_error = abs(naive_ols(frame).elasticity - TRUE_ELASTICITY)
        panel_error = abs(panel_fixed_effects(frame).elasticity - TRUE_ELASTICITY)

        assert panel_error < naive_error
        assert panel_fixed_effects(frame).elasticity == pytest.approx(
            TRUE_ELASTICITY, abs=0.20
        )

    def test_fixed_effects_cannot_fix_idiosyncratic_endogeneity(
        self, idiosyncratic_panel: pd.DataFrame
    ) -> None:
        """The documented limitation, tested rather than asserted.

        When price responds to a *daily* demand shock, the confounder varies
        within every dimension being absorbed. Fixed effects have nothing to
        remove, and the estimate stays biased. Anyone reading `panel_fe` as a
        general cure for endogeneity is reading it wrong.
        """
        frame = prepare_panel(idiosyncratic_panel)
        panel_error = abs(panel_fixed_effects(frame).elasticity - TRUE_ELASTICITY)

        assert panel_error > 0.25, (
            "fixed effects should NOT rescue idiosyncratic endogeneity; if this "
            "passes, the fixture is no longer creating it"
        )

    def test_the_estimator_says_so_in_its_warnings(
        self, clean_panel: pd.DataFrame
    ) -> None:
        estimate = panel_fixed_effects(prepare_panel(clean_panel))
        assert any("Does NOT absorb" in w for w in estimate.warnings)

    def test_naive_carries_a_warning_about_its_own_bias(
        self, clean_panel: pd.DataFrame
    ) -> None:
        estimate = naive_ols(prepare_panel(clean_panel))
        assert any("biased toward zero" in w for w in estimate.warnings)


class TestInstrumentGuard:
    def test_flags_an_instrument_with_no_cross_sectional_variation(
        self, clean_panel: pd.DataFrame, cost_index: pd.DataFrame
    ) -> None:
        """The check that catches what the F statistic missed.

        On the real dataset 2SLS had a median first-stage F of 484 and a
        correlation to truth of 0.25. A strong first stage does not make an
        instrument valid.
        """
        frame = prepare_panel(clean_panel, costs=cost_index)
        warnings = instrument_diagnostics(frame)
        assert any("NO CROSS-SECTIONAL VARIATION" in w for w in warnings)

    def test_no_warning_when_the_instrument_varies_within_a_date(
        self, clean_panel: pd.DataFrame, cost_index: pd.DataFrame
    ) -> None:
        frame = prepare_panel(clean_panel, costs=cost_index)
        rng = np.random.default_rng(1)
        frame["log_cost"] = frame["log_cost"] + rng.normal(0, 0.05, len(frame))
        assert not any("NO CROSS-SECTIONAL VARIATION" in w for w in instrument_diagnostics(frame))

    def test_iv_attaches_the_diagnostic_to_its_estimate(
        self, clean_panel: pd.DataFrame, cost_index: pd.DataFrame
    ) -> None:
        estimate = iv_two_stage(prepare_panel(clean_panel, costs=cost_index))
        assert any("NO CROSS-SECTIONAL VARIATION" in w for w in estimate.warnings)
        assert "first_stage_f" in estimate.diagnostics

    def test_iv_refuses_without_a_cost_column(self, clean_panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="no cost index"):
            iv_two_stage(prepare_panel(clean_panel))


class TestSelection:
    def test_prefers_panel_fixed_effects(self, clean_panel: pd.DataFrame) -> None:
        """Preference is by measured recovery, not theoretical appeal - which
        is why 2SLS, the textbook answer, ranks last."""
        estimates = estimate_all(prepare_panel(clean_panel))
        selected, reason = select_estimate(estimates)
        assert selected is not None
        assert selected.method == "panel_fe"
        assert "measured recovery" in reason

    def test_never_selects_a_known_biased_estimator(self) -> None:
        assert "naive_ols" in NOT_SELECTABLE
        assert "iv_2sls" in NOT_SELECTABLE
        assert METHOD_PREFERENCE[0] == "panel_fe"

    def test_returns_none_when_only_biased_methods_ran(
        self, clean_panel: pd.DataFrame
    ) -> None:
        frame = prepare_panel(clean_panel)
        estimates = {"naive_ols": naive_ols(frame)}
        selected, reason = select_estimate(estimates)
        assert selected is None
        assert "biased" in reason

    def test_empty_input_is_handled(self) -> None:
        selected, reason = select_estimate({})
        assert selected is None
        assert "no estimator" in reason


class TestComparison:
    def test_table_scores_against_truth_when_given(
        self, clean_panel: pd.DataFrame
    ) -> None:
        estimates = estimate_all(prepare_panel(clean_panel))
        table = comparison_table(estimates, truth=TRUE_ELASTICITY)
        assert "error" in table.columns
        assert "covers_truth" in table.columns
        assert table["truth"].eq(TRUE_ELASTICITY).all()

    def test_table_marks_unselectable_methods(self, clean_panel: pd.DataFrame) -> None:
        table = comparison_table(estimate_all(prepare_panel(clean_panel)))
        naive = table[table["method"] == "naive_ols"].iloc[0]
        assert not naive["selectable"]

    def test_failed_methods_are_omitted_not_faked(
        self, clean_panel: pd.DataFrame
    ) -> None:
        """No cost index joined, so 2SLS cannot run and simply is not there."""
        estimates = estimate_all(prepare_panel(clean_panel))
        assert "iv_2sls" not in estimates
        assert "panel_fe" in estimates


class TestIdentification:
    def test_warns_on_too_few_price_points(self) -> None:
        panel = make_panel(n_stores=2, n_days=80)
        frame = prepare_panel(panel)
        frame["log_price"] = np.log(10.0)
        assert any("distinct price points" in w for w in check_identification(frame))

    def test_warns_on_no_price_variation(self) -> None:
        panel = make_panel(n_stores=2, n_days=80)
        frame = prepare_panel(panel)
        frame["log_price"] = np.log(10.0)
        assert any("almost no price variation" in w for w in check_identification(frame))

    def test_clean_panel_has_no_identification_warnings(
        self, clean_panel: pd.DataFrame
    ) -> None:
        assert check_identification(prepare_panel(clean_panel)) == []


class TestElasticFlag:
    def test_elastic_above_one(self, clean_panel: pd.DataFrame) -> None:
        """|e| > 1 means a price rise reduces revenue. This single bit
        determines the direction of every pricing recommendation."""
        estimate = panel_fixed_effects(prepare_panel(clean_panel))
        assert estimate.is_elastic

    def test_inelastic_below_one(self) -> None:
        panel = make_panel(elasticity=-0.4, n_stores=8, n_days=300, seed=5)
        estimate = panel_fixed_effects(prepare_panel(panel))
        assert not estimate.is_elastic
