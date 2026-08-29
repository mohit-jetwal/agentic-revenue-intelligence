"""Placebo, sensitivity and the validation verdict (sections 22-24)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.controls import build_control_pool
from ml.promo_uplift.diagnostics import (
    PlaceboResult,
    SensitivityResult,
    SensitivityRow,
    evaluate_placebo,
    expected_effect_from_ground_truth,
    judge,
    placebo_frame,
)
from ml.promo_uplift.estimators import AIPWEstimator, EffectEstimate, fit_nuisances
from ml.promo_uplift.features import build_covariates
from ml.promo_uplift.matching import BalanceReport, BalanceRow
from ml.promo_uplift.treatment import AnalysisFrame, RowRole

pytestmark = [pytest.mark.causal, pytest.mark.models]


class TestPlaceboFrame:
    def test_real_treatment_rows_are_removed(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        """Leaving them in the control pool would put genuinely promoted days on
        the other side of the comparison, and the placebo would find a large
        negative effect for an entirely mechanical reason.
        """
        shifted = placebo_frame(analysis, config=confounded_config)
        real_treated = analysis.frame.loc[
            analysis.frame["role"] == RowRole.TREATED, ["product_id", "store_id", "date"]
        ]
        keys = set(map(tuple, real_treated.to_numpy()))
        placebo_keys = set(
            map(tuple, shifted.frame[["product_id", "store_id", "date"]].to_numpy())
        )
        assert not keys & placebo_keys

    def test_events_move_backwards_by_the_configured_shift(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        shifted = placebo_frame(analysis, config=confounded_config)
        offset = pd.Timedelta(days=confounded_config.validation.placebo_shift_days)

        original = analysis.events.set_index("promotion_id")["start_date"]
        moved = shifted.events.set_index("promotion_id")["start_date"]
        common = original.index.intersection(moved.index)
        assert (
            pd.to_datetime(original[common]) - pd.to_datetime(moved[common])
        ).eq(offset).all()

    def test_the_frame_still_has_both_arms(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        shifted = placebo_frame(analysis, config=confounded_config)
        assert shifted.treated_rows > 0
        assert shifted.control_rows > 0

    def test_placebo_finds_no_effect(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        """The closest thing causal inference has to a unit test.

        The true effect in the shifted window is zero by construction. Anything
        found there is attributable to the method, not the promotion.
        """
        shifted = placebo_frame(analysis, config=confounded_config)
        pool = build_control_pool(shifted, config=confounded_config)
        covariates = build_covariates(
            pool.frame, shifted.events, config=confounded_config, history=shifted.frame
        )
        nuisance = fit_nuisances(covariates, config=confounded_config)
        estimate = (
            AIPWEstimator(config=confounded_config)
            .fit(covariates, nuisance)
            .estimate_ate()
        )

        result = evaluate_placebo(estimate, config=confounded_config)
        assert result.passed, f"placebo found {result.effect_pct:+.2%}"


class TestPlaceboEvaluation:
    def test_a_large_effect_fails(self) -> None:
        estimate = EffectEstimate(
            method="augmented_ipw", ate=10.0, ate_pct=0.40, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        assert not evaluate_placebo(estimate).passed

    def test_context_against_the_real_estimate_is_reported(self) -> None:
        """A placebo of +2% next to a real +60% is reassuring; the same +2% next
        to a real +3% is not, and a bare pass/fail hides the difference."""
        placebo = EffectEstimate(
            method="augmented_ipw", ate=0.5, ate_pct=0.02, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        reference = EffectEstimate(
            method="augmented_ipw", ate=15.0, ate_pct=0.60, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        result = evaluate_placebo(placebo, reference=reference)
        assert result.ratio_to_reference == pytest.approx(0.02 / 0.60)


class TestSensitivity:
    def test_spread_is_relative_to_the_headline(self) -> None:
        """A 5-point spread around 60% is robustness; the same spread around 4%
        means the specification is doing the work."""
        rows = [
            SensitivityRow("washout_days", 0, 0.58, 100),
            SensitivityRow("washout_days", 10, 0.63, 100),
        ]
        result = SensitivityResult(rows=rows, reference_pct=0.60)
        assert result.spread() == pytest.approx(0.05)
        assert result.relative_spread() == pytest.approx(0.05 / 0.60)

    def test_failed_specifications_are_excluded_from_the_spread(self) -> None:
        rows = [
            SensitivityRow("control_window_days", 21, 0.58, 100),
            SensitivityRow("control_window_days", 45, 0.62, 100),
            SensitivityRow("control_window_days", 90, float("nan"), 0, failed="no controls"),
        ]
        result = SensitivityResult(rows=rows, reference_pct=0.60)
        assert len(result.usable) == 2
        assert result.spread() == pytest.approx(0.04)


class TestVerdict:
    def _balance(self, *, smd: float) -> BalanceReport:
        return BalanceReport(
            rows=[
                BalanceRow("demand_mean_28", 10.0, 9.0, 9.5, 0.30, smd),
            ],
            threshold=0.10,
            n_treated=100,
            n_control=400,
        )

    def test_balance_failure_blocks_ipw(self) -> None:
        """IPW has nothing but the propensity model. Unbalanced covariates mean
        the comparison is simply not adjusted."""
        estimate = EffectEstimate(
            method="inverse_probability_weighting", ate=5.0, ate_pct=0.2,
            baseline_units=25.0, n_treated=100, n_control=400,
        )
        verdict = judge(
            estimate=estimate,
            balance=self._balance(smd=0.40),
            overlap_warnings=[],
            placebo=None,
            sensitivity=None,
        )
        assert verdict.status == "failed"
        assert verdict.blocking

    def test_balance_failure_only_warns_for_aipw(self) -> None:
        """The doubly robust property, applied. AIPW stays consistent if the
        outcome model is right, so poor propensity balance is a caution rather
        than a disqualification - and this was observed directly: on the
        confounded panel the worst SMD was 0.38 and AIPW still recovered the
        true effect.
        """
        estimate = EffectEstimate(
            method="augmented_ipw", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        verdict = judge(
            estimate=estimate,
            balance=self._balance(smd=0.40),
            overlap_warnings=[],
            placebo=None,
            sensitivity=None,
        )
        assert verdict.status == "warnings"
        assert not verdict.blocking

    def test_overlap_violation_blocks_every_method(self) -> None:
        estimate = EffectEstimate(
            method="augmented_ipw", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        verdict = judge(
            estimate=estimate,
            balance=self._balance(smd=0.02),
            overlap_warnings=["30% of rows fall outside the propensity range"],
            placebo=None,
            sensitivity=None,
        )
        assert verdict.status == "failed"

    def test_placebo_failure_blocks(self) -> None:
        estimate = EffectEstimate(
            method="augmented_ipw", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        verdict = judge(
            estimate=estimate,
            balance=self._balance(smd=0.02),
            overlap_warnings=[],
            placebo=PlaceboResult(
                effect_pct=0.25, threshold=0.05, n_treated=100, shift_days=30
            ),
            sensitivity=None,
        )
        assert verdict.status == "failed"
        assert any("placebo" in b for b in verdict.blocking)

    def test_everything_clean_passes(self) -> None:
        estimate = EffectEstimate(
            method="augmented_ipw", ate=5.0, ate_pct=0.2, baseline_units=25.0,
            n_treated=100, n_control=400,
        )
        verdict = judge(
            estimate=estimate,
            balance=self._balance(smd=0.02),
            overlap_warnings=[],
            placebo=PlaceboResult(
                effect_pct=0.01, threshold=0.05, n_treated=100, shift_days=30
            ),
            sensitivity=SensitivityResult(
                rows=[SensitivityRow("washout_days", 10, 0.20, 100)], reference_pct=0.20
            ),
        )
        assert verdict.status == "passed"


class TestGroundTruthComparison:
    def test_returns_none_without_ground_truth_files(self, tmp_path) -> None:
        events = pd.DataFrame(
            {"product_id": ["P1"], "discount_depth": [0.2], "promotion_type": ["Display"]}
        )
        assert expected_effect_from_ground_truth(events, tmp_path) is None

    def test_expected_effect_combines_both_channels(self, tmp_path) -> None:
        """A promotion moves demand through the mechanic AND the price cut.

        Reporting only the mechanic would measure the smaller half - at a 20%
        depth the price channel is usually the larger of the two.
        """
        import json

        (tmp_path / "promotion_uplift.json").write_text(
            json.dumps({"values": {"P1": {"Display": {"a": 0.30, "b": 4.0}}}}),
            encoding="utf-8",
        )
        (tmp_path / "elasticity.json").write_text(
            json.dumps({"values": {"P1": -1.5}}), encoding="utf-8"
        )

        events = pd.DataFrame(
            {"product_id": ["P1"], "discount_depth": [0.20], "promotion_type": ["Display"]}
        )
        result = expected_effect_from_ground_truth(events, tmp_path)

        assert result is not None
        assert result.mechanic_pct > 0
        assert result.price_channel_pct > 0
        # The combined effect is multiplicative, so it exceeds either channel.
        assert result.expected_pct > max(result.mechanic_pct, result.price_channel_pct)

    def test_caveats_state_what_cannot_be_validated(self, tmp_path) -> None:
        import json

        (tmp_path / "promotion_uplift.json").write_text(
            json.dumps({"values": {"P1": {"Display": {"a": 0.30, "b": 4.0}}}}),
            encoding="utf-8",
        )
        (tmp_path / "elasticity.json").write_text(
            json.dumps({"values": {"P1": -1.5}}), encoding="utf-8"
        )
        events = pd.DataFrame(
            {"product_id": ["P1"], "discount_depth": [0.20], "promotion_type": ["Display"]}
        )
        result = expected_effect_from_ground_truth(events, tmp_path)

        assert result is not None
        assert any("AVERAGE" in c for c in result.caveats)


class TestConfigGuards:
    def test_placebo_shift_must_clear_the_washout(self) -> None:
        """Otherwise the placebo window overlaps real pull-forward effects and
        the test fails for the wrong reason.

        Caught at config load rather than at run time - a specification that
        cannot produce a meaningful placebo should never reach an estimator.
        """
        from pydantic import ValidationError

        from ml.promo_uplift.config import PromoUpliftConfig

        payload = get_promo_uplift_config().model_dump()
        payload["validation"]["placebo_shift_days"] = 3

        with pytest.raises(ValidationError, match="washout"):
            PromoUpliftConfig.model_validate(payload)

    def test_control_window_must_clear_the_washout(self) -> None:
        """Control rows drawn from the pull-forward dip are depressed by the
        very promotion whose effect they anchor, which biases uplift upward."""
        from pydantic import ValidationError

        from ml.promo_uplift.config import PromoUpliftConfig

        payload = get_promo_uplift_config().model_dump()
        payload["controls"]["same_series_window_days"] = 5
        payload["treatment"]["washout_days"] = 10

        with pytest.raises(ValidationError, match="pull-forward"):
            PromoUpliftConfig.model_validate(payload)
