"""Validation against known demand - the tests a real project cannot write.

Every other accuracy test in this suite compares predictions to *observed sales*,
which is the quantity the model was fitted on. That can only ever confirm the fit
was competent; it cannot detect that the model is confidently measuring the wrong
thing. A baseline that learned inventory censoring as demand scores *better*
against observed sales than the correct model does, because it is reproducing the
censoring too.

Here the truth is known - the fixture built the demand curve and then censored it
- so these tests check what the model is measuring rather than how closely it
reproduces the till.

The headline is :class:`TestStockoutCensoring`. It is the one that would catch the
error that silently inverts every downstream root-cause conclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline.evaluation import compute_metrics, irreducible_error
from ml.baseline.models import build_estimator
from ml.baseline.training import (
    PromotionApproach,
    build_temporal_split,
    prepare_training_rows,
    select_features,
    train_baseline,
)
from tests.model_fixtures import (
    PROMO_LIFT,
    STOCKOUT_AVAILABILITY,
)

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def fitted(feature_panel: pd.DataFrame):
    """A LightGBM baseline trained the way the pipeline trains it."""
    split = build_temporal_split(feature_panel)
    estimator = build_estimator("lightgbm", seed=7)
    return train_baseline(
        feature_panel, estimator, approach=PromotionApproach.EXCLUDE, split=split
    )


@pytest.fixture(scope="module")
def scored(
    fitted, feature_panel: pd.DataFrame, synthetic_panel: pd.DataFrame
) -> pd.DataFrame:
    """Test-window rows with predictions and ground truth side by side."""
    split = fitted.split
    dates = pd.to_datetime(feature_panel["date"]).dt.date
    mask = (dates >= split.test_start) & (dates <= split.test_end)

    rows = feature_panel[mask].copy()
    rows["baseline_units"] = fitted.predict_baseline(rows)
    rows["latent_units"] = synthetic_panel.loc[mask, "latent_units"].to_numpy()
    rows["true_baseline"] = synthetic_panel.loc[mask, "true_baseline"].to_numpy()
    return rows


class TestStockoutCensoring:
    """Did the model learn demand, or did it learn the supply failure?"""

    def test_baseline_exceeds_censored_sales_during_stockouts(
        self, scored: pd.DataFrame
    ) -> None:
        """The headline result of the whole step.

        On a stockout day the fixture allowed only 35% of demand to be sold. A
        model that learned demand must predict well above what was recorded; a
        model that learned the censoring predicts roughly what was recorded, and
        would report a supply failure as a demand collapse.
        """
        stockouts = scored[scored["stockout_flag"].astype(bool)]
        assert len(stockouts) > 50, "too few stockout rows to conclude anything"

        lift = stockouts["baseline_units"].sum() / stockouts["units"].sum()

        assert lift > 1.5, (
            f"baseline is only {lift:.2f}x censored sales during stockouts - "
            f"the model appears to have learned inventory availability as demand"
        )

    def test_baseline_tracks_true_demand_during_stockouts(
        self, scored: pd.DataFrame
    ) -> None:
        """And it must land near *actual* demand, not merely above the till.

        The previous test alone could be satisfied by a model that
        over-predicts everywhere. This one pins the level: on stockout rows the
        baseline should approximate the latent demand that existed.

        The fixture's stockouts are random rather than demand-driven, so unlike
        the generated dataset - where stockouts follow demand spikes and the
        ratio is legitimately below 1 - the expected ratio here is near 1.
        """
        stockouts = scored[scored["stockout_flag"].astype(bool)]

        ratio = stockouts["baseline_units"].sum() / stockouts["latent_units"].sum()

        assert 0.75 < ratio < 1.25, (
            f"baseline recovers {ratio:.2f}x true demand during stockouts"
        )

    def test_predicting_observed_sales_would_score_far_worse(
        self, scored: pd.DataFrame
    ) -> None:
        """Makes the failure mode concrete rather than hypothetical.

        Scores the model against true demand, then scores the censored sales
        themselves against true demand. The second is what a model trained
        without the stockout filter would converge toward - and it must be
        clearly worse, otherwise the filter is not buying anything.
        """
        stockouts = scored[scored["stockout_flag"].astype(bool)]

        model = compute_metrics(stockouts["latent_units"], stockouts["baseline_units"])
        censored = compute_metrics(stockouts["latent_units"], stockouts["units"])

        assert model.wmape < censored.wmape, (
            f"model WMAPE {model.wmape:.1%} is no better than simply echoing "
            f"censored sales ({censored.wmape:.1%})"
        )
        # The censored series should be biased low by roughly the availability
        # the fixture imposed, which is what makes it such a damaging target.
        assert censored.bias_pct == pytest.approx(STOCKOUT_AVAILABILITY - 1.0, abs=0.10)


class TestCleanRowAccuracy:
    def test_baseline_is_accurate_on_clean_rows(self, scored: pd.DataFrame) -> None:
        """No promotion, no stockout - observed sales *are* the baseline here."""
        clean = scored[
            ~scored["stockout_flag"].astype(bool) & ~scored["promotion_flag"].astype(bool)
        ]

        metrics = compute_metrics(clean["latent_units"], clean["baseline_units"])

        assert metrics.wmape < 0.20, f"clean-row WMAPE {metrics.wmape:.1%} is too high"

    def test_baseline_is_not_systematically_biased(self, scored: pd.DataFrame) -> None:
        """Bias matters more than error for this model's purpose.

        Random error averages out when uplift is aggregated over a campaign; a
        consistent 5% under-prediction does not, and turns directly into 5%
        phantom uplift on every promotion measured against it.
        """
        clean = scored[
            ~scored["stockout_flag"].astype(bool) & ~scored["promotion_flag"].astype(bool)
        ]

        metrics = compute_metrics(clean["latent_units"], clean["baseline_units"])

        assert abs(metrics.bias_pct) < 0.05, (
            f"baseline is biased by {metrics.bias_pct:+.1%} on clean rows, which "
            f"would appear as phantom uplift on every promotion"
        )

    def test_accuracy_is_within_reach_of_the_noise_floor(
        self, scored: pd.DataFrame
    ) -> None:
        """Contextualises the error rather than asserting an arbitrary number.

        The fixture adds 10% multiplicative noise, so no model can score near
        zero. Comparing against the achievable floor is a meaningful statement;
        comparing against a hard-coded threshold is a statement about how the
        fixture happens to be tuned.
        """
        clean = scored[
            ~scored["stockout_flag"].astype(bool) & ~scored["promotion_flag"].astype(bool)
        ]

        model = compute_metrics(clean["latent_units"], clean["baseline_units"])
        floor = compute_metrics(clean["latent_units"], clean["true_baseline"])

        assert model.wmape < floor.wmape * 2.0, (
            f"model WMAPE {model.wmape:.1%} against a noise floor of "
            f"{floor.wmape:.1%} - more than double the achievable error"
        )


class TestPromotionalRows:
    """Directional only - see the step's documented honest limit."""

    def test_actual_sales_exceed_baseline_on_promotions(
        self, scored: pd.DataFrame
    ) -> None:
        """The uplift signal exists and points the right way.

        A baseline that failed this would make every promotion look ineffective,
        which is the error mode that destroys trust in the platform fastest.
        """
        promoted = scored[
            scored["promotion_flag"].astype(bool) & ~scored["stockout_flag"].astype(bool)
        ]
        assert len(promoted) > 50

        assert promoted["units"].sum() > promoted["baseline_units"].sum()

    def test_measured_uplift_is_in_the_right_neighbourhood(
        self, scored: pd.DataFrame
    ) -> None:
        """The fixture applied a known multiplicative lift; recover it roughly.

        Loose bounds on purpose. The Approach C selection bias is present in the
        fixture by design - promotions are targeted at seasonal peaks - so the
        measured lift is expected to overstate the truth somewhat. Asserting a
        tight interval would be asserting that a known bias does not exist.
        """
        promoted = scored[
            scored["promotion_flag"].astype(bool) & ~scored["stockout_flag"].astype(bool)
        ]

        measured = promoted["units"].sum() / promoted["baseline_units"].sum()

        assert 1.3 < measured < 2.4, (
            f"measured lift {measured:.2f}x is nowhere near the {PROMO_LIFT}x the "
            f"fixture applied"
        )

    def test_baseline_is_below_promotional_actuals_for_most_rows(
        self, scored: pd.DataFrame
    ) -> None:
        """Not just in aggregate - the direction should hold row by row.

        An aggregate that passes while half the rows point the wrong way would
        mean the uplift is driven by a few outliers rather than a real effect.
        """
        promoted = scored[
            scored["promotion_flag"].astype(bool) & ~scored["stockout_flag"].astype(bool)
        ]

        share = (promoted["units"] > promoted["baseline_units"]).mean()

        assert share > 0.65, f"only {share:.0%} of promoted rows exceed baseline"


class TestNoiseFloor:
    def test_irreducible_error_is_computed_from_the_true_mean(
        self, latent_frame: pd.DataFrame
    ) -> None:
        floor = irreducible_error(latent_frame)

        assert floor is not None
        assert floor.wmape > 0, "a noise floor of zero would mean the data is noiseless"
        assert abs(floor.bias_pct) < 0.05, "the true mean should be unbiased by definition"

    def test_returns_none_when_ground_truth_is_absent(self) -> None:
        """Production has no ``mean_demand`` column, and that is not an error."""
        assert irreducible_error(pd.DataFrame()) is None
        assert irreducible_error(pd.DataFrame({"latent_units": [1.0, 2.0]})) is None

    def test_no_model_beats_the_floor(self, scored: pd.DataFrame) -> None:
        """A sanity check that doubles as a leakage detector.

        Scoring below the irreducible error is not a triumph - it is impossible
        without seeing the target. If this ever fails, something leaked.
        """
        clean = scored[
            ~scored["stockout_flag"].astype(bool) & ~scored["promotion_flag"].astype(bool)
        ]

        model = compute_metrics(clean["latent_units"], clean["baseline_units"])
        floor = compute_metrics(clean["latent_units"], clean["true_baseline"])

        assert model.wmape > floor.wmape * 0.9, (
            f"model WMAPE {model.wmape:.1%} is at or below the theoretical floor "
            f"{floor.wmape:.1%} - suspect target leakage"
        )


class TestApproachComparison:
    def test_both_approaches_produce_usable_baselines(
        self, feature_panel: pd.DataFrame, synthetic_panel: pd.DataFrame
    ) -> None:
        """Neither approach is allowed to be catastrophically wrong.

        Which one wins is an empirical question answered by the pipeline against
        real data. What this test asserts is weaker and more durable: both are
        implemented correctly enough to compete, so the comparison the pipeline
        performs is a genuine one rather than a walkover caused by a bug.
        """
        split = build_temporal_split(feature_panel)
        dates = pd.to_datetime(feature_panel["date"]).dt.date
        mask = (dates >= split.test_start) & (dates <= split.test_end)
        latent = synthetic_panel.loc[mask, "latent_units"].to_numpy()

        results: dict[str, float] = {}
        for approach in PromotionApproach:
            trained = train_baseline(
                feature_panel,
                build_estimator("lightgbm", seed=7),
                approach=approach,
                split=split,
            )
            rows = feature_panel[mask]
            clean = ~rows["stockout_flag"].astype(bool).to_numpy()
            predictions = trained.predict_baseline(rows)
            results[approach.value] = compute_metrics(
                pd.Series(latent[clean]), pd.Series(predictions[clean])
            ).wmape

        for approach, wmape in results.items():
            assert wmape < 0.35, f"{approach} scores WMAPE {wmape:.1%}, which is broken"

    def test_exclude_approach_never_trains_on_a_promoted_row(
        self, feature_panel: pd.DataFrame
    ) -> None:
        """The property that gives Approach C its counterfactual meaning."""
        rows, _ = prepare_training_rows(
            feature_panel, approach=PromotionApproach.EXCLUDE
        )
        features = select_features(feature_panel, approach=PromotionApproach.EXCLUDE)

        assert not rows["promotion_flag"].astype(bool).any()
        assert "promotion_flag" not in features


class TestPredictionSanity:
    def test_predictions_are_never_negative(self, scored: pd.DataFrame) -> None:
        """Negative baseline units are meaningless and break downstream maths."""
        assert (scored["baseline_units"] >= 0).all()

    def test_predictions_are_finite(self, scored: pd.DataFrame) -> None:
        assert np.isfinite(scored["baseline_units"]).all()

    def test_predictions_vary(self, scored: pd.DataFrame) -> None:
        """A constant prediction would pass several metrics and be useless."""
        assert scored["baseline_units"].std() > 1.0
