"""Metrics, and the model-selection rules built on them.

Two things are being protected here.

**The metrics themselves.** WMAPE is the headline because it is volume-weighted:
a 50% error on a hero SKU selling 10,000 units matters more than a 50% error on
a tail SKU selling three, and plain MAPE says they are identical. MAPE is still
reported, but only over non-zero actuals, with the excluded count alongside -
because ``actual = 0`` makes the ratio infinite and an ``inf`` silently poisoning
a mean is worse than an honestly absent number.

**The selection rules.** Which model wins is decided by code, and that code
encodes judgement: correctness outranks accuracy, and simplicity wins ties. Those
rules deserve tests as much as the arithmetic does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline.comparison import STOCKOUT_LIFT_FLOOR, Candidate, select_model
from ml.baseline.evaluation import BaselineMetrics, compute_metrics

pytestmark = pytest.mark.models


class TestMetrics:
    def test_perfect_predictions_score_zero_error(self) -> None:
        actual = pd.Series([10.0, 20.0, 30.0])

        metrics = compute_metrics(actual, actual)

        assert metrics.mae == pytest.approx(0.0)
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.wmape == pytest.approx(0.0)
        assert metrics.bias == pytest.approx(0.0)

    def test_wmape_weights_by_volume(self) -> None:
        """The property that makes WMAPE the right headline.

        Two series with identical *relative* errors but different volumes must
        score differently: the error on the high-volume row is the one that
        moves revenue.
        """
        actual = pd.Series([1000.0, 10.0])

        big_row_wrong = compute_metrics(actual, pd.Series([500.0, 10.0]))
        small_row_wrong = compute_metrics(actual, pd.Series([1000.0, 5.0]))

        assert big_row_wrong.wmape > small_row_wrong.wmape

    def test_bias_is_signed(self) -> None:
        """Direction matters more than magnitude for a baseline.

        Over- and under-prediction have opposite consequences downstream: one
        hides real uplift, the other invents it. A metric that reported only
        magnitude would treat those as the same failure.
        """
        actual = pd.Series([100.0, 100.0])

        over = compute_metrics(actual, pd.Series([120.0, 120.0]))
        under = compute_metrics(actual, pd.Series([80.0, 80.0]))

        assert over.bias > 0
        assert under.bias < 0
        assert over.bias_pct == pytest.approx(0.2)
        assert under.bias_pct == pytest.approx(-0.2)

    def test_mape_excludes_zero_actuals_and_reports_the_count(self) -> None:
        """A zero actual makes the ratio infinite - it must be excluded, and the
        exclusion must be visible.

        Silently dropping them would overstate accuracy on an intermittent-demand
        panel, where zero-sales days are common and are exactly the rows a model
        finds hardest.
        """
        actual = pd.Series([0.0, 0.0, 100.0, 200.0])
        predicted = pd.Series([5.0, 5.0, 110.0, 180.0])

        metrics = compute_metrics(actual, predicted)

        assert metrics.mape_excluded == 2
        assert metrics.mape is not None
        assert np.isfinite(metrics.mape)

    def test_mape_is_none_when_every_actual_is_zero(self) -> None:
        """No defensible value exists, so none is reported."""
        metrics = compute_metrics(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))

        assert metrics.mape is None
        assert metrics.mape_excluded == 2

    def test_rmse_punishes_large_errors_more_than_mae(self) -> None:
        actual = pd.Series([100.0, 100.0, 100.0, 100.0])
        spread = compute_metrics(actual, pd.Series([100.0, 100.0, 100.0, 200.0]))

        assert spread.rmse > spread.mae

    def test_totals_are_reported_for_reconciliation(self) -> None:
        """Aggregates let a caller sanity-check a metric against known volume."""
        metrics = compute_metrics(pd.Series([10.0, 20.0]), pd.Series([12.0, 18.0]))

        assert metrics.actual_total == pytest.approx(30.0)
        assert metrics.predicted_total == pytest.approx(30.0)
        assert metrics.n == 2

    def test_empty_input_does_not_raise(self) -> None:
        """Empty slices happen - a filtered segment with no rows is normal."""
        metrics = compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))

        assert metrics.n == 0


def _candidate(
    name: str,
    *,
    latent_wmape: float,
    stockout_lift: float,
    estimator_name: str | None = None,
) -> Candidate:
    """A Candidate stub carrying only what the selection rules read.

    Built with simple namespaces rather than trained models: selection is pure
    decision logic over metrics, and fitting three real estimators to test a
    tie-break would make these tests slow and their failures ambiguous.
    """
    estimator_name = estimator_name or name

    class _Stub(Candidate):  # type: ignore[misc]
        def __init__(self) -> None:
            self.trained = type(
                "T",
                (),
                {
                    "name": name,
                    "estimator": type("E", (), {"name": estimator_name})(),
                    "approach": type("A", (), {"value": "exclude"})(),
                    "metrics": {},
                    "coverage": None,
                },
            )()
            self.backtest = None
            self.latent_metrics = {}
            self._latent_wmape = latent_wmape
            self._stockout_lift = stockout_lift

        @property
        def name(self) -> str:
            return name

        @property
        def latent_wmape(self) -> float:
            return self._latent_wmape

        @property
        def test_wmape(self) -> float:
            return self._latent_wmape

        @property
        def stockout_lift(self) -> float:
            return self._stockout_lift

        @property
        def stockout_vs_latent_ratio(self) -> float:
            return float("nan")

    return _Stub()


class TestModelSelection:
    def test_most_accurate_candidate_wins(self) -> None:
        result = select_model(
            [
                _candidate("ridge", latent_wmape=0.30, stockout_lift=2.0),
                _candidate("lightgbm", latent_wmape=0.20, stockout_lift=2.0),
            ]
        )

        assert result.selected.name == "lightgbm"

    def test_a_model_that_learned_censoring_is_disqualified_despite_winning_on_accuracy(
        self,
    ) -> None:
        """Correctness outranks accuracy - the most important rule here.

        A model that learned inventory censoring scores *well* against observed
        sales precisely because it reproduces the censoring. Ranking on accuracy
        alone would therefore select it, and every downstream root-cause
        conclusion would be inverted.
        """
        result = select_model(
            [
                _candidate("leaky", latent_wmape=0.10, stockout_lift=1.0),
                _candidate("honest", latent_wmape=0.30, stockout_lift=2.0),
            ]
        )

        assert result.selected.name == "honest"
        assert any("disqualified" in reason for reason in result.rationale)

    def test_the_disqualification_threshold_is_the_documented_one(self) -> None:
        """Guards the boundary rather than a value comfortably inside it."""
        just_under = select_model(
            [
                _candidate("a", latent_wmape=0.10, stockout_lift=STOCKOUT_LIFT_FLOOR - 0.01),
                _candidate("b", latent_wmape=0.30, stockout_lift=2.0),
            ]
        )
        just_over = select_model(
            [
                _candidate("a", latent_wmape=0.10, stockout_lift=STOCKOUT_LIFT_FLOOR),
                _candidate("b", latent_wmape=0.30, stockout_lift=2.0),
            ]
        )

        assert just_under.selected.name == "b"
        assert just_over.selected.name == "a"

    def test_seasonal_naive_wins_when_the_gap_is_marginal(self) -> None:
        """Section 41: the most complex model does not win by default.

        Two percentage points of WMAPE do not justify fifty times the training
        cost and a model nobody can explain in a meeting.
        """
        result = select_model(
            [
                _candidate(
                    "seasonal_naive",
                    latent_wmape=0.215,
                    stockout_lift=2.0,
                    estimator_name="seasonal_naive",
                ),
                _candidate(
                    "lightgbm", latent_wmape=0.20, stockout_lift=2.0, estimator_name="lightgbm"
                ),
            ],
            complexity_tolerance=0.02,
        )

        assert result.selected.trained.estimator.name == "seasonal_naive"

    def test_complexity_is_justified_when_the_gap_is_large(self) -> None:
        result = select_model(
            [
                _candidate(
                    "seasonal_naive",
                    latent_wmape=0.40,
                    stockout_lift=2.0,
                    estimator_name="seasonal_naive",
                ),
                _candidate(
                    "lightgbm", latent_wmape=0.20, stockout_lift=2.0, estimator_name="lightgbm"
                ),
            ],
            complexity_tolerance=0.02,
        )

        assert result.selected.trained.estimator.name == "lightgbm"
        assert any("earning its place" in reason for reason in result.rationale)

    def test_total_failure_is_reported_rather_than_hidden(self) -> None:
        """When every candidate fails the correctness check, say so.

        Returning the least-bad model with a confident rationale would be the
        worst possible outcome: the pipeline would look successful while every
        number it produced was measuring the wrong quantity.
        """
        result = select_model(
            [
                _candidate("a", latent_wmape=0.10, stockout_lift=1.0),
                _candidate("b", latent_wmape=0.20, stockout_lift=1.0),
            ]
        )

        assert any("No candidate passed" in reason for reason in result.rationale)

    def test_missing_ground_truth_does_not_disqualify_anyone(self) -> None:
        """Production has no latent demand, and absence of evidence is not
        evidence of failure. The check must abstain, not reject."""
        result = select_model([_candidate("a", latent_wmape=0.25, stockout_lift=float("nan"))])

        assert result.selected.name == "a"
        assert not any("disqualified" in reason for reason in result.rationale)

    def test_selecting_from_nothing_raises(self) -> None:
        with pytest.raises(ValueError):
            select_model([])

    def test_rationale_is_never_empty(self) -> None:
        """Every selection must explain itself.

        The comparison table is a deliverable of this step in its own right; a
        chosen model with no stated reason is not reviewable.
        """
        result = select_model([_candidate("a", latent_wmape=0.25, stockout_lift=2.0)])

        assert result.rationale


class TestMetricsSummary:
    def test_summary_includes_the_headline_numbers(self) -> None:
        metrics = BaselineMetrics(
            n=100, mae=5.0, rmse=7.0, wmape=0.12, bias=1.0, bias_pct=0.02,
            mape=0.15, mape_excluded=3, actual_total=1000.0, predicted_total=1020.0,
        )

        summary = metrics.summary()

        assert "12.0%" in summary
        assert "100" in summary
