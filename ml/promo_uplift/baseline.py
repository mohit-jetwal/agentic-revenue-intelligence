"""The naive estimator and the baseline-model counterfactual (brief section 10).

Two estimators, and the first one is deliberately wrong.

**NaiveDuringVsBefore** compares sales during a promotion against sales just
before it. This is how promotional performance is reported in most of the
industry, and it is the number this whole capability exists to replace. It is
kept, run and published alongside the others because "the naive method
overstates uplift" is an assertion until you show it, next to the same data, with
the size of the gap on the page. On the confounded synthetic panel it comes back
at +108% against a true +63%.

It fails for two compounding reasons:

* **Seasonal targeting.** Promotions are scheduled when demand is expected to be
  strong, so the days before are systematically weaker than the promotion days
  would have been anyway. The comparison attributes the season to the promotion.
* **Pull-forward.** Shoppers load their pantry, so the days *after* dip. The
  naive window ends before that dip, banking the borrowed volume as incremental.

**BaselineCounterfactual** reuses the Step 5 baseline sales model, which was
trained with ``PromotionApproach.EXCLUDE`` - it has literally never seen a
promotional row, so its prediction on a promoted day *is* the no-promotion
expectation. That is a genuinely different and much better answer, and it costs
nothing extra because the artifact already exists.

Its weakness is documented in Step 5 itself: the selected LightGBM baseline
over-predicts by about 6.7%. Uplift measured against it is therefore understated
by roughly that much. Step 5 flagged the trade-off and left it to "whoever owns
the uplift numbers"; this module owns them, so the bias is corrected explicitly
and the correction is reported rather than folded in silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.exceptions import EstimationError, UpliftModelUnavailableError
from ml.promo_uplift.treatment import DATE, KEYS, AnalysisFrame, RowRole

logger = get_logger(__name__)

#: Measured over-prediction of the Step 5 baseline against true latent demand,
#: from ``docs/models/baseline_sales.md``. Applied as a correction and reported,
#: never absorbed.
BASELINE_BIAS = 0.067


@dataclass
class NaiveEstimator:
    """Sales during the promotion versus the days immediately before it.

    Present as a benchmark and a warning, not as a candidate. The pipeline
    reports it first so every other number is read against it.
    """

    name: str = "naive_during_vs_before"
    lookback_days: int = 14

    def estimate(
        self, analysis: AnalysisFrame, *, config: PromoUpliftConfig | None = None
    ) -> EffectEstimate:
        settings = config or get_promo_uplift_config()
        panel = analysis.frame
        events = analysis.events

        if events.empty:
            raise EstimationError("no qualifying events", method=self.name)

        during: list[float] = []
        before: list[float] = []

        starts = pd.to_datetime(events["start_date"])
        indexed = panel.set_index([*KEYS, DATE]).sort_index()

        for event, start in zip(events.to_dict("records"), starts, strict=True):
            key = (str(event["product_id"]), str(event["store_id"]))
            try:
                series = indexed.loc[key]
            except KeyError:
                continue
            if not isinstance(series, pd.DataFrame):
                # A listing with a single row collapses to a Series, which has
                # no promotion_id column to filter on. One day is not a window
                # worth comparing anyway.
                continue

            promoted = series[series["promotion_id"] == event["promotion_id"]]
            window = series[
                (series.index >= start - pd.Timedelta(days=self.lookback_days))
                & (series.index < start)
            ]
            if promoted.empty or window.empty:
                continue
            during.append(float(promoted[settings.target].mean()))
            before.append(float(window[settings.target].mean()))

        if not during:
            raise EstimationError(
                "no event had both promotion days and a pre-period to compare "
                "against",
                method=self.name,
            )

        during_mean = float(np.mean(during))
        before_mean = float(np.mean(before))
        effect = during_mean - before_mean

        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / before_mean if before_mean > 0 else 0.0,
            baseline_units=before_mean,
            n_treated=int((panel["role"] == RowRole.TREATED).sum()),
            n_control=len(before),
            assumptions=[
                "ASSUMES the days before a promotion are what the promotion days "
                "would have been. They are not: promotions are scheduled into "
                "seasonal peaks, so the pre-period is systematically weaker.",
                "IGNORES pull-forward entirely - the post-promotion dip falls "
                "outside the comparison window, so borrowed volume is counted "
                "as incremental.",
                "This estimate is reported as a benchmark for the size of that "
                "bias. It is not a causal estimate and must not be used as one.",
            ],
            warnings=[
                "naive during-versus-before comparison; expected to overstate "
                "incrementality"
            ],
            diagnostics={
                "events_compared": float(len(during)),
                "lookback_days": float(self.lookback_days),
            },
        )


@dataclass
class BaselineCounterfactual:
    """Observed sales minus the Step 5 baseline model's no-promotion prediction.

    The baseline is a *model* of what normal demand looks like, so this inherits
    whatever that model gets wrong - including its measured +6.7% over-prediction,
    which is corrected here and reported in the diagnostics.
    """

    name: str = "baseline_counterfactual"
    #: Set False to see the uncorrected number, which is what a caller reading
    #: the Step 5 artifact directly would get.
    correct_bias: bool = True

    def estimate(
        self,
        analysis: AnalysisFrame,
        baseline_units: pd.Series | None,
        *,
        config: PromoUpliftConfig | None = None,
    ) -> EffectEstimate:
        settings = config or get_promo_uplift_config()

        if baseline_units is None:
            raise UpliftModelUnavailableError(
                "the baseline sales model is required for this estimator but no "
                "trained artifact was found; train one with "
                "scripts/train_baseline.py, or run the other estimators, which "
                "do not depend on it",
                model="baseline_sales",
            )

        panel = analysis.frame
        treated = panel["role"] == RowRole.TREATED
        if not treated.any():
            raise EstimationError("no treated rows", method=self.name)

        observed = panel.loc[treated, settings.target].to_numpy(dtype=float)
        expected = baseline_units.reindex(panel.index)[treated].to_numpy(dtype=float)

        usable = ~np.isnan(expected)
        if not usable.any():
            raise EstimationError(
                "the baseline model produced no prediction for any treated row",
                method=self.name,
            )
        observed = observed[usable]
        expected = expected[usable]

        # Undo the known over-prediction before differencing. Applying it to the
        # baseline rather than to the effect keeps the reported baseline the
        # quantity it claims to be - expected units - rather than a number that
        # only makes sense after subtraction.
        correction = 1.0 / (1.0 + BASELINE_BIAS) if self.correct_bias else 1.0
        corrected = expected * correction

        effect = float((observed - corrected).mean())
        baseline_mean = float(corrected.mean())

        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / baseline_mean if baseline_mean > 0 else 0.0,
            baseline_units=baseline_mean,
            n_treated=int(usable.sum()),
            n_control=0,
            assumptions=[
                "ASSUMES the Step 5 baseline is an unbiased estimate of demand "
                "without a promotion. It was trained only on unpromoted rows, so "
                "its prediction is a genuine no-promotion expectation - but it "
                "is still a model, and its errors flow straight into this "
                "number.",
                f"The baseline's measured over-prediction of {BASELINE_BIAS:.1%} "
                f"is {'corrected' if self.correct_bias else 'NOT corrected'} here.",
                "No control group is used, so there is nothing to check "
                "comparability against. A shock affecting the promotion window "
                "and not the training period is indistinguishable from uplift.",
            ],
            warnings=(
                []
                if self.correct_bias
                else [
                    f"the baseline over-predicts by {BASELINE_BIAS:.1%}, so this "
                    f"uplift is understated by roughly the same amount"
                ]
            ),
            diagnostics={
                "bias_correction": correction,
                "rows_without_baseline": float((~usable).sum()),
                "mean_observed": float(observed.mean()),
                "mean_baseline_raw": float(expected.mean()),
            },
        )


__all__ = ["BASELINE_BIAS", "BaselineCounterfactual", "NaiveEstimator"]
