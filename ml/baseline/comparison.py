"""Model and approach comparison (brief sections 21, 41).

Produces the table that justifies the selection, and then makes the selection
from the numbers rather than from an assumption about which model ought to win.

Section 41 is the point: *the best model is not necessarily the most complex*.
So the selection weighs accuracy against stability, cost and interpretability,
and a gradient-boosted model that beats the seasonal naive by two percentage
points at fifty times the training cost should not automatically take it.

The comparison spans two axes at once - three estimators by two promotion
approaches - because the approach question (section 5) cannot be settled on
argument. Both have a real bias; which one dominates is an empirical fact about
this data, and here it can be measured against ``latent_units``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.evaluation import BaselineMetrics, evaluate_against_latent
from ml.baseline.training import BacktestResult, PromotionApproach, TrainedBaseline

logger = get_logger(__name__)

#: Minimum ratio of predicted baseline to *observed* sales on stockout rows.
#:
#: Observed sales are censored by inventory on those rows, so a model that
#: learned demand must sit clearly above them. 1.20 is deliberately a floor
#: rather than a target: it is loose enough that a merely-mediocre model still
#: passes, and tight enough that a model which learned the censoring - which
#: lands near 1.0 by construction - cannot.
STOCKOUT_LIFT_FLOOR = 1.20


@dataclass
class Candidate:
    """One trained model under one approach, with everything it is judged on."""

    trained: TrainedBaseline
    backtest: BacktestResult | None = None
    #: Metrics against Step 2's hidden ground truth, where available.
    latent_metrics: dict[str, BaselineMetrics] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.trained.name

    @property
    def test_wmape(self) -> float:
        metric = self.trained.metrics.get("test_clean")
        return metric.wmape if metric else float("nan")

    @property
    def test_bias(self) -> float:
        metric = self.trained.metrics.get("test_clean")
        return metric.bias_pct if metric else float("nan")

    @property
    def latent_wmape(self) -> float:
        """WMAPE against true demand on clean rows.

        The more meaningful number where ground truth exists: accuracy against
        observed sales can be flattered by learning the censoring, whereas
        accuracy against latent demand cannot.
        """
        metric = self.latent_metrics.get("clean_vs_latent")
        return metric.wmape if metric else float("nan")

    @property
    def stockout_lift(self) -> float:
        """Predicted baseline over *observed* sales on stockout rows.

        The headline diagnostic, and the criterion needs stating carefully
        because the obvious version is wrong.

        Observed sales during a stockout are censored by inventory, so a model
        that learned demand must predict materially **above** them. A ratio near
        1.0 means the model learned the supply failure itself and would report a
        stockout as a demand collapse - exactly backwards, and exactly the error
        the Root Cause agent in Step 17 must not inherit.

        The tempting alternative - comparing the prediction to *latent* demand -
        is misleading here, because stockouts in this data are **endogenous**:
        they occur when demand spikes, and measured against the generated
        dataset latent demand during a stockout runs about 1.57x normal. A
        baseline correctly predicting normal demand therefore lands near 0.6 of
        stockout-period latent demand, and judging it against 1.0 would
        disqualify a perfectly good model. Ratio-to-observed has no such
        confound.
        """
        metric = self.latent_metrics.get("stockout_vs_observed")
        if metric is None or metric.actual_total <= 0:
            return float("nan")
        return metric.predicted_total / metric.actual_total

    @property
    def stockout_vs_latent_ratio(self) -> float:
        """Predicted baseline over true demand on stockout rows.

        Reported for transparency rather than used for selection - see
        :attr:`stockout_lift` for why it is the wrong criterion.
        """
        metric = self.latent_metrics.get("stockout_vs_latent")
        if metric is None or metric.actual_total <= 0:
            return float("nan")
        return metric.predicted_total / metric.actual_total

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.trained.estimator.name,
            "approach": self.trained.approach.value,
            "test_wmape": self.test_wmape,
            "latent_wmape": self.latent_wmape,
            "bias_pct": self.test_bias,
            "mae": self.trained.metrics["test_clean"].mae
            if "test_clean" in self.trained.metrics else float("nan"),
            "rmse": self.trained.metrics["test_clean"].rmse
            if "test_clean" in self.trained.metrics else float("nan"),
            "stockout_lift": self.stockout_lift,
            "vs_latent_so": self.stockout_vs_latent_ratio,
            "backtest_wmape": self.backtest.mean_wmape if self.backtest else float("nan"),
            "backtest_std": self.backtest.std_wmape if self.backtest else float("nan"),
            "stable": self.backtest.is_stable if self.backtest else None,
            "coverage": self.trained.coverage.empirical if self.trained.coverage else float("nan"),
            "train_seconds": round(self.trained.train_seconds, 2),
            "predict_seconds": round(self.trained.predict_seconds, 3),
        }


@dataclass
class ComparisonResult:
    """The full comparison and the selection made from it."""

    candidates: list[Candidate]
    selected: Candidate
    rationale: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([c.to_row() for c in self.candidates])
        # Ranked by accuracy against ground truth where available, since that is
        # what the model is actually for.
        sort_column = "latent_wmape" if frame["latent_wmape"].notna().any() else "test_wmape"
        return frame.sort_values(sort_column).reset_index(drop=True)

    def summary(self) -> str:
        frame = self.to_frame()
        lines = ["Model comparison", ""]

        display = frame.copy()
        for column in ("test_wmape", "latent_wmape", "bias_pct", "backtest_wmape",
                       "backtest_std", "coverage"):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "-"
                )
        for column in ("stockout_lift", "vs_latent_so"):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "-"
                )
        lines.append(display.to_string(index=False))

        lines += ["", f"Selected: {self.selected.name}", ""]
        lines.extend(f"  - {reason}" for reason in self.rationale)
        return "\n".join(lines)


def select_model(
    candidates: list[Candidate],
    *,
    complexity_tolerance: float = 0.02,
) -> ComparisonResult:
    """Choose a model, weighing accuracy against stability and cost.

    Selection rules, applied in order:

    1. **Correctness first.** A candidate that failed the stockout check is
       disqualified regardless of headline accuracy - it has learned censored
       sales as demand, which makes every downstream uplift number wrong in the
       same direction. An accurate model measuring the wrong quantity is worse
       than a less accurate one measuring the right quantity.
    2. **Accuracy against ground truth** where available, observed sales
       otherwise.
    3. **Simplicity when the gap is small.** If the seasonal naive is within
       ``complexity_tolerance`` of the best model, it wins: section 41, and a
       benchmark that holds its own is telling you the signal is simple.
    4. **Stability breaks ties.** A model whose accuracy swings between quarters
       cannot be trusted behind a recommendation.
    """
    if not candidates:
        raise ValueError("no candidates to select from")

    rationale: list[str] = []

    # 1. Disqualify on correctness.
    def learned_demand(candidate: Candidate) -> bool:
        lift = candidate.stockout_lift
        # No ground truth available - cannot disqualify on evidence we lack.
        if pd.isna(lift):
            return True
        return lift >= STOCKOUT_LIFT_FLOOR

    eligible = [c for c in candidates if learned_demand(c)]
    disqualified = [c for c in candidates if c not in eligible]
    for candidate in disqualified:
        rationale.append(
            f"{candidate.name} disqualified: during stockouts it predicts only "
            f"{candidate.stockout_lift:.2f}x the censored observed sales, so it "
            f"tracked the supply failure rather than seeing through it to demand."
        )
    if not eligible:
        rationale.append(
            "No candidate passed the stockout check; falling back to the full set "
            "and reporting the failure rather than hiding it."
        )
        eligible = candidates

    # 2. Rank on the most meaningful accuracy available.
    def accuracy(candidate: Candidate) -> float:
        value = candidate.latent_wmape
        return value if pd.notna(value) else candidate.test_wmape

    ranked = sorted(eligible, key=accuracy)
    best = ranked[0]
    rationale.append(
        f"{best.name} is most accurate at WMAPE {accuracy(best):.1%} against "
        f"{'true demand' if pd.notna(best.latent_wmape) else 'observed sales'}."
    )

    # 3. Prefer simplicity when the difference is marginal.
    selected = best
    naive = next((c for c in ranked if c.trained.estimator.name == "seasonal_naive"), None)
    if naive is not None and naive is not best:
        gap = accuracy(naive) - accuracy(best)
        if gap <= complexity_tolerance:
            selected = naive
            rationale.append(
                f"Selected the seasonal naive instead: it is within {gap:.1%} of "
                f"{best.name}, which does not justify the added complexity, "
                f"training cost and opacity."
            )
        else:
            rationale.append(
                f"The seasonal naive benchmark trails by {gap:.1%}, so the added "
                f"complexity is earning its place."
            )

    # 4. Stability check on whatever was chosen.
    if selected.backtest is not None:
        if selected.backtest.is_stable:
            rationale.append(f"Backtest is stable: {selected.backtest.summary()}")
        else:
            rationale.append(
                f"WARNING - backtest is unstable: {selected.backtest.summary()}. "
                f"Accuracy varies enough between quarters that a single headline "
                f"number is misleading."
            )

    # Report the approach comparison, which is a finding in its own right.
    by_approach: dict[str, float] = {}
    for candidate in eligible:
        if candidate.trained.estimator.name == selected.trained.estimator.name:
            by_approach[candidate.trained.approach.value] = accuracy(candidate)
    if len(by_approach) == 2:
        exclude = by_approach.get(PromotionApproach.EXCLUDE.value, float("nan"))
        control = by_approach.get(PromotionApproach.CONTROL.value, float("nan"))
        better = "exclude" if exclude < control else "control"
        rationale.append(
            f"Promotion handling: exclude {exclude:.1%} vs control {control:.1%} "
            f"WMAPE - '{better}' wins on this data by {abs(exclude - control):.1%}."
        )

    if selected.trained.coverage is not None:
        rationale.append(f"Prediction interval: {selected.trained.coverage.summary()}")

    logger.info(
        "baseline.model_selected",
        model=selected.trained.estimator.name,
        approach=selected.trained.approach.value,
        wmape=round(accuracy(selected), 4),
    )
    return ComparisonResult(candidates=candidates, selected=selected, rationale=rationale)


def score_against_latent(
    candidate: Candidate,
    panel: pd.DataFrame,
    latent: pd.DataFrame,
    *,
    date_column: str = "date",
) -> dict[str, BaselineMetrics]:
    """Score a candidate on the test window against Step 2's true demand.

    The validation almost no real project can run. ``latent_units`` is demand
    before inventory censored it, so this separates a model that learned demand
    from one that learned what the till happened to record.
    """
    split = candidate.trained.split
    dates = pd.to_datetime(panel[date_column]).dt.date
    test_rows = panel[(dates >= split.test_start) & (dates <= split.test_end)]
    if test_rows.empty or latent.empty:
        return {}

    predictions = test_rows[["date", "product_id", "store_id"]].copy()
    predictions["baseline_units"] = candidate.trained.predict_baseline(test_rows)
    predictions["actual_units"] = test_rows["units"].to_numpy()
    for flag in ("promotion_flag", "stockout_flag"):
        if flag in test_rows.columns:
            predictions[flag] = test_rows[flag].to_numpy()

    return evaluate_against_latent(predictions, latent)
