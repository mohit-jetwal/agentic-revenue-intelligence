"""Baseline evaluation metrics (brief sections 13-16).

Everything downstream depends on measuring correctly, so the definitions are
written out rather than pulled from a library where the zero-handling would be
someone else's decision.

**WMAPE is the headline.** Volume-weighted, so a large error on a hero SKU is not
hidden by small errors across a long tail of slow movers. Step 2's generator
draws base demand log-normally precisely to create that skew - a plain MAPE
would report the tail and call it the business.

**MAPE is computed only where the actual is non-zero**, and the count of
excluded rows is reported alongside. Section 13 is explicit about this. A silent
``inf`` or a quietly dropped denominator is worse than an absent number, because
it looks like a measurement.

**Bias is reported and is not an afterthought.** A baseline that is 8% low on
average produces an 8% phantom uplift on every promotion measured against it.
For this model, systematic bias matters more than dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Denominators below this are treated as zero. Guards the ratio metrics without
#: silently reclassifying a genuine small value as missing.
_EPSILON = 1e-9


@dataclass(frozen=True)
class BaselineMetrics:
    """Accuracy of a baseline against a set of actuals."""

    n: int
    mae: float
    rmse: float
    wmape: float
    bias: float
    bias_pct: float
    #: Mean absolute percentage error over non-zero actuals only.
    mape: float | None
    #: Rows excluded from MAPE because the actual was zero.
    mape_excluded: int
    #: Sum of actuals, so segment tables can be volume-weighted.
    actual_total: float
    predicted_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "wmape": round(self.wmape, 6),
            "bias": round(self.bias, 4),
            "bias_pct": round(self.bias_pct, 6),
            "mape": round(self.mape, 6) if self.mape is not None else None,
            "mape_excluded": self.mape_excluded,
            "actual_total": round(self.actual_total, 2),
            "predicted_total": round(self.predicted_total, 2),
        }

    def summary(self) -> str:
        mape = f"{self.mape:.1%}" if self.mape is not None else "n/a"
        return (
            f"n={self.n:,}  WMAPE={self.wmape:.1%}  MAE={self.mae:.2f}  "
            f"RMSE={self.rmse:.2f}  bias={self.bias_pct:+.1%}  MAPE={mape}"
        )


def compute_metrics(
    actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray
) -> BaselineMetrics:
    """Compute the full metric set for one aligned actual/predicted pair."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)

    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: actual {y.shape} vs predicted {yhat.shape}")

    # Rows where either side is missing carry no information and would poison
    # every aggregate. Dropped, and the drop is visible in `n`.
    usable = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[usable], yhat[usable]

    if y.size == 0:
        return BaselineMetrics(
            n=0, mae=float("nan"), rmse=float("nan"), wmape=float("nan"),
            bias=float("nan"), bias_pct=float("nan"), mape=None, mape_excluded=0,
            actual_total=0.0, predicted_total=0.0,
        )

    error = yhat - y
    absolute = np.abs(error)

    actual_total = float(y.sum())
    # WMAPE: total absolute error over total actual volume. Equivalently, MAE
    # weighted by volume share - which is why it survives a long tail of small
    # series that would dominate an unweighted MAPE.
    wmape = float(absolute.sum() / actual_total) if abs(actual_total) > _EPSILON else float("nan")

    bias = float(error.mean())
    bias_pct = float(error.sum() / actual_total) if abs(actual_total) > _EPSILON else float("nan")

    # Section 13: MAPE only where the actual is non-zero. Zero-sales days are
    # common for slow movers, so this exclusion is substantial, not incidental -
    # hence reporting the count rather than burying it.
    non_zero = np.abs(y) > _EPSILON
    excluded = int((~non_zero).sum())
    mape = float(np.mean(absolute[non_zero] / np.abs(y[non_zero]))) if non_zero.any() else None

    return BaselineMetrics(
        n=int(y.size),
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        wmape=wmape,
        bias=bias,
        bias_pct=bias_pct,
        mape=mape,
        mape_excluded=excluded,
        actual_total=actual_total,
        predicted_total=float(yhat.sum()),
    )


def evaluate_by_segment(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_units",
    predicted_column: str = "baseline_units",
    segment: str,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Metrics broken down by one segment (brief section 14).

    Aggregate accuracy hides where a model is weak. A 12% WMAPE overall can be
    6% on hypermarkets and 40% on convenience stores, and only the second number
    tells you whether to trust a convenience-store recommendation.

    Segments with fewer than ``min_rows`` observations are dropped: WMAPE on
    eleven rows is noise wearing a percentage sign.
    """
    if segment not in frame.columns:
        raise KeyError(f"segment column {segment!r} not present in the frame")

    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(segment, observed=True):
        if len(group) < min_rows:
            continue
        metrics = compute_metrics(group[actual_column], group[predicted_column])
        rows.append({segment: value, **metrics.to_dict()})

    if not rows:
        return pd.DataFrame()

    # Sorted by volume: the segments carrying the business appear first, which
    # is the order someone reading the table actually cares about.
    return pd.DataFrame(rows).sort_values("actual_total", ascending=False).reset_index(drop=True)


def evaluate_all_segments(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_units",
    predicted_column: str = "baseline_units",
    segments: tuple[str, ...] = (
        "category",
        "brand",
        "region",
        "channel",
        "store_type",
        "season",
        "promotion_flag",
        "stockout_flag",
    ),
    min_rows: int = 30,
) -> dict[str, pd.DataFrame]:
    """Segment tables for every dimension present in the frame."""
    return {
        segment: evaluate_by_segment(
            frame,
            actual_column=actual_column,
            predicted_column=predicted_column,
            segment=segment,
            min_rows=min_rows,
        )
        for segment in segments
        if segment in frame.columns
    }


@dataclass
class ErrorAnalysis:
    """Where the model is weakest (brief section 23)."""

    worst_products: pd.DataFrame
    worst_stores: pd.DataFrame
    worst_segments: dict[str, pd.DataFrame]
    largest_errors: pd.DataFrame
    #: Diagnostic notes generated from the numbers, not hand-written.
    findings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Error analysis", ""]
        lines.extend(f"  - {finding}" for finding in self.findings)
        return "\n".join(lines)


def analyse_errors(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_units",
    predicted_column: str = "baseline_units",
    top_n: int = 15,
) -> ErrorAnalysis:
    """Identify where the baseline goes wrong, and say what the pattern is.

    Findings are derived from the numbers rather than written by hand, so the
    report stays true when the model changes. A hand-written "the model
    struggles with new products" survives a fix and becomes a lie.
    """
    working = frame.copy()
    working["_error"] = working[predicted_column] - working[actual_column]
    working["_abs_error"] = working["_error"].abs()

    worst_products = (
        evaluate_by_segment(
            working, actual_column=actual_column, predicted_column=predicted_column,
            segment="product_id", min_rows=30,
        )
        .sort_values("wmape", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    worst_stores = (
        evaluate_by_segment(
            working, actual_column=actual_column, predicted_column=predicted_column,
            segment="store_id", min_rows=30,
        )
        .sort_values("wmape", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    worst_segments = evaluate_all_segments(
        working, actual_column=actual_column, predicted_column=predicted_column
    )

    display_columns = [
        c
        for c in ("date", "product_id", "store_id", actual_column, predicted_column,
                  "_error", "promotion_flag", "stockout_flag")
        if c in working.columns
    ]
    largest_errors = (
        working.nlargest(top_n, "_abs_error")[display_columns].reset_index(drop=True)
    )

    findings: list[str] = []
    overall = compute_metrics(working[actual_column], working[predicted_column])
    findings.append(f"Overall: {overall.summary()}")

    if abs(overall.bias_pct) > 0.02:
        direction = "over" if overall.bias_pct > 0 else "under"
        findings.append(
            f"Systematic {direction}-prediction of {abs(overall.bias_pct):.1%}. For a "
            f"baseline this matters more than dispersion: every promotion measured "
            f"against it inherits the same {direction}statement of uplift."
        )

    if not worst_products.empty:
        spread = worst_products["wmape"].iloc[0] / max(overall.wmape, _EPSILON)
        findings.append(
            f"Worst product WMAPE is {worst_products['wmape'].iloc[0]:.1%}, "
            f"{spread:.1f}x the overall rate - accuracy is not uniform across the "
            f"catalogue."
        )

    if "is_new_product" in working.columns:
        new = working[working["is_new_product"].astype(bool)]
        established = working[~working["is_new_product"].astype(bool)]
        if len(new) > 30 and len(established) > 30:
            new_metrics = compute_metrics(new[actual_column], new[predicted_column])
            old_metrics = compute_metrics(
                established[actual_column], established[predicted_column]
            )
            verdict = (
                "materially worse"
                if new_metrics.wmape > old_metrics.wmape * 1.3
                else "comparable"
            )
            findings.append(
                f"New products (<90 days): WMAPE {new_metrics.wmape:.1%} vs "
                f"{old_metrics.wmape:.1%} for established - cold start is {verdict}."
            )

    if "promotion_flag" in working.columns:
        promo = working[working["promotion_flag"].astype(bool)]
        if len(promo) > 30:
            promo_metrics = compute_metrics(promo[actual_column], promo[predicted_column])
            findings.append(
                f"On promotional rows the baseline under-predicts by "
                f"{-promo_metrics.bias_pct:.1%}. That is intended - the gap is the "
                f"uplift - so this is not an accuracy failure."
            )

    return ErrorAnalysis(
        worst_products=worst_products,
        worst_stores=worst_stores,
        worst_segments=worst_segments,
        largest_errors=largest_errors,
        findings=findings,
    )


def evaluate_against_latent(
    predictions: pd.DataFrame,
    latent: pd.DataFrame,
    *,
    predicted_column: str = "baseline_units",
) -> dict[str, BaselineMetrics]:
    """Score the baseline against **true** demand (Step 2 ground truth).

    The validation a real project cannot do. ``latent_units`` is the demand that
    existed before inventory censored it, so this answers the question that
    actually matters - did the model learn demand, or did it learn what the till
    happened to record?

    Three row types, and they mean different things:

    * ``clean``    - no promotion, in stock. Here latent equals the true
      baseline, so this is direct accuracy.
    * ``stockout`` - latent far exceeds observed. A baseline tracking latent
      learned demand; one tracking observed learned the supply failure.
    * ``promotional`` - latent includes the promotional lift, so the baseline
      *should* fall below it. Reported for direction, not accuracy.
    """
    keys = ["date", "product_id", "store_id"]
    merged = predictions.merge(
        latent[[*keys, "latent_units", "observed_units", "lost_units"]],
        on=keys,
        how="inner",
    )
    if merged.empty:
        return {}

    promotional = merged["promotion_flag"].astype(bool) if "promotion_flag" in merged else False
    stockout = merged["stockout_flag"].astype(bool) if "stockout_flag" in merged else False

    results: dict[str, BaselineMetrics] = {}

    clean = merged[~promotional & ~stockout]
    if not clean.empty:
        results["clean_vs_latent"] = compute_metrics(
            clean["latent_units"], clean[predicted_column]
        )

    censored = merged[stockout]
    if not censored.empty:
        # The headline test. Against latent, a good baseline is accurate; against
        # observed it should look "wrong" by exactly the lost volume - and that
        # apparent error is the model being right.
        results["stockout_vs_latent"] = compute_metrics(
            censored["latent_units"], censored[predicted_column]
        )
        results["stockout_vs_observed"] = compute_metrics(
            censored["observed_units"], censored[predicted_column]
        )

    promoted = merged[promotional]
    if not promoted.empty:
        results["promotional_vs_latent"] = compute_metrics(
            promoted["latent_units"], promoted[predicted_column]
        )

    return results


def irreducible_error(latent: pd.DataFrame) -> BaselineMetrics | None:
    """The noise floor: how well a *perfect* model could possibly score.

    Step 2 stores ``mean_demand`` - the true conditional mean ``exp(log lambda)``
    - alongside the realised ``latent_units`` drawn from it. The error between
    those two is pure sampling noise from the negative-binomial draw, and no
    model can do better because there is nothing left to learn.

    This matters for reading every other number here. Demand is over-dispersed:
    with a mean near 60 and a dispersion parameter of 3-9, the coefficient of
    variation is large, and the noise floor lands around 35% WMAPE. A model
    scoring 40% is therefore at roughly 1.15x the theoretical best, not
    "inaccurate" - and a model scoring 20% would be impossible and would mean
    something had leaked.

    Without this benchmark, a headline WMAPE is uninterpretable in both
    directions: it invites despair at a good model and suspicion of nothing at a
    broken one.
    """
    if latent.empty or "mean_demand" not in latent.columns:
        return None
    usable = latent[latent["mean_demand"] > 0]
    if usable.empty:
        return None
    return compute_metrics(usable["latent_units"], usable["mean_demand"])


def format_comparison(metrics: dict[str, BaselineMetrics]) -> str:
    """Render a named metric set as an aligned table."""
    if not metrics:
        return "(no metrics)"
    width = max(len(name) for name in metrics)
    lines = [f"{'segment'.ljust(width)}  {'n':>9}  {'WMAPE':>8}  {'MAE':>9}  {'bias':>8}"]
    lines.append("-" * len(lines[0]))
    for name, metric in metrics.items():
        lines.append(
            f"{name.ljust(width)}  {metric.n:>9,}  {metric.wmape:>7.1%}  "
            f"{metric.mae:>9.2f}  {metric.bias_pct:>+7.1%}"
        )
    return "\n".join(lines)
