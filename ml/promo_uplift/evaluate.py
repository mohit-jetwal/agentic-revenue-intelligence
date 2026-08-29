"""Uplift metrics and the method comparison (brief sections 17, 26).

Two jobs.

**Comparing methods.** The comparison table is the deliverable, not a step
toward one. Every estimator here targets the same ATT and differs only in what
must be true for the answer to be right, so the table is really a table of
assumptions with numbers attached. Reading it well means asking *why* two rows
disagree, not which row is largest.

The selection rule is deliberately not "pick the biggest uplift". It is:

1. Discard any method whose identifying assumption was rejected.
2. Among the rest, prefer the weakest assumption set - doubly robust over
   single-model, adjusted over unadjusted.
3. Break ties on the tighter interval.

Applied to this data that means AIPW, not because it is the largest - it is
usually smaller than the naive and IPW numbers - but because it survives the
most scrutiny.

**Ranking, via Qini and AUUC.** These answer a different question from the ATT:
not "how much did promotions do" but "if you could only promote a fraction of
these, would the model pick the right ones". That is precisely what Step 8 needs,
and a model can be excellent at one and poor at the other.

**Qini is not optimised here, and the brief is right to warn against it.** The
business objective is incremental *profit*, and Qini ranks on incremental
*units*. A promotion with high volume uplift on a thin-margin SKU ranks well on
Qini and destroys money. So the Qini curve is computed for model selection
between CATE models, and the profit ranking is what leaves the building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.diagnostics import ValidationVerdict
from ml.promo_uplift.estimators import EffectEstimate

logger = get_logger(__name__)

#: Assumption strength, weakest requirement last. Used by :func:`select_method`.
_ASSUMPTION_RANK: dict[str, int] = {
    "naive_during_vs_before": 0,
    "baseline_counterfactual": 1,
    "difference_in_differences": 2,
    "inverse_probability_weighting": 3,
    "dr_learner": 4,
    "augmented_ipw": 5,
}


@dataclass
class MethodRow:
    """One method's line in the comparison."""

    method: str
    estimate: EffectEstimate
    verdict: ValidationVerdict | None = None
    incremental_units: float | None = None
    incremental_profit: float | None = None
    roi: float | None = None

    @property
    def eligible(self) -> bool:
        """Whether this method's assumptions survived.

        The naive estimator is never eligible regardless of its diagnostics: it
        has no identifying assumption to test, so it cannot pass one.
        """
        if self.method == "naive_during_vs_before":
            return False
        return self.verdict is None or self.verdict.status != "failed"


def comparison_table(rows: list[MethodRow]) -> pd.DataFrame:
    """The side-by-side table, ordered as the report should read it."""
    records = []
    for row in rows:
        estimate = row.estimate
        interval = estimate.interval_pct()
        records.append(
            {
                "method": row.method,
                "uplift_pct": estimate.ate_pct,
                "uplift_units_per_day": estimate.ate,
                "ci_lower_pct": interval[0] if interval else None,
                "ci_upper_pct": interval[1] if interval else None,
                "has_interval": estimate.has_interval,
                "n_treated": estimate.n_treated,
                "n_control": estimate.n_control,
                "incremental_units": row.incremental_units,
                "incremental_profit": row.incremental_profit,
                "roi": row.roi,
                "validation": row.verdict.status if row.verdict else "not_assessed",
                "eligible": row.eligible,
                "assumptions": len(estimate.assumptions),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["_rank"] = frame["method"].map(_ASSUMPTION_RANK).fillna(99)
    return frame.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)


def select_method(rows: list[MethodRow]) -> tuple[MethodRow | None, str]:
    """Choose the headline estimate, and say why.

    Returns the row and the reasoning, because a selection without a stated
    rationale is indistinguishable from picking the number someone liked.
    """
    eligible = [r for r in rows if r.eligible]
    if not eligible:
        return None, (
            "no method's identifying assumptions survived validation; there is "
            "no defensible causal estimate for this request"
        )

    best = max(eligible, key=lambda r: _ASSUMPTION_RANK.get(r.method, 0))
    tied = [
        r
        for r in eligible
        if _ASSUMPTION_RANK.get(r.method, 0) == _ASSUMPTION_RANK.get(best.method, 0)
    ]
    if len(tied) > 1:
        # Tie-break on interval width, treating a missing interval as infinitely
        # wide - an estimate whose uncertainty was never established should not
        # win a tie against one that measured it.
        def width(row: MethodRow) -> float:
            interval = row.estimate.interval_pct()
            return interval[1] - interval[0] if interval else float("inf")

        best = min(tied, key=width)

    rejected = [r.method for r in rows if not r.eligible]
    reason = (
        f"{best.method} selected: it makes the weakest identifying assumptions "
        f"among the methods that passed validation"
    )
    if rejected:
        reason += f". Excluded: {', '.join(rejected)}"
    if best.verdict and best.verdict.status == "warnings":
        reason += (
            f". Note it carries {len(best.verdict.warnings)} warning(s) - the "
            f"estimate is identified but not unqualified"
        )
    logger.info("promo_uplift.method_selected", method=best.method)
    return best, reason


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------


def qini_curve(
    cate: np.ndarray, t: np.ndarray, y: np.ndarray, *, n_bins: int = 20
) -> pd.DataFrame:
    """Cumulative incremental response as the population is targeted by score.

    At each point on the curve, the top ``k`` units by predicted uplift are
    "targeted", and the Qini value is the incremental response among them:

    .. code-block:: text

        Q(k) = Y_treated(k) - Y_control(k) * N_treated(k) / N_control(k)

    The second term rescales the control response to the treated group's size,
    which is what makes the two comparable at every point on the curve. A model
    that ranks well pushes the curve above the diagonal; a useless one traces it.
    """
    treated = t.astype(bool)
    order = np.argsort(-cate)

    y_ordered = y[order]
    t_ordered = treated[order]

    cum_treated_y = np.cumsum(np.where(t_ordered, y_ordered, 0.0))
    cum_control_y = np.cumsum(np.where(~t_ordered, y_ordered, 0.0))
    cum_treated_n = np.cumsum(t_ordered)
    cum_control_n = np.cumsum(~t_ordered)

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_treated_y - np.where(
            cum_control_n > 0, cum_control_y * cum_treated_n / cum_control_n, 0.0
        )

    n = len(cate)
    cuts = np.linspace(0, n - 1, min(n_bins + 1, n)).astype(int)
    return pd.DataFrame(
        {
            "targeted_fraction": (cuts + 1) / n,
            "targeted_units": cuts + 1,
            "qini": qini[cuts],
            "random_qini": qini[-1] * (cuts + 1) / n,
        }
    )


def auuc(cate: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Area under the uplift curve, normalised against random targeting.

    Zero means the ranking is no better than random; positive means better.
    Reported alongside the ATT, never instead of it - a model can rank perfectly
    while being badly wrong about the magnitude, and a budget is set on the
    magnitude.
    """
    curve = qini_curve(cate, t, y, n_bins=len(cate) - 1 if len(cate) < 100 else 100)
    if curve.empty or len(curve) < 2:
        return 0.0
    x = curve["targeted_fraction"].to_numpy()
    model_area = float(np.trapezoid(curve["qini"].to_numpy(), x))
    random_area = float(np.trapezoid(curve["random_qini"].to_numpy(), x))
    denominator = abs(random_area)
    return (model_area - random_area) / denominator if denominator > 1e-12 else 0.0


def segment_summary(
    segments: pd.DataFrame, *, roi_break_even: float = 1.0
) -> pd.DataFrame:
    """Label segments by what should be done about them.

    Four labels rather than a ranking, because the actions differ in kind. A
    negative segment should be stopped; an uncertain one should be measured
    before anything else happens to it. Collapsing both into "low" invites the
    same decision for two different problems.
    """
    if segments.empty:
        return segments

    frame = segments.copy()
    estimable = frame.get("estimable", pd.Series(True, index=frame.index))
    uplift = frame["uplift_pct"]
    se = frame.get("standard_error")

    uncertain = ~estimable
    if se is not None:
        # Uncertain when the interval spans zero: the sign of the effect is not
        # established, so neither growing nor cutting the segment is supported.
        uncertain = uncertain | (uplift.abs() < 1.96 * se / frame["baseline"].abs())

    frame["classification"] = np.select(
        [
            uncertain,
            uplift < 0,
            uplift >= uplift.quantile(0.66),
        ],
        ["uncertain", "negative", "high_uplift"],
        default="low_uplift",
    )
    frame["action"] = frame["classification"].map(
        {
            "high_uplift": "candidate for more investment",
            "low_uplift": "works, but returns are thin - check margin before scaling",
            "negative": "stop; this promotion reduces volume",
            "uncertain": "measure before acting - the effect's sign is not established",
        }
    )
    _ = roi_break_even
    return frame


def format_comparison(frame: pd.DataFrame) -> str:
    """The comparison table as markdown, for the report and the CLI."""
    if frame.empty:
        return "_no estimates produced_"

    lines = [
        "| Method | Uplift | 95% CI | Incremental units | Incremental profit | ROI | Validation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in frame.to_dict("records"):
        interval = (
            f"[{float(row['ci_lower_pct']):+.1%}, {float(row['ci_upper_pct']):+.1%}]"
            if row["has_interval"]
            else "not estimated"
        )
        units = (
            f"{float(row['incremental_units']):,.0f}"
            if pd.notna(row["incremental_units"])
            else "-"
        )
        profit = (
            f"{float(row['incremental_profit']):,.0f}"
            if pd.notna(row["incremental_profit"])
            else "-"
        )
        roi = f"{float(row['roi']):.2f}" if pd.notna(row["roi"]) else "-"
        marker = "" if row["eligible"] else " *(not eligible)*"
        lines.append(
            f"| {row['method']}{marker} | {float(row['uplift_pct']):+.1%} | {interval} | "
            f"{units} | {profit} | {roi} | {row['validation']} |"
        )
    return "\n".join(lines)


__all__ = [
    "MethodRow",
    "auuc",
    "comparison_table",
    "format_comparison",
    "qini_curve",
    "segment_summary",
    "select_method",
]
