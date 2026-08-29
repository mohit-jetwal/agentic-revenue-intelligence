"""Causal validation: placebo, sensitivity, balance verdicts (sections 22-24).

Ordinary cross-validation cannot validate a causal estimate. Hold out any share
of the data and the counterfactual is still missing from the held-out part, so a
model can predict the outcome perfectly and be wrong about the effect by any
margin at all. Predictive accuracy and causal correctness are different
properties, and only one of them is testable by splitting rows.

What *is* testable:

**Placebo.** Move the treatment window to a period where no promotion ran. The
true effect there is zero by construction. Anything the method finds is
attributable to the method, not the promotion. This is the closest thing to a
unit test that causal inference has, and it is the one diagnostic that can
invalidate a run outright.

**Sensitivity.** Re-estimate while varying the choices nobody can prove correct -
the washout length, the control window, the trimming level. An estimate that
swings from +12% to +45% across defensible specifications is not a finding, no
matter how tight the interval on any one of them. This is where an honest
analysis usually loses its headline number.

**Balance, method-aware.** Poor covariate balance is fatal for IPW and matching,
which have nothing but the propensity model. It is a *warning* for AIPW and the
DR-learner, which stay consistent if the outcome model is right. That is not a
convenient distinction - it is the doubly robust property, and it was observed
directly here: on the confounded synthetic panel the worst standardised
difference sat at 0.38 after weighting, well past the 0.10 threshold, and AIPW
still recovered +65.2% against a true +63.3%.

**Ground truth.** The platform generator records the true promotion response
curve per product and mechanic, so on that dataset the estimate can be compared
against the parameters that produced the data. Two caveats, stated in the result
rather than buried: the store's promo responsiveness and a per-event regional
draw are not persisted, so only the **average** is recoverable, never a
per-event effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.matching import BalanceReport
from ml.promo_uplift.treatment import DATE, KEYS, AnalysisFrame, RowRole

logger = get_logger(__name__)

#: Methods whose only identifying assumption is the propensity model. Balance
#: failure disqualifies these; it merely warns the doubly robust ones.
_PROPENSITY_ONLY: frozenset[str] = frozenset(
    {"inverse_probability_weighting", "matched_control"}
)


@dataclass
class PlaceboResult:
    """A treatment effect estimated where no treatment occurred."""

    effect_pct: float
    threshold: float
    n_treated: int
    shift_days: int
    #: The real estimate, for context. A placebo of +2% next to a real +60% is
    #: reassuring; the same +2% next to a real +3% is not, and a bare pass/fail
    #: would hide the difference.
    reference_pct: float | None = None

    @property
    def passed(self) -> bool:
        return abs(self.effect_pct) <= self.threshold

    @property
    def ratio_to_reference(self) -> float | None:
        if self.reference_pct is None or abs(self.reference_pct) < 1e-9:
            return None
        return abs(self.effect_pct) / abs(self.reference_pct)

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        ratio = self.ratio_to_reference
        context = f", {ratio:.0%} of the real estimate" if ratio is not None else ""
        return (
            f"placebo {verdict}: {self.effect_pct:+.2%} where the true effect is "
            f"zero (threshold {self.threshold:.1%}{context})"
        )


@dataclass
class SensitivityRow:
    """One specification and what it produced."""

    parameter: str
    value: Any
    effect_pct: float
    n_treated: int
    failed: str | None = None


@dataclass
class SensitivityResult:
    """How the estimate moves across defensible specifications."""

    rows: list[SensitivityRow]
    reference_pct: float

    @property
    def usable(self) -> list[SensitivityRow]:
        return [r for r in self.rows if r.failed is None]

    def spread(self) -> float:
        values = [r.effect_pct for r in self.usable]
        return max(values) - min(values) if values else 0.0

    def relative_spread(self) -> float:
        """Spread as a multiple of the headline estimate.

        The number that matters. A 5-point spread around a 60% effect is
        robustness; the same spread around a 4% effect means the specification
        is doing the work, not the data.
        """
        return self.spread() / abs(self.reference_pct) if abs(self.reference_pct) > 1e-9 else (
            float("inf") if self.spread() > 0 else 0.0
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "parameter": r.parameter,
                    "value": r.value,
                    "uplift_pct": r.effect_pct,
                    "n_treated": r.n_treated,
                    "failed": r.failed,
                }
                for r in self.rows
            ]
        )

    def summary(self) -> str:
        return (
            f"sensitivity: {len(self.usable)}/{len(self.rows)} specifications "
            f"estimable, spread {self.spread():.1%} "
            f"({self.relative_spread():.0%} of the headline estimate)"
        )


@dataclass
class ValidationVerdict:
    """The overall judgement, and what drove it."""

    status: str
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def summary(self) -> str:
        return (
            f"validation {self.status}: {len(self.blocking)} blocking, "
            f"{len(self.warnings)} warnings"
        )


def placebo_frame(
    analysis: AnalysisFrame, *, config: PromoUpliftConfig | None = None
) -> AnalysisFrame:
    """Relabel each event onto an earlier, untreated window.

    Every event is shifted back by ``placebo_shift_days`` and the real treated
    and washout rows are dropped entirely. Dropping them is essential: leaving
    them in the control pool would put genuinely promoted days on the other side
    of the comparison, and the placebo would find a large *negative* effect for
    an entirely mechanical reason.

    The shift is validated at config load to clear the washout window, so the
    placebo period cannot overlap the pull-forward dip either.
    """
    settings = config or get_promo_uplift_config()
    shift = pd.Timedelta(days=settings.validation.placebo_shift_days)

    panel = analysis.frame.copy()
    events = analysis.events.copy()

    # Anything genuinely touched by a promotion leaves the frame.
    contaminated = panel["role"].isin([RowRole.TREATED, RowRole.WASHOUT, RowRole.EXCLUDED])
    clean = panel[~contaminated].copy()

    events["start_date"] = pd.to_datetime(events["start_date"]) - shift
    events["end_date"] = pd.to_datetime(events["end_date"]) - shift

    # Rebuild roles against the shifted windows. Expanding the events to one row
    # per day and merging is both simpler and faster than an index-assignment
    # loop: it is the same technique `features/engineering/promotion.py` uses to
    # attach the real promotion calendar, so the placebo frame is built the way
    # the genuine one is.
    clean["role"] = RowRole.CONTROL
    clean["promotion_id"] = None

    expanded = events[[*KEYS, "promotion_id", "start_date", "end_date"]].copy()
    expanded["_days"] = pd.Series(
        [
            pd.date_range(start, end, freq="D")
            for start, end in zip(expanded["start_date"], expanded["end_date"], strict=True)
        ],
        index=expanded.index,
        dtype=object,
    )
    calendar = expanded.explode("_days", ignore_index=True)
    calendar[DATE] = pd.to_datetime(calendar["_days"])
    calendar = calendar[[*KEYS, DATE, "promotion_id"]].drop_duplicates(
        subset=[*KEYS, DATE], keep="first"
    )

    rebuilt = clean.drop(columns=["promotion_id"]).merge(
        calendar, on=[*KEYS, DATE], how="left"
    )
    rebuilt["role"] = np.where(
        rebuilt["promotion_id"].notna(), RowRole.TREATED, RowRole.CONTROL
    )
    rebuilt["role"] = pd.Categorical(rebuilt["role"], categories=list(RowRole))
    rebuilt["treatment"] = rebuilt["role"] == RowRole.TREATED

    return AnalysisFrame(
        frame=rebuilt.sort_values([*KEYS, DATE]).reset_index(drop=True),
        events=events,
        excluded={"placebo_dropped_real_treatment": int(contaminated.sum())},
        warnings=["placebo frame: treatment labels moved to an untreated window"],
    )


def evaluate_placebo(
    effect: EffectEstimate,
    *,
    reference: EffectEstimate | None = None,
    config: PromoUpliftConfig | None = None,
) -> PlaceboResult:
    """Judge a placebo estimate against the configured threshold."""
    settings = config or get_promo_uplift_config()
    result = PlaceboResult(
        effect_pct=effect.ate_pct,
        threshold=settings.validation.placebo_max_abs_effect,
        n_treated=effect.n_treated,
        shift_days=settings.validation.placebo_shift_days,
        reference_pct=reference.ate_pct if reference else None,
    )
    logger.info(
        "promo_uplift.placebo_evaluated",
        effect_pct=round(result.effect_pct, 5),
        passed=result.passed,
    )
    return result


def judge(
    *,
    estimate: EffectEstimate,
    balance: BalanceReport | None,
    overlap_warnings: list[str],
    placebo: PlaceboResult | None,
    sensitivity: SensitivityResult | None,
    config: PromoUpliftConfig | None = None,
) -> ValidationVerdict:
    """Combine the diagnostics into one status.

    Three levels. ``passed`` means every check the method depends on held.
    ``warnings`` means something is off but the estimate is still identified -
    typically balance failing for a doubly robust method. ``failed`` means an
    assumption the method cannot do without was violated, and the estimate must
    not be presented as causal.

    The estimate is returned in all three cases. Withholding it does not protect
    anyone: the caller can compute a naive number in one line of SQL, so the
    choice is not between our number and no number, it is between our number
    labelled and theirs unlabelled.
    """
    settings = config or get_promo_uplift_config()
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    blocking: list[str] = []

    propensity_only = estimate.method in _PROPENSITY_ONLY

    if balance is not None:
        checks["covariate_balance"] = balance.satisfied
        if not balance.satisfied:
            worst = balance.worst
            detail = (
                f"{len(balance.unbalanced)} covariates exceed "
                f"{balance.threshold:.2f} standardised difference after "
                f"weighting (worst: {worst.covariate} at {worst.smd_after:+.3f})"
                if worst
                else "covariates remain unbalanced after weighting"
            )
            if propensity_only:
                blocking.append(
                    f"{detail}. {estimate.method} relies entirely on the "
                    f"propensity model, so unbalanced covariates mean the "
                    f"comparison is not adjusted and the estimate is not causal"
                )
            else:
                warnings.append(
                    f"{detail}. {estimate.method} remains consistent if the "
                    f"outcome model is correctly specified, so this is a "
                    f"caution rather than a disqualification - but the doubly "
                    f"robust property is now carrying the whole argument"
                )

    if overlap_warnings:
        checks["overlap"] = False
        blocking.extend(overlap_warnings)
    else:
        checks["overlap"] = True

    if placebo is not None:
        checks["placebo"] = placebo.passed
        if not placebo.passed:
            blocking.append(
                f"placebo test failed: {placebo.effect_pct:+.2%} estimated in a "
                f"window with no promotion, above the "
                f"{placebo.threshold:.1%} threshold. The method is detecting "
                f"something other than the treatment"
            )

    if sensitivity is not None:
        stable = sensitivity.relative_spread() <= 0.5
        checks["sensitivity"] = stable
        if not stable:
            warnings.append(
                f"the estimate spans {sensitivity.spread():.1%} across "
                f"defensible specifications, "
                f"{sensitivity.relative_spread():.0%} of the headline value; "
                f"the specification is doing much of the work"
            )

    if estimate.warnings:
        warnings.extend(estimate.warnings)

    status = "failed" if blocking else ("warnings" if warnings else "passed")
    verdict = ValidationVerdict(
        status=status, checks=checks, warnings=warnings, blocking=blocking
    )
    logger.info(
        "promo_uplift.validation_judged",
        status=status,
        blocking=len(blocking),
        warnings=len(warnings),
        method=estimate.method,
    )
    _ = settings
    return verdict


# --------------------------------------------------------------------------
# Ground-truth validation, platform dataset only
# --------------------------------------------------------------------------


@dataclass
class GroundTruthComparison:
    """Estimated ATT against the parameters that generated the data."""

    estimated_pct: float
    expected_pct: float
    n_events: int
    #: Components of the expected effect, so the two channels are visible.
    mechanic_pct: float
    price_channel_pct: float
    caveats: list[str] = field(default_factory=list)

    @property
    def absolute_error(self) -> float:
        return abs(self.estimated_pct - self.expected_pct)

    @property
    def relative_error(self) -> float:
        return (
            self.absolute_error / abs(self.expected_pct)
            if abs(self.expected_pct) > 1e-9
            else float("inf")
        )

    def summary(self) -> str:
        return (
            f"ground truth: estimated {self.estimated_pct:+.1%} against an "
            f"expected {self.expected_pct:+.1%} "
            f"(mechanic {self.mechanic_pct:+.1%} + price "
            f"{self.price_channel_pct:+.1%}); error "
            f"{self.absolute_error:.1%} absolute, {self.relative_error:.0%} relative"
        )


def expected_effect_from_ground_truth(
    events: pd.DataFrame,
    ground_truth_dir: Path,
) -> GroundTruthComparison | None:
    """Rebuild the true average effect from the generator's own parameters.

    The platform generator applies a promotion through two additive terms in log
    demand:

    .. code-block:: text

        mechanic     = a * (1 - exp(-b * depth)) * responsiveness * regional
        price cut    = beta_own * log(1 - depth)

    ``a``, ``b`` and ``beta_own`` are persisted in ``ground_truth/``.
    ``responsiveness`` (a hidden store attribute) and ``regional`` (a per-event
    ``N(1, 0.18)`` draw) are **not**. Both have expectation near 1, so averaging
    over many events recovers the mean effect while no individual event's effect
    is knowable.

    That limitation is returned in ``caveats`` rather than mentioned in a
    docstring, because it travels with the number into the report.
    """
    promo_path = ground_truth_dir / "promotion_uplift.json"
    elasticity_path = ground_truth_dir / "elasticity.json"
    if not promo_path.is_file() or not elasticity_path.is_file():
        return None

    responses: dict[str, Any] = json.loads(promo_path.read_text(encoding="utf-8"))["values"]
    elasticities: dict[str, float] = json.loads(
        elasticity_path.read_text(encoding="utf-8")
    )["values"]

    mechanic_terms: list[float] = []
    price_terms: list[float] = []

    for event in events.itertuples(index=False):
        product = str(event.product_id)
        depth = float(getattr(event, "discount_depth", 0.0))
        mechanic_type = getattr(event, "promotion_type", None)

        per_type = responses.get(product)
        if not per_type:
            continue
        curve = per_type.get(str(mechanic_type)) if mechanic_type else None
        if curve is None:
            # Unknown mechanic: average the product's curves rather than skip.
            # Skipping would drop exactly the events whose type failed to join,
            # which is a selection, not a gap.
            a = float(np.mean([c["a"] for c in per_type.values()]))
            b = float(np.mean([c["b"] for c in per_type.values()]))
        else:
            a, b = float(curve["a"]), float(curve["b"])

        mechanic_terms.append(a * (1.0 - np.exp(-b * depth)))
        beta = float(elasticities.get(product, 0.0))
        price_terms.append(beta * np.log(max(1.0 - depth, 1e-6)))

    if not mechanic_terms:
        return None

    mechanic_log = float(np.mean(mechanic_terms))
    price_log = float(np.mean(price_terms))

    return GroundTruthComparison(
        estimated_pct=float("nan"),
        expected_pct=float(np.expm1(mechanic_log + price_log)),
        n_events=len(mechanic_terms),
        mechanic_pct=float(np.expm1(mechanic_log)),
        price_channel_pct=float(np.expm1(price_log)),
        caveats=[
            "The store-level promo responsiveness and the per-event regional "
            "factor are not persisted by the generator, so this validates the "
            "AVERAGE effect only. No per-event effect can be checked.",
            "Both unpersisted terms have expectation near 1, so the average is "
            "recoverable; the regional factor is floored at 0.4, which biases "
            "the true mean slightly above this expectation.",
            "Expected effect is computed on the log scale and converted once at "
            "the end. Averaging per-event percentages instead would overstate "
            "it, by Jensen's inequality.",
        ],
    )


def validate_against_ground_truth(
    estimate: EffectEstimate,
    events: pd.DataFrame,
    ground_truth_dir: Path,
) -> GroundTruthComparison | None:
    """Compare an estimate against the generator's recorded parameters."""
    expected = expected_effect_from_ground_truth(events, ground_truth_dir)
    if expected is None:
        return None
    comparison = GroundTruthComparison(
        estimated_pct=estimate.ate_pct,
        expected_pct=expected.expected_pct,
        n_events=expected.n_events,
        mechanic_pct=expected.mechanic_pct,
        price_channel_pct=expected.price_channel_pct,
        caveats=expected.caveats,
    )
    logger.info(
        "promo_uplift.ground_truth_compared",
        estimated=round(comparison.estimated_pct, 4),
        expected=round(comparison.expected_pct, 4),
        error=round(comparison.absolute_error, 4),
    )
    return comparison


__all__ = [
    "GroundTruthComparison",
    "PlaceboResult",
    "SensitivityResult",
    "SensitivityRow",
    "ValidationVerdict",
    "evaluate_placebo",
    "expected_effect_from_ground_truth",
    "judge",
    "placebo_frame",
    "validate_against_ground_truth",
]
