"""Matching and covariate balance (brief sections 16, 22).

Balance is the diagnostic that decides whether an adjustment worked. Everything
else in this package - propensity models, weights, doubly robust corrections -
exists to make the treated and control groups comparable, and the standardised
mean difference is how you check whether they now are.

**Why the standardised mean difference and not a t-test.** A t-test on a
covariate answers "is this difference distinguishable from zero", which is a
question about sample size. With 40,000 rows a 0.5% difference in trailing demand
is highly significant and completely irrelevant; with 200 rows a 40% difference
can be non-significant and fatal. The SMD is the difference in units of pooled
standard deviation, so it measures *how far apart* the groups are rather than how
confident we are that they differ at all. The 0.1 convention comes from the
matching literature and is used here as a threshold, not a law.

**Why matching as well as weighting.** They fail differently, which is the point
of having both. Weighting keeps every observation but can hand enormous influence
to a handful of them. Matching discards unmatched treated units - changing the
estimand to "the effect on promotions that had a comparable control" - but every
retained pair is concretely comparable, and you can look at the pairs. Agreement
between the two is evidence; disagreement is a finding worth chasing.

**What matching does not do.** It balances what you matched on. Two store-days
identical in trailing demand, price and season may still differ in something
nobody recorded, and no amount of matching addresses that. Matching improves
comparability on observables; it does not manufacture ignorability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config

logger = get_logger(__name__)


@dataclass
class BalanceRow:
    """One covariate's balance, before and after adjustment."""

    covariate: str
    treated_mean: float
    control_mean: float
    control_mean_weighted: float
    smd_before: float
    smd_after: float

    @property
    def improved(self) -> bool:
        return abs(self.smd_after) <= abs(self.smd_before)


@dataclass
class BalanceReport:
    """Covariate balance across the adjustment set."""

    rows: list[BalanceRow]
    threshold: float
    n_treated: int
    n_control: int

    @property
    def worst(self) -> BalanceRow | None:
        return max(self.rows, key=lambda r: abs(r.smd_after), default=None)

    @property
    def unbalanced(self) -> list[BalanceRow]:
        return [r for r in self.rows if abs(r.smd_after) > self.threshold]

    @property
    def satisfied(self) -> bool:
        return not self.unbalanced

    def max_smd(self) -> float:
        return max((abs(r.smd_after) for r in self.rows), default=0.0)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "covariate": r.covariate,
                    "treated_mean": r.treated_mean,
                    "control_mean": r.control_mean,
                    "control_mean_weighted": r.control_mean_weighted,
                    "smd_before": r.smd_before,
                    "smd_after": r.smd_after,
                    "balanced": abs(r.smd_after) <= self.threshold,
                }
                for r in self.rows
            ]
        )

    def summary(self) -> str:
        worst = self.worst
        detail = f"worst {worst.covariate} at {worst.smd_after:+.3f}" if worst else "no covariates"
        return (
            f"balance: {len(self.unbalanced)}/{len(self.rows)} covariates above "
            f"{self.threshold:.2f} SMD ({detail})"
        )


def standardised_difference(
    values: np.ndarray, treated: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted standardised mean difference for one covariate.

    The denominator uses the **unweighted** pooled standard deviation of both
    arms, deliberately. Recomputing it from the weighted sample makes the
    denominator move with the weights, so a weighting scheme that happened to
    inflate the spread would report better balance while changing nothing about
    the means. A fixed yardstick is the only way to compare before and after.
    """
    treated = treated.astype(bool)
    if not treated.any() or treated.all():
        return 0.0

    if weights is None:
        weights = np.ones_like(values, dtype=float)

    def weighted_mean(mask: np.ndarray) -> float:
        w = weights[mask]
        total = w.sum()
        return float(np.dot(values[mask], w) / total) if total > 0 else float("nan")

    mean_t = weighted_mean(treated)
    mean_c = weighted_mean(~treated)

    var_t = float(np.var(values[treated], ddof=1)) if treated.sum() > 1 else 0.0
    var_c = float(np.var(values[~treated], ddof=1)) if (~treated).sum() > 1 else 0.0
    pooled = np.sqrt((var_t + var_c) / 2.0)

    # A covariate with no variation is perfectly balanced by definition; 0/0
    # would otherwise surface as a NaN that looks like a failed check.
    if pooled < 1e-12:
        return 0.0
    return float((mean_t - mean_c) / pooled)


def balance_table(
    X: pd.DataFrame,
    t: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    numeric: tuple[str, ...] | None = None,
    categorical: tuple[str, ...] = (),
    config: PromoUpliftConfig | None = None,
) -> BalanceReport:
    """Balance for every covariate, before and after weighting.

    Categorical covariates are expanded to one indicator per level, because a
    category's *mean* is undefined but the share of rows in each level is
    exactly the thing that must match between arms.
    """
    settings = config or get_promo_uplift_config()
    treated = t.astype(bool)
    columns = list(numeric) if numeric is not None else [
        c for c in X.columns if c not in categorical
    ]

    rows: list[BalanceRow] = []
    for name in columns:
        values = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=float(np.nanmean(values)) if len(values) else 0.0)
        rows.append(_balance_row(name, values, treated, weights))

    for name in categorical:
        if name not in X.columns:
            continue
        series = X[name].astype(str)
        for level in sorted(series.unique()):
            indicator = (series == level).to_numpy(dtype=float)
            rows.append(_balance_row(f"{name}={level}", indicator, treated, weights))

    report = BalanceReport(
        rows=rows,
        threshold=settings.propensity.max_standardised_difference,
        n_treated=int(treated.sum()),
        n_control=int((~treated).sum()),
    )
    logger.info(
        "promo_uplift.balance_assessed",
        covariates=len(rows),
        unbalanced=len(report.unbalanced),
        max_smd=round(report.max_smd(), 4),
    )
    return report


def _balance_row(
    name: str, values: np.ndarray, treated: np.ndarray, weights: np.ndarray | None
) -> BalanceRow:
    before = standardised_difference(values, treated, None)
    after = standardised_difference(values, treated, weights) if weights is not None else before

    control = ~treated
    weighted_control = float("nan")
    if control.any():
        if weights is None:
            weighted_control = float(values[control].mean())
        else:
            w = weights[control]
            weighted_control = float(np.dot(values[control], w) / w.sum()) if w.sum() else float(
                "nan"
            )

    return BalanceRow(
        covariate=name,
        treated_mean=float(values[treated].mean()) if treated.any() else float("nan"),
        control_mean=float(values[control].mean()) if control.any() else float("nan"),
        control_mean_weighted=weighted_control,
        smd_before=before,
        smd_after=after,
    )


@dataclass
class MatchResult:
    """Matched treated/control pairs."""

    treated_index: np.ndarray
    control_index: np.ndarray
    #: Absolute propensity distance within each pair.
    distance: np.ndarray
    #: Treated rows with no control inside the caliper. Reported, not hidden -
    #: dropping them changes the estimand from "all promotions" to "promotions
    #: with a comparable control", and a caller must be able to see how many.
    unmatched_treated: int
    caliper: float

    @property
    def n_pairs(self) -> int:
        return len(self.treated_index)

    def summary(self) -> str:
        return (
            f"{self.n_pairs:,} matched pairs, {self.unmatched_treated:,} treated "
            f"rows unmatched within a caliper of {self.caliper:.4f}"
        )


def match_on_propensity(
    scores: np.ndarray,
    t: np.ndarray,
    *,
    caliper_sd: float = 0.2,
    with_replacement: bool = True,
    seed: int = 42,
) -> MatchResult:
    """Nearest-neighbour matching on the propensity score.

    Matching is done on the **logit** of the score rather than the score itself.
    On the probability scale the distance between 0.01 and 0.02 looks identical
    to the distance between 0.50 and 0.51, but the first pair differs by a factor
    of two in odds and the second by a few percent. The logit scale makes the
    metric behave sensibly at the extremes, where matching quality matters most.

    ``with_replacement`` defaults to True: reusing a good control beats accepting
    a poor one, and the bias/variance trade favours bias reduction when the
    control pool is the scarce resource. The cost is that a heavily reused
    control dominates the estimate, which is why the pair count and the
    unmatched count are both reported.
    """
    rng = np.random.default_rng(seed)
    treated = t.astype(bool)

    logit = _logit(scores)
    caliper = caliper_sd * float(np.std(logit))

    treated_idx = np.flatnonzero(treated)
    control_idx = np.flatnonzero(~treated)
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return MatchResult(
            treated_index=np.array([], dtype=int),
            control_index=np.array([], dtype=int),
            distance=np.array([]),
            unmatched_treated=len(treated_idx),
            caliper=caliper,
        )

    control_values = logit[control_idx]
    order = np.argsort(control_values)
    sorted_controls = control_values[order]
    sorted_idx = control_idx[order]

    # Shuffle the treated order so that, without replacement, the units matched
    # first are not systematically those with the smallest index - which on a
    # date-sorted panel would mean the earliest promotions get the best controls.
    processing = rng.permutation(treated_idx)

    used = np.zeros(len(sorted_idx), dtype=bool)
    matched_t: list[int] = []
    matched_c: list[int] = []
    distances: list[float] = []

    for i in processing:
        position = np.searchsorted(sorted_controls, logit[i])
        best = _nearest_available(
            sorted_controls, used, position, logit[i], with_replacement
        )
        if best is None:
            continue
        distance = abs(sorted_controls[best] - logit[i])
        if distance > caliper:
            continue
        matched_t.append(int(i))
        matched_c.append(int(sorted_idx[best]))
        distances.append(float(distance))
        if not with_replacement:
            used[best] = True

    return MatchResult(
        treated_index=np.array(matched_t, dtype=int),
        control_index=np.array(matched_c, dtype=int),
        distance=np.array(distances),
        unmatched_treated=len(treated_idx) - len(matched_t),
        caliper=caliper,
    )


def _nearest_available(
    sorted_values: np.ndarray,
    used: np.ndarray,
    position: int,
    target: float,
    with_replacement: bool,
) -> int | None:
    """Closest unused control to ``target``, walking outward from ``position``.

    Linear scan outward rather than a fresh sort per treated unit: the control
    array is already sorted, so the nearest candidate is adjacent to the
    insertion point and the walk stops as soon as both directions are worse than
    the best found.
    """
    n = len(sorted_values)
    left = min(position, n - 1)
    right = position

    best: int | None = None
    best_distance = float("inf")

    while left >= 0 or right < n:
        progressed = False
        for candidate in (left, right):
            if 0 <= candidate < n and (with_replacement or not used[candidate]):
                distance = abs(sorted_values[candidate] - target)
                if distance < best_distance:
                    best_distance = distance
                    best = candidate
                progressed = True
        # Once both frontiers are further away than the best pair found, no
        # candidate further out can improve on it - the array is sorted.
        if best is not None and progressed:
            left_gap = abs(sorted_values[left] - target) if left >= 0 else float("inf")
            right_gap = abs(sorted_values[right] - target) if right < n else float("inf")
            if min(left_gap, right_gap) > best_distance:
                break
        left -= 1
        right += 1
    return best


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, clipped away from the asymptotes."""
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


__all__ = [
    "BalanceReport",
    "BalanceRow",
    "MatchResult",
    "balance_table",
    "match_on_propensity",
    "standardised_difference",
]
