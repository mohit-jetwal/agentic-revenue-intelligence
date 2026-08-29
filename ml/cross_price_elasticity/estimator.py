"""Cross-price elasticity: how B's price moves A's demand.

Sign convention, since it is the whole point:

* **positive** => substitutes (B gets dearer, A sells more)
* **negative** => complements (B gets dearer, A sells less)

Getting it backwards inverts every assortment and cannibalisation conclusion
drawn from it, which is why the sign is asserted in the tests rather than
assumed from the coefficient.

**The multiple-comparisons problem is the real difficulty.** With 300 products
there are 89,700 ordered pairs. Test them all at the 5% level and roughly 4,485
come back "significant" with no relationship whatsoever. Two defences:

*Restrict the candidate set.* Only within-category pairs and pairs the
relationship table already declares. This is not a shortcut — a cross-price
effect between shampoo and frozen peas is not a finding, it is a coincidence
that survived a t-test.

*Correct what remains.* Benjamini-Hochberg over the surviving pairs, which
controls the false discovery rate rather than the family-wise error rate.
Bonferroni would be the stricter choice and the wrong one: with a few dozen
candidates it would leave nothing significant and hide the real substitutions.

Estimation is the same within-transformation as own-price: demand on the focal
product's own log price *and* the candidate's, with listing and date fixed
effects. Omitting the own-price term would let the two prices' shared movement -
category cost shocks move both - load onto the cross coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from app.observability.logging import get_logger
from app.schemas.domain import RelationshipType

logger = get_logger(__name__)

#: Minimum overlapping observations before a pair is worth testing.
MIN_PAIR_ROWS = 120

#: |elasticity| bands for the ``strength`` label.
_STRONG = 0.30
_MODERATE = 0.10


@dataclass
class PairEstimate:
    """One directed pair's cross-price coefficient."""

    source_product_id: str
    target_product_id: str
    cross_elasticity: float
    standard_error: float
    p_value: float
    #: Benjamini-Hochberg adjusted p-value. Populated by :func:`adjust_p_values`.
    adjusted_p_value: float | None = None
    n_obs: int = 0
    own_price_coefficient: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def relationship(self) -> RelationshipType:
        """Substitute, complement, or unrelated.

        Judged on the **adjusted** p-value where one exists. Using the raw value
        is how a candidate set of forty pairs produces two spurious findings.
        """
        p = self.adjusted_p_value if self.adjusted_p_value is not None else self.p_value
        if p > 0.05 or abs(self.cross_elasticity) < _MODERATE:
            return RelationshipType.UNRELATED
        return (
            RelationshipType.SUBSTITUTE
            if self.cross_elasticity > 0
            else RelationshipType.COMPLEMENT
        )

    @property
    def strength(self) -> str:
        magnitude = abs(self.cross_elasticity)
        if self.relationship is RelationshipType.UNRELATED:
            return "none"
        if magnitude >= _STRONG:
            return "strong"
        return "moderate" if magnitude >= _MODERATE else "weak"

    @property
    def is_significant(self) -> bool:
        p = self.adjusted_p_value if self.adjusted_p_value is not None else self.p_value
        return p <= 0.05

    def confidence_interval(self) -> tuple[float, float]:
        critical = 1.96
        return (
            self.cross_elasticity - critical * self.standard_error,
            self.cross_elasticity + critical * self.standard_error,
        )


def candidate_pairs(
    focal_product: str,
    products: pd.DataFrame,
    relationships: pd.DataFrame | None = None,
    *,
    max_candidates: int = 40,
) -> list[str]:
    """Products worth testing against the focal one.

    Declared relationships first, then same-category products. Both are
    restrictions on the hypothesis space made *before* looking at the outcome,
    which is what keeps the multiple-comparisons correction honest — choosing
    candidates after seeing which ones look significant would invalidate it.
    """
    candidates: list[str] = []

    if relationships is not None and not relationships.empty:
        declared = relationships[
            (relationships["product_a"] == focal_product)
            & (relationships.get("relationship_type", "unrelated") != "unrelated")
        ]
        candidates.extend(declared["product_b"].astype(str).tolist())

    focal_row = products[products["product_id"] == focal_product]
    if not focal_row.empty and "category" in products.columns:
        category = focal_row.iloc[0]["category"]
        same = products[
            (products["category"] == category) & (products["product_id"] != focal_product)
        ]["product_id"].astype(str)
        candidates.extend(same.tolist())

    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate != focal_product:
            seen.setdefault(candidate, None)
    return list(seen)[:max_candidates]


def estimate_pair(
    focal: pd.DataFrame,
    source: pd.DataFrame,
    *,
    focal_product: str,
    source_product: str,
) -> PairEstimate:
    """Cross-price coefficient for one directed pair.

    Joined on ``(date, store_id)``: substitution happens on a shelf, so the
    relevant price is the candidate's price *in the same store on the same day*.
    A national average price would blur exactly the store-level variation that
    identifies the effect.

    **The two panels are prepared differently, and it matters more than it
    looks.** The focal panel drops promoted rows, because a promotion on the
    focal product moves its own demand through a mechanic that has nothing to do
    with the candidate's price. The source panel **keeps** them, because the
    candidate's promotional price cut is the single largest source of the price
    variation that identifies the cross effect.

    Measured on P00003/P00036: dropping the source's promoted rows cut its
    log-price standard deviation from 0.155 to 0.120 and the joined sample from
    4,999 rows to 3,640 — the true +0.44 substitute then failed to surface at
    all. Deleting a product's promotions to "clean" its price series deletes the
    experiment.
    """
    merged = focal.merge(
        source[["date", "store_id", "log_price"]].rename(
            columns={"log_price": "log_price_source"}
        ),
        on=["date", "store_id"],
        how="inner",
    )
    if len(merged) < MIN_PAIR_ROWS:
        raise ValueError(
            f"only {len(merged)} overlapping store-days for "
            f"{focal_product}/{source_product}; need {MIN_PAIR_ROWS}"
        )

    merged["_unit"] = merged["product_id"].astype(str) + "|" + merged["store_id"].astype(str)

    def demean(column: str) -> np.ndarray:
        series = merged[column].astype(float)
        by_unit = series - series.groupby(merged["_unit"].to_numpy()).transform("mean")
        return (by_unit - by_unit.groupby(merged["date"].to_numpy()).transform("mean")).to_numpy()

    y = demean("log_units")
    # Own price is included deliberately. Category cost shocks move both prices
    # together, so omitting the own-price term lets that shared movement load
    # onto the cross coefficient and manufactures a relationship.
    X = np.column_stack([demean("log_price"), demean("log_price_source")])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    xtx_inv = np.linalg.pinv(X.T @ X)
    covariance = xtx_inv @ (X.T @ np.diag(residuals**2) @ X) @ xtx_inv
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    dof = max(len(merged) - 2, 1)
    coefficient = float(beta[1])
    se = float(errors[1])
    p_value = float(2 * (1 - stats.t.cdf(abs(coefficient / se), dof))) if se > 0 else 1.0

    return PairEstimate(
        source_product_id=source_product,
        target_product_id=focal_product,
        cross_elasticity=coefficient,
        standard_error=se,
        p_value=p_value,
        n_obs=len(merged),
        own_price_coefficient=float(beta[0]),
    )


def adjust_p_values(pairs: list[PairEstimate]) -> list[PairEstimate]:
    """Benjamini-Hochberg false-discovery-rate correction, in place.

    FDR rather than Bonferroni. With a few dozen candidates Bonferroni would
    leave nothing significant and hide the real substitutions; controlling the
    expected *proportion* of false discoveries is the right trade when the
    candidate set has already been restricted to plausible pairs.
    """
    if not pairs:
        return pairs

    ordered = sorted(pairs, key=lambda p: p.p_value)
    n = len(ordered)
    running_min = 1.0
    # Walk from the largest p-value down, carrying the running minimum. That
    # enforces monotonicity, without which an adjusted p-value could exceed one
    # computed from a larger raw value.
    for rank in range(n, 0, -1):
        estimate = ordered[rank - 1]
        adjusted = min(1.0, estimate.p_value * n / rank)
        running_min = min(running_min, adjusted)
        estimate.adjusted_p_value = running_min
    return pairs


def estimate_cross_elasticities(
    panels: dict[str, pd.DataFrame],
    focal_product: str,
    candidates: list[str],
) -> tuple[list[PairEstimate], int]:
    """Test the focal product against every candidate.

    Returns the estimates and the number of pairs tested — the second is needed
    to interpret significance honestly, and a result that reports only the
    survivors is a result that has hidden its own denominator.
    """
    focal = panels.get(focal_product)
    if focal is None or focal.empty:
        raise ValueError(f"no panel for focal product {focal_product}")

    pairs: list[PairEstimate] = []
    tested = 0
    for candidate in candidates:
        source = panels.get(candidate)
        if source is None or source.empty:
            continue
        tested += 1
        try:
            pairs.append(
                estimate_pair(
                    focal, source, focal_product=focal_product, source_product=candidate
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.info(
                "cross_elasticity.pair_skipped", pair=f"{focal_product}/{candidate}",
                error=str(exc),
            )

    adjust_p_values(pairs)
    logger.info(
        "cross_elasticity.estimated",
        focal=focal_product,
        tested=tested,
        estimated=len(pairs),
        significant=sum(p.is_significant for p in pairs),
    )
    return pairs, tested


__all__ = [
    "MIN_PAIR_ROWS",
    "PairEstimate",
    "adjust_p_values",
    "candidate_pairs",
    "estimate_cross_elasticities",
    "estimate_pair",
]
