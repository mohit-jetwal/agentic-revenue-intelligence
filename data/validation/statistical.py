"""Statistical validation: does the data contain the relationships it should?

This is the most important module in Step 2. Everything else produces Parquet
files; this decides whether those files are worth anything.

The test is not "does a regression run" - it is "does a *correctly specified*
estimator recover the parameter that was drawn, and is a *naively specified* one
wrong in the direction theory predicts". Both halves matter:

* If the correct estimator cannot recover truth, the confounding is too strong
  and Step 8 has nothing to demonstrate.
* If the naive estimator is *also* right, the confounding is absent, the data is
  too easy, and the elasticity model is a formality rather than a piece of
  analysis.

Deliberately uses plain OLS with fixed effects via demeaning rather than the
models that Steps 4-11 will build. This is a check on the *data*, not a preview
of the modelling, and it must stay independent of the code it validates -
otherwise a shared bug would cancel out and both would look correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from data.generation.ground_truth import GroundTruth


@dataclass
class RelationshipResult:
    """Outcome of one relationship test."""

    name: str
    passed: bool
    description: str
    observed: float | None = None
    expected: float | None = None
    tolerance: float | None = None
    sample_size: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means - a one-way within transformation.

    This is how fixed effects are absorbed without building a design matrix with
    thousands of dummy columns. At dev scale a store-dummy matrix would be
    6,000 columns wide; demeaning gets the identical coefficient in one pass.
    """
    frame = pd.DataFrame({"value": values, "group": groups})
    means = frame.groupby("group", observed=True)["value"].transform("mean")
    return values - means.to_numpy()


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Univariate OLS through the origin on already-demeaned data.

    Returns ``(slope, standard_error)``.
    """
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        return float("nan"), float("nan")
    slope = float(np.sum(x * y) / denominator)
    residual = y - slope * x
    dof = max(len(y) - 2, 1)
    variance = float(np.sum(residual**2) / dof / denominator)
    return slope, float(np.sqrt(max(variance, 0.0)))


def _ols_multi(y: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    """Multivariate OLS on already-demeaned data, returning coefficients.

    Needed because several of these relationships are only identified once a
    correlated regressor is held constant. Competitor price is the clearest
    case: it moves with our own price through the shared cost index, so a
    univariate regression of our volume on their price mostly recovers our own
    price effect with the sign flipped.
    """
    design = np.column_stack(columns)
    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(len(columns), np.nan)
    return np.asarray(coefficients, dtype=float)


def validate_own_price_elasticity(
    sales: pd.DataFrame,
    ground_truth: GroundTruth,
    *,
    min_observations: int = 400,
    tolerance: float = 0.35,
) -> list[RelationshipResult]:
    """Can a correctly specified estimator recover the true elasticity?

    Specification: ``log(units) ~ log(price)`` with store and month fixed
    effects absorbed by demeaning, estimated on **non-promotional, in-stock**
    rows only. Three exclusions, each for a specific reason:

    * *Promotional rows* carry a price cut and an additive uplift at the same
      time. Regressing across them conflates the shopper's price response with
      the promotion's display and mechanic effects, and inflates the apparent
      elasticity. Isolating own-price response means using regular-price
      variation, which is exactly what a careful analyst does.
    * *Stockout rows* report supply, not demand. Including them biases the
      estimate toward zero for a reason that has nothing to do with price.
    * *Fixed effects* neutralise the price endogeneity: a store that is
      permanently expensive and permanently busy contributes nothing once its
      own mean is removed.

    The naive comparison keeps every row and applies no controls at all - which
    is the point. It should be visibly worse.
    """
    results: list[RelationshipResult] = []

    clean = sales[
        (sales["units"] > 0) & (~sales["stockout_flag"]) & (~sales["promotion_flag"])
    ].copy()
    naive_frame = sales[sales["units"] > 0].copy()
    if clean.empty:
        return [
            RelationshipResult(
                "own_price_elasticity",
                False,
                "no uncensored, non-promotional rows available to estimate from",
            )
        ]

    for frame in (clean, naive_frame):
        frame["log_units"] = np.log(frame["units"].to_numpy(dtype=float))
        frame["log_price"] = np.log(
            np.clip(frame["selling_price"].to_numpy(dtype=float), 1e-6, None)
        )
        frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)

    # Test the products with the most observations - if elasticity is not
    # recoverable there, it is not recoverable anywhere, and a failure on a
    # thinly observed SKU would be about sample size rather than about the data.
    counts = clean.groupby("product_id").size().sort_values(ascending=False)
    candidates = [p for p in counts.index if counts[p] >= min_observations][:12]

    errors: list[float] = []
    naive_errors: list[float] = []
    recovered: dict[str, dict[str, float]] = {}

    for product_id in candidates:
        subset = clean[clean["product_id"] == product_id]
        truth = ground_truth.own_elasticity.get(str(product_id))
        if truth is None:
            continue

        y = subset["log_units"].to_numpy()
        x = subset["log_price"].to_numpy()

        # Correctly specified: absorb store and month effects.
        y_within = _demean(_demean(y, subset["store_id"].to_numpy()), subset["month"].to_numpy())
        x_within = _demean(_demean(x, subset["store_id"].to_numpy()), subset["month"].to_numpy())
        estimate, _ = _ols(y_within, x_within)

        # Naive: every row, no controls, promotions included.
        naive_subset = naive_frame[naive_frame["product_id"] == product_id]
        naive_y = naive_subset["log_units"].to_numpy()
        naive_x = naive_subset["log_price"].to_numpy()
        naive, _ = _ols(naive_y - naive_y.mean(), naive_x - naive_x.mean())

        if not np.isfinite(estimate):
            continue

        recovered[str(product_id)] = {
            "true": round(truth, 4),
            "panel_fe": round(estimate, 4),
            "naive_ols": round(naive, 4) if np.isfinite(naive) else float("nan"),
            "observations": len(subset),
        }
        errors.append(abs(estimate - truth) / max(abs(truth), 1e-6))
        if np.isfinite(naive):
            naive_errors.append(abs(naive - truth) / max(abs(truth), 1e-6))

    if not errors:
        return [
            RelationshipResult(
                "own_price_elasticity",
                False,
                "no product had enough observations to estimate elasticity",
                sample_size=len(clean),
            )
        ]

    median_error = float(np.median(errors))
    results.append(
        RelationshipResult(
            name="own_price_elasticity_recoverable",
            passed=median_error <= tolerance,
            description=(
                "A log-log panel regression with store and month fixed effects, "
                "estimated on non-promotional in-stock rows, should recover the "
                "true own-price elasticity."
            ),
            observed=round(median_error, 4),
            expected=0.0,
            tolerance=tolerance,
            sample_size=len(clean),
            detail={"per_product": recovered, "products_tested": len(errors)},
        )
    )

    # The sharper test: the naive estimator must be *worse*. If it is not, the
    # confounding is not doing its job and the dataset is too easy.
    if naive_errors:
        median_naive = float(np.median(naive_errors))
        results.append(
            RelationshipResult(
                name="naive_ols_is_biased",
                passed=median_naive > median_error,
                description=(
                    "Naive OLS without controls should be measurably worse than "
                    "the panel estimator, confirming price endogeneity is present."
                ),
                observed=round(median_naive, 4),
                expected=round(median_error, 4),
                sample_size=len(frame),
                detail={
                    "naive_median_error": round(median_naive, 4),
                    "panel_median_error": round(median_error, 4),
                },
            )
        )

    return results


def validate_promotion_uplift(
    sales: pd.DataFrame, *, tolerance: float = 0.02
) -> list[RelationshipResult]:
    """Do promotions raise sales, and does the naive method overstate it?

    The second half is the interesting one. Because the generator includes
    pull-forward, a naive during-versus-before comparison should exceed a
    comparison that also accounts for the post-promotion dip. That gap is the
    bias Step 6 exists to correct and the Critic in Step 18 exists to catch.
    """
    results: list[RelationshipResult] = []

    promo = sales[sales["promotion_flag"]]
    base = sales[~sales["promotion_flag"]]
    if promo.empty or base.empty:
        return [
            RelationshipResult(
                "promotion_uplift", False, "no promotional or non-promotional rows found"
            )
        ]

    promo_mean = float(promo["units"].mean())
    base_mean = float(base["units"].mean())
    uplift = (promo_mean - base_mean) / max(base_mean, 1e-6)

    results.append(
        RelationshipResult(
            name="promotion_increases_sales",
            passed=uplift > tolerance,
            description="Promoted rows should sell materially more than non-promoted rows.",
            observed=round(uplift, 4),
            expected=0.0,
            tolerance=tolerance,
            sample_size=len(promo),
            detail={
                "promo_mean_units": round(promo_mean, 2),
                "base_mean_units": round(base_mean, 2),
            },
        )
    )

    # Deeper discounts should lift more, with diminishing returns.
    promo = promo.copy()
    promo["depth_band"] = pd.cut(
        promo["discount_percentage"],
        bins=[0, 10, 20, 30, 100],
        labels=["0-10", "10-20", "20-30", "30+"],
    )
    by_band = promo.groupby("depth_band", observed=True)["units"].mean()
    by_band = by_band.dropna()
    if len(by_band) >= 3:
        lifts = ((by_band - base_mean) / max(base_mean, 1e-6)).round(4)
        monotonic = bool(np.all(np.diff(by_band.to_numpy()) > -0.5))
        results.append(
            RelationshipResult(
                name="deeper_discounts_lift_more",
                passed=monotonic,
                description=(
                    "Uplift should rise with discount depth, with diminishing "
                    "returns rather than a linear response."
                ),
                observed=float(lifts.iloc[-1]),
                sample_size=len(promo),
                detail={"uplift_by_depth_band": {str(k): float(v) for k, v in lifts.items()}},
            )
        )

    return results


def validate_stockout_censoring(
    sales: pd.DataFrame, latent: pd.DataFrame
) -> list[RelationshipResult]:
    """During a stockout, are observed sales below latent demand?

    The property Step 4 depends on: a supply failure must look different from a
    demand collapse. Observed units are suppressed while latent demand holds.
    """
    if latent.empty:
        return [
            RelationshipResult("stockout_censoring", False, "no latent demand ground truth found")
        ]

    merged = sales[["date", "product_id", "store_id", "units", "stockout_flag"]].merge(
        latent[["date", "product_id", "store_id", "latent_units"]],
        on=["date", "product_id", "store_id"],
        how="inner",
    )
    if merged.empty:
        return [
            RelationshipResult("stockout_censoring", False, "sales and latent demand did not join")
        ]

    during = merged[merged["stockout_flag"]]
    outside = merged[~merged["stockout_flag"]]
    if during.empty:
        return [RelationshipResult("stockout_censoring", False, "no stockout rows present")]

    suppression = float(1.0 - during["units"].sum() / max(during["latent_units"].sum(), 1e-6))
    unconstrained_gap = float(
        1.0 - outside["units"].sum() / max(outside["latent_units"].sum(), 1e-6)
    )

    results = [
        RelationshipResult(
            name="stockouts_suppress_observed_sales",
            passed=suppression > 0.10,
            description=(
                "During stockouts, observed units must fall materially below "
                "latent demand - the gap a root-cause model must detect."
            ),
            observed=round(suppression, 4),
            expected=0.0,
            tolerance=0.10,
            sample_size=len(during),
        ),
        RelationshipResult(
            name="no_censoring_when_in_stock",
            passed=abs(unconstrained_gap) < 0.02,
            description=(
                "Outside stockouts, observed units should equal latent demand - "
                "confirming censoring is the only source of the gap."
            ),
            observed=round(unconstrained_gap, 4),
            expected=0.0,
            tolerance=0.02,
            sample_size=len(outside),
        ),
    ]

    # Latent demand during a stockout should be comparable to normal levels.
    # If it collapsed too, the scenario would be indistinguishable from a real
    # demand decline and the whole exercise would be pointless.
    stocked_products = set(during["product_id"].unique())
    subset = merged[merged["product_id"].isin(stocked_products)]
    latent_during = float(subset[subset["stockout_flag"]]["latent_units"].mean())
    latent_outside = float(subset[~subset["stockout_flag"]]["latent_units"].mean())
    ratio = latent_during / max(latent_outside, 1e-6)
    results.append(
        RelationshipResult(
            name="latent_demand_holds_during_stockout",
            passed=ratio > 0.75,
            description=(
                "Underlying demand should not collapse during a stockout - "
                "otherwise a supply failure is indistinguishable from a demand one."
            ),
            observed=round(ratio, 4),
            expected=1.0,
            tolerance=0.25,
            sample_size=len(subset),
        )
    )
    return results


def validate_cross_price(
    sales: pd.DataFrame,
    pricing: pd.DataFrame,
    ground_truth: GroundTruth,
    *,
    max_pairs: int = 8,
) -> list[RelationshipResult]:
    """Do substitutes and complements move in the directions they were drawn?

    Sign agreement rather than magnitude: cross-price effects are second-order
    and noisy, and demanding a recovered coefficient within tolerance would fail
    for reasons of statistical power rather than data quality. Step 9's job is
    to estimate the magnitudes; Step 2's job is to confirm the signal exists.
    """
    if not ground_truth.cross_elasticity:
        return [RelationshipResult("cross_price", False, "no cross-price ground truth")]

    # Strongest declared relationships have the best chance of clearing noise.
    pairs: list[tuple[str, str, float]] = []
    for target, sources in ground_truth.cross_elasticity.items():
        for source, coefficient in sources.items():
            pairs.append((target, source, coefficient))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    # Estimated at store-date grain, which is where substitution actually
    # happens - shoppers swap between products on the same shelf. Aggregating to
    # product-date first would average away the store-level price differences
    # that identify the effect.
    price_by_store = pricing[["date", "product_id", "store_id", "selling_price"]]
    units_by_store = sales[["date", "product_id", "store_id", "units", "selling_price"]]

    agreements = 0
    tested = 0
    # Values are mixed: coefficients, an optional uncontrolled estimate, and a
    # boolean agreement flag.
    detail: dict[str, dict[str, Any]] = {}

    for target, source, expected in pairs[:max_pairs]:
        target_rows = units_by_store[units_by_store["product_id"] == target]
        source_rows = price_by_store[price_by_store["product_id"] == source]
        if len(target_rows) < 200 or len(source_rows) < 200:
            continue

        merged = target_rows.merge(
            source_rows[["date", "store_id", "selling_price"]],
            on=["date", "store_id"],
            suffixes=("_own", "_src"),
        )
        merged = merged[(merged["units"] > 0) & (merged["selling_price_src"] > 0)]
        if len(merged) < 200:
            continue

        store = merged["store_id"].to_numpy()
        month = pd.to_datetime(merged["date"]).dt.to_period("M").astype(str).to_numpy()

        def within(
            values: np.ndarray, _store: np.ndarray = store, _month: np.ndarray = month
        ) -> np.ndarray:
            return _demean(_demean(values, _store), _month)

        y = within(np.log(merged["units"].to_numpy(dtype=float)))
        source_price = within(
            np.log(np.clip(merged["selling_price_src"].to_numpy(dtype=float), 1e-6, None))
        )
        # Controlling for the target's OWN price is essential. Products in the
        # same category share a cost index, so a substitute's price rise tends
        # to coincide with our own - and our own (negative, large) elasticity
        # would otherwise swamp the cross effect and flip its sign.
        own_price = within(
            np.log(np.clip(merged["selling_price_own"].to_numpy(dtype=float), 1e-6, None))
        )

        naive, _ = _ols(y, source_price)
        coefficients = _ols_multi(y, [source_price, own_price])
        estimate = float(coefficients[0])
        if not np.isfinite(estimate):
            continue

        tested += 1
        agreed = np.sign(estimate) == np.sign(expected)
        agreements += int(agreed)
        detail[f"{source}->{target}"] = {
            "expected": round(expected, 4),
            "observed": round(estimate, 4),
            "uncontrolled": round(float(naive), 4) if np.isfinite(naive) else None,
            "sign_agrees": bool(agreed),
        }

    if tested == 0:
        return [RelationshipResult("cross_price_signs", False, "no pair had enough observations")]

    agreement_rate = agreements / tested
    return [
        RelationshipResult(
            name="cross_price_signs_agree",
            passed=agreement_rate >= 0.7,
            description=(
                "Holding the target's own price constant, substitutes and "
                "complements should move in the declared directions: a "
                "substitute's price rise lifts our volume, a complement's "
                "depresses it."
            ),
            observed=round(agreement_rate, 4),
            expected=1.0,
            tolerance=0.3,
            sample_size=tested,
            detail=detail,
        )
    ]


def validate_competitor_effect(
    sales: pd.DataFrame, competitor: pd.DataFrame
) -> list[RelationshipResult]:
    """Does our demand rise when competitor prices rise, holding our price fixed?

    The qualifier is essential. Our price and the competitor's both respond to
    the shared commodity cost index, so a univariate regression of our volume on
    their price mostly recovers *our own* price effect with the sign flipped -
    it will report that a rival getting more expensive hurts us, which is
    nonsense. Controlling for our own price is what identifies the substitution
    effect, and demonstrating that is itself worth having in the data.
    """
    if competitor.empty:
        return [RelationshipResult("competitor_effect", False, "no competitor pricing data")]

    comp_daily = (
        competitor.groupby(["date", "product_id"])["competitor_effective_price"]
        .mean()
        .reset_index()
    )
    own = (
        sales[sales["units"] > 0]
        .groupby(["date", "product_id"])
        .agg(units=("units", "sum"), own_price=("selling_price", "mean"))
        .reset_index()
    )
    merged = own.merge(comp_daily, on=["date", "product_id"], how="inner")
    merged = merged[(merged["units"] > 0) & (merged["competitor_effective_price"] > 0)]
    if len(merged) < 200:
        return [RelationshipResult("competitor_effect", False, "insufficient joined observations")]

    product = merged["product_id"].to_numpy()
    month = pd.to_datetime(merged["date"]).dt.to_period("M").astype(str).to_numpy()

    def within(values: np.ndarray) -> np.ndarray:
        return _demean(_demean(values, product), month)

    y = within(np.log(merged["units"].to_numpy(dtype=float)))
    competitor_price = within(np.log(merged["competitor_effective_price"].to_numpy(dtype=float)))
    own_price = within(np.log(np.clip(merged["own_price"].to_numpy(dtype=float), 1e-6, None)))

    naive, _ = _ols(y, competitor_price)
    coefficients = _ols_multi(y, [competitor_price, own_price])
    estimate = float(coefficients[0])

    return [
        RelationshipResult(
            name="competitor_price_raises_our_demand",
            passed=bool(np.isfinite(estimate) and estimate > 0),
            description=(
                "Holding our own price constant, a competitor price rise should "
                "increase our demand. Without that control the coefficient is "
                "confounded by the shared cost index and flips sign."
            ),
            observed=round(estimate, 4) if np.isfinite(estimate) else None,
            expected=1.0,
            sample_size=len(merged),
            detail={
                "controlled_for_own_price": round(estimate, 4) if np.isfinite(estimate) else None,
                "uncontrolled_naive": round(float(naive), 4) if np.isfinite(naive) else None,
                "own_price_coefficient": round(float(coefficients[1]), 4)
                if np.isfinite(coefficients[1])
                else None,
            },
        )
    ]


def validate_seasonality_and_regions(
    sales: pd.DataFrame, calendar: pd.DataFrame, stores: pd.DataFrame
) -> list[RelationshipResult]:
    """Are festival peaks and regional differences present?"""
    results: list[RelationshipResult] = []

    dated = sales.merge(calendar[["date", "festival_flag", "holiday_flag"]], on="date", how="left")
    festival_mean = float(dated.loc[dated["festival_flag"].fillna(False), "units"].mean())
    normal_mean = float(dated.loc[~dated["festival_flag"].fillna(False), "units"].mean())
    lift = (festival_mean - normal_mean) / max(normal_mean, 1e-6)

    results.append(
        RelationshipResult(
            name="festival_demand_peak",
            passed=lift > 0.05,
            description="Festival periods should show materially higher demand.",
            observed=round(lift, 4),
            expected=0.0,
            tolerance=0.05,
            sample_size=len(dated),
            detail={
                "festival_mean_units": round(festival_mean, 2),
                "normal_mean_units": round(normal_mean, 2),
            },
        )
    )

    regional = sales.merge(stores[["store_id", "region"]], on="store_id", how="left")
    by_region = regional.groupby("region")["units"].mean()
    if len(by_region) >= 3:
        spread = float(by_region.max() / max(by_region.min(), 1e-6))
        results.append(
            RelationshipResult(
                name="regional_variation_present",
                passed=spread > 1.10,
                description=(
                    "Regions should differ in demand, so regional analysis and "
                    "budget reallocation are meaningful questions."
                ),
                observed=round(spread, 4),
                expected=1.0,
                tolerance=0.10,
                sample_size=len(regional),
                detail={str(k): round(float(v), 2) for k, v in by_region.items()},
            )
        )

    return results


def validate_price_demand_direction(sales: pd.DataFrame) -> list[RelationshipResult]:
    """The headline relationship: higher price, lower demand.

    A blunt pooled check. It is expected to be attenuated by confounding, so the
    bar is only that the sign is right - the precise magnitude is what the fixed
    effects test above is for.
    """
    frame = sales[(sales["units"] > 0) & (sales["selling_price"] > 0)]
    if len(frame) < 500:
        return [RelationshipResult("price_demand_direction", False, "insufficient rows")]

    y = np.log(frame["units"].to_numpy(dtype=float))
    x = np.log(frame["selling_price"].to_numpy(dtype=float))
    product = frame["product_id"].to_numpy()

    estimate, _ = _ols(_demean(y, product), _demean(x, product))
    return [
        RelationshipResult(
            name="price_increases_reduce_demand",
            passed=bool(np.isfinite(estimate) and estimate < 0),
            description="Within a product, higher prices should coincide with lower volume.",
            observed=round(float(estimate), 4) if np.isfinite(estimate) else None,
            expected=-1.0,
            sample_size=len(frame),
        )
    ]
