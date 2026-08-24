"""Relationship recovery - the tests that make the dataset falsifiable.

These are the highest-value tests in Step 2. Every other test confirms the data
is internally consistent; these confirm it is *learnable*.

Two assertions carry the weight:

* A correctly specified estimator recovers the true elasticity that was drawn
  before any sales existed. If this fails, Step 8 has nothing to demonstrate.
* A naively specified one is measurably worse. If *that* fails, the confounding
  built into the generator is absent, the data is too easy, and the elasticity
  model becomes a formality rather than analysis.

Marked ``statistical`` so they can be deselected if the CI budget is ever
threatened - but they should not be, because they are the point.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.generation.ground_truth import GroundTruth
from data.generation.pipeline import GenerationResult
from data.validation.report import validate_dataset
from data.validation.statistical import (
    validate_competitor_effect,
    validate_cross_price,
    validate_own_price_elasticity,
    validate_price_demand_direction,
    validate_promotion_uplift,
    validate_seasonality_and_regions,
    validate_stockout_censoring,
)

pytestmark = [pytest.mark.statistical, pytest.mark.data]


def _named(results: list, name: str):  # type: ignore[no-untyped-def]
    for result in results:
        if result.name == name:
            return result
    raise AssertionError(f"no result named {name!r}; got {[r.name for r in results]}")


# --- own-price elasticity --------------------------------------------------


def test_true_elasticity_is_recoverable(
    sales: pd.DataFrame, smoke_ground_truth: GroundTruth
) -> None:
    """A panel regression with fixed effects must land near the drawn value."""
    results = validate_own_price_elasticity(sales, smoke_ground_truth)
    recovery = _named(results, "own_price_elasticity_recoverable")

    assert recovery.passed, (
        f"median relative error {recovery.observed} exceeds tolerance "
        f"{recovery.tolerance}; the data may be over-confounded"
    )
    assert recovery.detail["products_tested"] >= 5


def test_recovered_elasticities_have_the_right_sign(
    sales: pd.DataFrame, smoke_ground_truth: GroundTruth
) -> None:
    results = validate_own_price_elasticity(sales, smoke_ground_truth)
    per_product = _named(results, "own_price_elasticity_recoverable").detail["per_product"]

    for product_id, values in per_product.items():
        assert values["panel_fe"] < 0, f"{product_id} recovered a non-negative elasticity"


def test_elastic_and_inelastic_products_are_distinguishable(
    sales: pd.DataFrame, smoke_ground_truth: GroundTruth
) -> None:
    """The recovered ordering must track the true ordering.

    A pricing recommendation turns on whether |e| > 1, so recovering the level
    is not enough - the model must also rank products correctly.
    """
    results = validate_own_price_elasticity(sales, smoke_ground_truth)
    per_product = _named(results, "own_price_elasticity_recoverable").detail["per_product"]

    frame = pd.DataFrame(per_product).T
    assert len(frame) >= 5
    correlation = frame["true"].corr(frame["panel_fe"], method="spearman")
    assert correlation > 0.7, f"rank correlation {correlation:.2f} is too weak"


def test_naive_estimator_is_visibly_biased(
    sales: pd.DataFrame, smoke_ground_truth: GroundTruth
) -> None:
    """Proves the confounding is real rather than decorative.

    If an uncontrolled regression recovered truth just as well, the generator
    would not be producing the endogeneity and promotional confounding it claims
    to, and Step 8's careful specification would be pointless.
    """
    results = validate_own_price_elasticity(sales, smoke_ground_truth)
    bias = _named(results, "naive_ols_is_biased")

    assert bias.passed
    assert bias.detail["naive_median_error"] > bias.detail["panel_median_error"] * 1.5


# --- price direction -------------------------------------------------------


def test_higher_prices_reduce_demand(sales: pd.DataFrame) -> None:
    result = _named(validate_price_demand_direction(sales), "price_increases_reduce_demand")
    assert result.passed
    assert result.observed is not None and result.observed < 0


# --- promotions ------------------------------------------------------------


def test_promotions_lift_sales(sales: pd.DataFrame) -> None:
    result = _named(validate_promotion_uplift(sales), "promotion_increases_sales")
    assert result.passed
    assert result.observed is not None and result.observed > 0.02


def test_uplift_rises_with_discount_depth(sales: pd.DataFrame) -> None:
    """Deeper discounts must lift more.

    Only monotonicity is asserted here. The *shape* of the response - whether
    it saturates - is verified deterministically against the response function
    itself in ``tests/data/test_generators.py``; observed band means mix in
    display, bundle and mechanic effects that vary with depth, so they are the
    wrong instrument for testing curvature.
    """
    results = validate_promotion_uplift(sales)
    result = _named(results, "deeper_discounts_lift_more")
    assert result.passed

    bands = result.detail["uplift_by_depth_band"]
    shallow = bands.get("0-10")
    deep = bands.get("30+")
    if shallow is not None and deep is not None:
        assert deep > shallow


# --- stockouts -------------------------------------------------------------


def test_stockouts_suppress_observed_sales(sales: pd.DataFrame, smoke_latent: pd.DataFrame) -> None:
    results = validate_stockout_censoring(sales, smoke_latent)
    assert _named(results, "stockouts_suppress_observed_sales").passed


def test_latent_demand_survives_the_stockout(
    sales: pd.DataFrame, smoke_latent: pd.DataFrame
) -> None:
    """The distinction the Root Cause agent must be able to make.

    If underlying demand collapsed alongside observed sales, a supply failure
    would be indistinguishable from a demand failure and Scenario D would prove
    nothing.
    """
    results = validate_stockout_censoring(sales, smoke_latent)
    assert _named(results, "latent_demand_holds_during_stockout").passed


def test_no_censoring_when_stock_is_available(
    sales: pd.DataFrame, smoke_latent: pd.DataFrame
) -> None:
    results = validate_stockout_censoring(sales, smoke_latent)
    result = _named(results, "no_censoring_when_in_stock")
    assert result.passed
    assert result.observed is not None and abs(result.observed) < 0.02


# --- cross-price -----------------------------------------------------------


def test_cross_price_signs_match_declared_relationships(
    sales: pd.DataFrame,
    smoke_tables: dict[str, pd.DataFrame],
    smoke_ground_truth: GroundTruth,
) -> None:
    results = validate_cross_price(sales, smoke_tables["pricing"], smoke_ground_truth)
    result = _named(results, "cross_price_signs_agree")

    assert result.passed, f"only {result.observed:.0%} of pairs agreed in sign"
    assert result.sample_size >= 4


def test_controlling_for_own_price_changes_cross_estimates(
    sales: pd.DataFrame,
    smoke_tables: dict[str, pd.DataFrame],
    smoke_ground_truth: GroundTruth,
) -> None:
    """Same-category products share a cost index, so their prices move together.

    Holding the target's own price constant must therefore move the estimate.
    Asserted as a material change rather than a sign flip: the store and month
    fixed effects already absorb much of the shared variation, so a flip happens
    sometimes but is not guaranteed - and a test that demands one would be
    asserting a coincidence rather than a property.
    """
    results = validate_cross_price(sales, smoke_tables["pricing"], smoke_ground_truth)
    detail = _named(results, "cross_price_signs_agree").detail

    comparable = [v for v in detail.values() if v.get("uncontrolled") is not None]
    assert comparable

    shifted = [v for v in comparable if abs(v["observed"] - v["uncontrolled"]) > 0.05]
    assert shifted, (
        "controlling for own price left every cross-price estimate unchanged, "
        "which would mean the shared cost-index confounding is absent"
    )


# --- competitor ------------------------------------------------------------


def test_competitor_price_rise_lifts_our_demand(
    sales: pd.DataFrame, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    results = validate_competitor_effect(sales, smoke_tables["competitor_pricing"])
    result = _named(results, "competitor_price_raises_our_demand")

    assert result.passed
    assert result.observed is not None and result.observed > 0


def test_competitor_effect_is_confounded_without_control(
    sales: pd.DataFrame, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """Our price and theirs both track the shared cost index.

    Regressing volume on competitor price alone recovers our own price effect
    with the sign flipped - a mistake worth being able to demonstrate.
    """
    results = validate_competitor_effect(sales, smoke_tables["competitor_pricing"])
    detail = _named(results, "competitor_price_raises_our_demand").detail

    assert detail["controlled_for_own_price"] > 0
    assert detail["uncontrolled_naive"] < detail["controlled_for_own_price"]
    assert detail["own_price_coefficient"] < 0


# --- seasonality and regions -----------------------------------------------


def test_festival_periods_show_a_demand_peak(
    sales: pd.DataFrame, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    results = validate_seasonality_and_regions(
        sales, smoke_tables["calendar"], smoke_tables["stores"]
    )
    assert _named(results, "festival_demand_peak").passed


def test_regions_differ_in_demand(
    sales: pd.DataFrame, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    results = validate_seasonality_and_regions(
        sales, smoke_tables["calendar"], smoke_tables["stores"]
    )
    assert _named(results, "regional_variation_present").passed


# --- end to end ------------------------------------------------------------


def test_full_validation_passes(smoke_result: GenerationResult) -> None:
    """The gate the CLI uses: every invariant and every relationship."""
    report = validate_dataset(smoke_result.root, sample_rows=None)

    failed_checks = [r.name for r in report.checks.failures]
    failed_relationships = [r.name for r in report.failed_relationships]

    assert not failed_checks, f"invariant failures: {failed_checks}"
    assert not failed_relationships, f"relationship failures: {failed_relationships}"
    assert report.passed


def test_validation_report_renders(smoke_result: GenerationResult) -> None:
    report = validate_dataset(smoke_result.root, sample_rows=None)
    markdown = report.to_markdown()
    assert "# Dataset Validation Report" in markdown
    assert "own_price_elasticity_recoverable" in markdown
