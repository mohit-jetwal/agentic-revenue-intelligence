"""Scenario injection (brief section 19).

Each scenario must be *identifiable in the data*, not merely registered. A
scenario recorded in the registry but invisible in the sales fact would be worse
than useless: it would make the Step 21 evaluation set assert things that are
not true.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.generation.ground_truth import GroundTruth
from data.generation.pipeline import GenerationResult

pytestmark = pytest.mark.data


def _by_label(scenarios: list[dict], label: str) -> list[dict]:
    return [s for s in scenarios if s["label"] == label]


def test_scenario_registry_is_populated(smoke_ground_truth: GroundTruth) -> None:
    assert smoke_ground_truth.scenarios


def test_all_required_scenarios_are_registered(smoke_ground_truth: GroundTruth) -> None:
    """A-J from brief section 19."""
    labels = {s["label"] for s in smoke_ground_truth.scenarios}
    required = {
        "price_increase",  # A
        "successful_promo",  # B
        "bad_promo",  # C
        "stockout",  # D
        "competitor_price_cut",  # E
        "regional_shock",  # H
        "seasonal_peak",  # I
        "product_launch",  # J
    }
    assert required <= labels, f"missing scenarios: {required - labels}"


def test_scenarios_record_their_window_and_expectation(
    smoke_ground_truth: GroundTruth,
) -> None:
    """The registry seeds Step 21's golden evaluation set, so it must be precise."""
    for scenario in smoke_ground_truth.scenarios:
        assert scenario["scenario_id"]
        assert scenario["description"]
        assert scenario["expected_effect"]


def test_price_increase_scenario_names_substitutes_and_complements(
    smoke_ground_truth: GroundTruth,
) -> None:
    """Scenarios F and G fall out of A automatically via cross-price effects."""
    price_scenarios = _by_label(smoke_ground_truth.scenarios, "price_increase")
    assert price_scenarios
    assert any(s["detail"].get("substitutes") for s in price_scenarios)


def test_price_increase_is_visible_in_the_data(
    smoke_ground_truth: GroundTruth, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """Scenario A: price genuinely higher inside the window than before it."""
    scenarios = _by_label(smoke_ground_truth.scenarios, "price_increase")
    assert scenarios
    scenario = scenarios[0]
    product_id = scenario["product_ids"][0]

    pricing = smoke_tables["pricing"]
    subset = pricing[pricing["product_id"] == product_id].copy()
    subset["date"] = pd.to_datetime(subset["date"])

    start = pd.Timestamp(scenario["start_date"])
    end = pd.Timestamp(scenario["end_date"])
    during = subset[(subset["date"] >= start) & (subset["date"] <= end)]
    before = subset[subset["date"] < start]

    assert not during.empty and not before.empty
    assert float(during["regular_price"].mean()) > float(before["regular_price"].mean())


def test_stockout_scenario_suppresses_observed_sales(
    smoke_ground_truth: GroundTruth,
    smoke_tables: dict[str, pd.DataFrame],
    smoke_latent: pd.DataFrame,
) -> None:
    """Scenario D: observed sales fall while latent demand does not."""
    scenarios = _by_label(smoke_ground_truth.scenarios, "stockout")
    assert scenarios
    scenario = scenarios[0]
    product_id = scenario["product_ids"][0]
    affected_stores = set(scenario["store_ids"])

    latent = smoke_latent[
        (smoke_latent["product_id"] == product_id)
        & (smoke_latent["store_id"].isin(affected_stores))
    ].copy()
    assert not latent.empty

    latent["date"] = pd.to_datetime(latent["date"])
    start = pd.Timestamp(scenario["start_date"])
    end = pd.Timestamp(scenario["end_date"])

    during = latent[(latent["date"] >= start) & (latent["date"] <= end)]
    assert not during.empty

    # Observed below latent during the window: supply, not demand, is binding.
    assert int(during["lost_units"].sum()) > 0
    assert float(during["observed_units"].sum()) < float(during["latent_units"].sum())


def test_regional_shock_is_distribution_driven_not_demand_driven(
    smoke_ground_truth: GroundTruth, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """Scenario H: the decline comes from lost listings, not weaker shoppers.

    This is the distinction the Root Cause agent must make in Step 17. Both look
    like "North is down" in a summary report; only one is a demand problem.
    """
    scenarios = _by_label(smoke_ground_truth.scenarios, "regional_shock")
    if not scenarios:
        pytest.skip("regional shock not injected in this profile")

    scenario = scenarios[0]
    assert scenario["region"]
    assert scenario["detail"]["listings_lost"] > 0
    # Distribution loss must dominate the residual demand softening.
    assert abs(scenario["detail"]["residual_demand_effect"]) < 0.15


def test_bad_promotion_has_worse_margin_than_the_good_one(
    smoke_ground_truth: GroundTruth, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """Scenarios B and C must be genuinely distinguishable.

    Both lift volume. The difference is that C's discount is deep enough that
    incremental margin cannot cover it - so Step 6 should measure positive
    uplift on both, while Step 7 funds only one.
    """
    good = _by_label(smoke_ground_truth.scenarios, "successful_promo")
    bad = _by_label(smoke_ground_truth.scenarios, "bad_promo")
    assert good and bad
    assert bad[0]["magnitude"] > good[0]["magnitude"]

    sales = smoke_tables["sales_daily"]

    def margin_during(scenario: dict) -> float:
        product_id = scenario["product_ids"][0]
        frame = sales[sales["product_id"] == product_id].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        window = frame[
            (frame["date"] >= pd.Timestamp(scenario["start_date"]))
            & (frame["date"] <= pd.Timestamp(scenario["end_date"]))
        ]
        revenue = float(window["revenue"].sum())
        return float(window["gross_profit"].sum()) / revenue if revenue > 0 else 0.0

    assert margin_during(bad[0]) < margin_during(good[0])


def test_competitor_cut_is_recorded_with_a_negative_magnitude(
    smoke_ground_truth: GroundTruth,
) -> None:
    scenarios = _by_label(smoke_ground_truth.scenarios, "competitor_price_cut")
    assert scenarios
    assert scenarios[0]["magnitude"] < 0


def test_product_launch_ramps_gradually(
    smoke_ground_truth: GroundTruth, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """Scenario J: a launch builds distribution rather than starting at full rate."""
    scenarios = _by_label(smoke_ground_truth.scenarios, "product_launch")
    if not scenarios or not scenarios[0]["product_ids"]:
        pytest.skip("no mid-history launches in this profile")

    product_id = scenarios[0]["product_ids"][0]
    sales = smoke_tables["sales_daily"]
    products = smoke_tables["products"]

    launch_date = pd.Timestamp(
        products.loc[products["product_id"] == product_id, "launch_date"].iloc[0]
    )
    frame = sales[sales["product_id"] == product_id].copy()
    frame["date"] = pd.to_datetime(frame["date"])

    before = frame[frame["date"] < launch_date]
    early = frame[
        (frame["date"] >= launch_date) & (frame["date"] < launch_date + pd.Timedelta(days=30))
    ]
    later = frame[
        (frame["date"] >= launch_date + pd.Timedelta(days=150))
        & (frame["date"] < launch_date + pd.Timedelta(days=240))
    ]

    assert int(before["units"].sum()) == 0, "no sales may occur before launch"
    if not early.empty and not later.empty:
        assert float(later["units"].mean()) > float(early["units"].mean())


def test_scenario_products_are_well_observed(
    smoke_ground_truth: GroundTruth, smoke_tables: dict[str, pd.DataFrame]
) -> None:
    """A scenario on a thinly stocked SKU would be invisible under noise."""
    sales = smoke_tables["sales_daily"]
    counts = sales.groupby("product_id").size()
    median = float(counts.median())

    for scenario in smoke_ground_truth.scenarios:
        for product_id in scenario.get("product_ids", []):
            if product_id in counts.index:
                assert counts[product_id] >= median * 0.5


def test_scenario_windows_leave_a_pre_period(
    smoke_ground_truth: GroundTruth, smoke_result: GenerationResult
) -> None:
    """Every scenario needs history before it to be measured against."""
    start = pd.Timestamp(smoke_result.config.time.start_date)
    for scenario in smoke_ground_truth.scenarios:
        if scenario.get("start_date") and scenario["label"] != "product_launch":
            assert pd.Timestamp(scenario["start_date"]) > start + pd.Timedelta(days=30)
