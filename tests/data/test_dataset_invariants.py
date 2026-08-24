"""Invariants and reproducibility of the generated dataset.

Runs against the session-scoped smoke dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.generation.ground_truth import GroundTruth
from data.generation.pipeline import GenerationResult
from data.validation.checks import Severity, run_all_checks

pytestmark = pytest.mark.data


# --- business invariants ---------------------------------------------------


def test_all_error_severity_checks_pass(smoke_tables: dict[str, pd.DataFrame]) -> None:
    suite = run_all_checks(smoke_tables)
    failures = [f"{r.name}: {r.message}" for r in suite.failures]
    assert not failures, "invariant failures: " + "; ".join(failures)


def test_check_suite_covers_every_core_table(smoke_tables: dict[str, pd.DataFrame]) -> None:
    suite = run_all_checks(smoke_tables)
    covered = {r.table for r in suite.results}
    assert {"sales_daily", "inventory", "promotions", "pricing", "products"} <= covered


def test_inventory_reconciles_exactly(inventory: pd.DataFrame) -> None:
    """opening + received - sold = closing, with no tolerance."""
    expected = (
        inventory["opening_inventory"] + inventory["received_units"] - inventory["sold_units"]
    )
    assert (inventory["closing_inventory"] == expected).all()


def test_revenue_identity_holds(sales: pd.DataFrame) -> None:
    gap = (sales["revenue"] - sales["units"] * sales["selling_price"]).abs()
    assert float(gap.max()) < 0.05


def test_gross_profit_identity_holds(sales: pd.DataFrame) -> None:
    gap = (sales["gross_profit"] - (sales["revenue"] - sales["cost"])).abs()
    assert float(gap.max()) < 0.05


def test_no_negative_units_in_gold(sales: pd.DataFrame) -> None:
    assert int((sales["units"] < 0).sum()) == 0


def test_selling_price_never_exceeds_regular(sales: pd.DataFrame) -> None:
    assert int((sales["selling_price"] > sales["regular_price"] + 0.01).sum()) == 0


def test_cannot_sell_more_than_available(inventory: pd.DataFrame) -> None:
    available = inventory["opening_inventory"] + inventory["received_units"]
    assert int((inventory["sold_units"] > available).sum()) == 0


# --- leakage ---------------------------------------------------------------


def test_gold_exposes_no_latent_parameters(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """Brief section 33: no future-looking or hidden-parameter leakage.

    Publishing ``_base_demand`` or the true elasticity would hand every future
    model the answer it is supposed to estimate.
    """
    for name, frame in smoke_tables.items():
        leaked = [c for c in frame.columns if c.startswith("_")]
        assert not leaked, f"table {name} exposes latent columns: {leaked}"


def test_gold_contains_no_ground_truth_tables(
    smoke_result: GenerationResult,
) -> None:
    gold_entries = {p.name for p in (smoke_result.root / "gold").iterdir()}
    assert "latent_demand" not in gold_entries
    assert not any("elasticity" in name for name in gold_entries)


def test_ground_truth_is_stored_separately(smoke_result: GenerationResult) -> None:
    directory = smoke_result.root / "ground_truth"
    assert directory.is_dir()
    for filename in (
        "elasticity.json",
        "cross_elasticity.json",
        "promotion_uplift.json",
        "scenario_config.json",
    ):
        assert (directory / filename).is_file()


# --- referential integrity -------------------------------------------------


def test_sales_keys_resolve(
    sales: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame
) -> None:
    assert set(sales["product_id"]) <= set(products["product_id"])
    assert set(sales["store_id"]) <= set(stores["store_id"])


def test_promotion_references_resolve(sales: pd.DataFrame, promotions: pd.DataFrame) -> None:
    referenced = set(sales["promotion_id"].dropna())
    assert referenced <= set(promotions["promotion_id"])


def test_transactions_reconcile_to_the_panel(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """Sampled transactions must sum back to the panel rows they came from."""
    transactions = smoke_tables["sales_transactions"]
    sales = smoke_tables["sales_daily"]

    keys = ["date", "product_id", "store_id"]
    by_key = transactions.groupby(keys, observed=True)["units"].sum().reset_index()
    merged = by_key.merge(sales[[*keys, "units"]], on=keys, suffixes=("_tx", "_panel"))

    assert not merged.empty
    assert (merged["units_tx"] == merged["units_panel"]).all()


# --- reproducibility -------------------------------------------------------


def test_same_seed_produces_identical_output(
    smoke_result: GenerationResult, second_run: GenerationResult
) -> None:
    """The acceptance criterion from brief section 22.

    Hashes file contents, not row counts: a shape-preserving value drift is
    exactly the regression a count-based check would miss.
    """
    assert smoke_result.gold_hash == second_run.gold_hash
    assert smoke_result.row_counts == second_run.row_counts


def test_ground_truth_is_reproducible(
    smoke_result: GenerationResult, second_run: GenerationResult
) -> None:
    first = GroundTruth.load(smoke_result.root)
    second = GroundTruth.load(second_run.root)
    assert first.own_elasticity == second.own_elasticity
    assert first.cross_elasticity == second.cross_elasticity


def test_manifest_records_provenance(smoke_result: GenerationResult) -> None:
    manifest = json.loads((smoke_result.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["dataset_version"]
    assert manifest["config_hash"]
    assert manifest["scenario_version"]
    assert manifest["total_rows"] > 0


# --- shape and coverage ----------------------------------------------------


def test_generated_row_counts_are_plausible(
    smoke_result: GenerationResult, smoke_config: object
) -> None:
    counts = smoke_result.row_counts
    assert counts["sales_daily"] > 100_000
    # The panel, pricing and inventory share one grain, so they must match.
    assert counts["sales_daily"] == counts["pricing"] == counts["inventory"]


def test_full_date_range_is_covered(sales: pd.DataFrame, smoke_config: object) -> None:
    dates = pd.to_datetime(sales["date"])
    assert dates.min().date() == smoke_config.time.start_date  # type: ignore[attr-defined]
    assert dates.max().date() == smoke_config.time.end_date  # type: ignore[attr-defined]


def test_weekly_and_monthly_aggregates_reconcile(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """Aggregates are precomputed; they must agree with the daily fact."""
    daily_units = int(smoke_tables["sales_daily"]["units"].sum())
    weekly_units = int(smoke_tables["sales_weekly"]["units"].sum())
    monthly_units = int(smoke_tables["sales_monthly"]["units"].sum())
    assert daily_units == weekly_units == monthly_units


def test_price_variation_exists(smoke_tables: dict[str, pd.DataFrame]) -> None:
    """Without price variation, elasticity is not identified at all."""
    pricing = smoke_tables["pricing"]
    per_product = pricing.groupby("product_id")["regular_price"].nunique()
    assert float((per_product > 1).mean()) > 0.95


def test_randomised_price_tests_exist(smoke_tables: dict[str, pd.DataFrame]) -> None:
    """The exogenous subset that makes elasticity cleanly identifiable."""
    pricing = smoke_tables["pricing"]
    changes = pricing[pricing["price_change_flag"]]
    assert "randomised_test" in set(changes["price_change_reason"])


def test_stockouts_occur_but_do_not_dominate(sales: pd.DataFrame) -> None:
    rate = float(sales["stockout_flag"].mean())
    assert 0.0 < rate < 0.35


def test_promotions_span_multiple_mechanics(promotions: pd.DataFrame) -> None:
    assert promotions["promotion_type"].nunique() >= 3


def test_promotion_roi_varies_including_poor_performers(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """Step 7 needs bad promotions to have anything to allocate away from."""
    trade = smoke_tables["trade_promotions"]
    assert not trade.empty
    assert float(trade["roi"].min()) < 1.0
    assert float(trade["roi"].max()) > 1.0


# --- bronze layer ----------------------------------------------------------


def test_bronze_contains_injected_defects(smoke_result: GenerationResult) -> None:
    """The quality framework needs real defects, or its checks are theatre."""
    assert smoke_result.corruption
    injected = {issue for issues in smoke_result.corruption.values() for issue in issues}
    assert {"missing_product_id", "invalid_price"} <= injected


def test_bronze_sales_are_dirtier_than_gold(smoke_result: GenerationResult) -> None:
    bronze = pd.concat(
        (
            pd.read_parquet(p)
            for p in (smoke_result.root / "bronze" / "sales_daily").rglob("*.parquet")
        ),
        ignore_index=True,
    )
    assert int(bronze["product_id"].isna().sum()) > 0
    assert int((bronze["selling_price"] <= 0).sum()) > 0


def test_gold_remains_clean_despite_bronze_corruption(sales: pd.DataFrame) -> None:
    """The whole point of the two-layer split."""
    assert int(sales["product_id"].isna().sum()) == 0
    assert int((sales["selling_price"] <= 0).sum()) == 0
    assert int((sales["units"] < 0).sum()) == 0


def test_warnings_do_not_fail_the_suite(smoke_tables: dict[str, pd.DataFrame]) -> None:
    suite = run_all_checks(smoke_tables)
    assert all(r.severity is not Severity.ERROR for r in suite.warnings)
    assert suite.passed


def test_no_nan_in_core_numeric_columns(sales: pd.DataFrame) -> None:
    for column in ("units", "revenue", "cost", "gross_profit", "selling_price"):
        assert not sales[column].isna().any(), f"{column} contains NaN"


def test_units_are_integral(sales: pd.DataFrame) -> None:
    """Sales are counts; a fractional unit would betray a Gaussian draw."""
    units = sales["units"].to_numpy()
    assert np.all(units == np.floor(units))
