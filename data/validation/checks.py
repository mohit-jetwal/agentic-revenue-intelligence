"""Business-invariant checks (brief section 24).

A small typed framework rather than Pandera or Great Expectations. The
trade-off, stated honestly: those libraries are the more conventional answer and
would look more familiar in review. But most checks here are *business*
invariants - ``opening + received - sold = closing``, ``revenue ~= units x
price`` - which schema libraries express awkwardly, and both bring a large
dependency for what amounts to a hundred lines of arithmetic. If the check set
grows past simple predicates, Pandera is the right next step.

Every check returns a :class:`CheckResult` rather than raising, so one failure
does not hide the twenty checks behind it. Severity decides whether the pipeline
exits non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CheckResult:
    """Outcome of one invariant check."""

    name: str
    table: str
    passed: bool
    severity: Severity
    message: str
    observed: float | None = None
    threshold: float | None = None
    failing_rows: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity is Severity.ERROR else "WARN"


class CheckSuite:
    """Collects check results and reports whether the dataset is usable."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def check(
        self,
        name: str,
        table: str,
        condition: bool,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        observed: float | None = None,
        threshold: float | None = None,
        failing_rows: int = 0,
    ) -> CheckResult:
        return self.add(
            CheckResult(
                name=name,
                table=table,
                passed=bool(condition),
                severity=severity,
                message=message,
                observed=observed,
                threshold=threshold,
                failing_rows=failing_rows,
            )
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": len(self.failures),
            "warnings": len(self.warnings),
        }


# ---------------------------------------------------------------------------
# Table checks
# ---------------------------------------------------------------------------


def check_sales(sales: pd.DataFrame, suite: CheckSuite) -> None:
    """Sales fact invariants."""
    table = "sales_daily"

    negative = int((sales["units"] < 0).sum())
    suite.check(
        "units_non_negative",
        table,
        negative == 0,
        "units must never be negative in the gold layer",
        observed=negative,
        threshold=0,
        failing_rows=negative,
    )

    bad_price = int((sales["selling_price"] <= 0).sum())
    suite.check(
        "selling_price_positive",
        table,
        bad_price == 0,
        "selling_price must be strictly positive",
        observed=bad_price,
        threshold=0,
        failing_rows=bad_price,
    )

    bad_discount = int(
        ((sales["discount_percentage"] < 0) | (sales["discount_percentage"] > 100)).sum()
    )
    suite.check(
        "discount_in_range",
        table,
        bad_discount == 0,
        "discount_percentage must fall within [0, 100]",
        observed=bad_discount,
        threshold=0,
        failing_rows=bad_discount,
    )

    over_list = int((sales["selling_price"] > sales["regular_price"] + 0.01).sum())
    suite.check(
        "selling_price_not_above_regular",
        table,
        over_list == 0,
        "selling_price must not exceed regular_price",
        observed=over_list,
        threshold=0,
        failing_rows=over_list,
    )

    # Revenue identity. Tolerance rather than equality because both sides are
    # rounded to paise independently; an exact test would fail on rounding alone.
    expected_revenue = sales["units"] * sales["selling_price"]
    revenue_gap = (sales["revenue"] - expected_revenue).abs()
    revenue_breaks = int((revenue_gap > 0.05).sum())
    suite.check(
        "revenue_identity",
        table,
        revenue_breaks == 0,
        "revenue must equal units x selling_price",
        observed=revenue_breaks,
        threshold=0,
        failing_rows=revenue_breaks,
    )

    profit_gap = (sales["gross_profit"] - (sales["revenue"] - sales["cost"])).abs()
    profit_breaks = int((profit_gap > 0.05).sum())
    suite.check(
        "gross_profit_identity",
        table,
        profit_breaks == 0,
        "gross_profit must equal revenue minus cost",
        observed=profit_breaks,
        threshold=0,
        failing_rows=profit_breaks,
    )

    nulls = int(sales[["product_id", "store_id", "date"]].isna().sum().sum())
    suite.check(
        "keys_not_null",
        table,
        nulls == 0,
        "product_id, store_id and date must never be null in gold",
        observed=nulls,
        threshold=0,
        failing_rows=nulls,
    )

    # Zero-sales days are expected for slow movers, but an implausibly high
    # share suggests demand was calibrated too low to model.
    zero_share = float((sales["units"] == 0).mean())
    suite.check(
        "zero_sales_share_reasonable",
        table,
        zero_share < 0.75,
        "share of zero-unit days should stay below 75%; higher suggests base "
        "demand is calibrated too low for models to learn from",
        severity=Severity.WARNING,
        observed=round(zero_share, 4),
        threshold=0.75,
    )


def check_inventory(inventory: pd.DataFrame, suite: CheckSuite) -> None:
    """Inventory reconciliation - the identity from brief section 17."""
    table = "inventory"

    expected = (
        inventory["opening_inventory"] + inventory["received_units"] - inventory["sold_units"]
    )
    breaks = int((inventory["closing_inventory"] - expected).abs().gt(0).sum())
    suite.check(
        "inventory_reconciliation",
        table,
        breaks == 0,
        "opening + received - sold must equal closing",
        observed=breaks,
        threshold=0,
        failing_rows=breaks,
    )

    negative = int((inventory["closing_inventory"] < 0).sum())
    suite.check(
        "inventory_non_negative",
        table,
        negative == 0,
        "closing_inventory must never be negative",
        observed=negative,
        threshold=0,
        failing_rows=negative,
    )

    oversold = int(
        (
            inventory["sold_units"] > inventory["opening_inventory"] + inventory["received_units"]
        ).sum()
    )
    suite.check(
        "cannot_sell_more_than_available",
        table,
        oversold == 0,
        "sold_units must not exceed opening + received",
        observed=oversold,
        threshold=0,
        failing_rows=oversold,
    )

    stockout_rate = float(inventory["stockout_flag"].mean())
    suite.check(
        "stockouts_present",
        table,
        0.0 < stockout_rate < 0.35,
        "stockouts must occur but stay below 35% of rows, otherwise censoring "
        "dominates and demand signal is lost",
        severity=Severity.WARNING,
        observed=round(stockout_rate, 4),
        threshold=0.35,
    )


def check_promotions(promotions: pd.DataFrame, suite: CheckSuite) -> None:
    """Promotion event invariants."""
    table = "promotions"
    if promotions.empty:
        suite.check("promotions_present", table, False, "no promotions generated")
        return

    starts = pd.to_datetime(promotions["start_date"])
    ends = pd.to_datetime(promotions["end_date"])
    inverted = int((starts > ends).sum())
    suite.check(
        "promotion_dates_ordered",
        table,
        inverted == 0,
        "start_date must not fall after end_date",
        observed=inverted,
        threshold=0,
        failing_rows=inverted,
    )

    bad_discount = int(
        (
            (promotions["discount_percentage"] <= 0) | (promotions["discount_percentage"] >= 100)
        ).sum()
    )
    suite.check(
        "promotion_discount_in_range",
        table,
        bad_discount == 0,
        "promotion discount must fall within (0, 100)",
        observed=bad_discount,
        threshold=0,
        failing_rows=bad_discount,
    )

    duplicates = int(promotions["promotion_id"].duplicated().sum())
    suite.check(
        "promotion_id_unique",
        table,
        duplicates == 0,
        "promotion_id must be unique in gold",
        observed=duplicates,
        threshold=0,
        failing_rows=duplicates,
    )

    if "promotion_spend" in promotions:
        negative = int((promotions["promotion_spend"] < 0).sum())
        suite.check(
            "promotion_spend_non_negative",
            table,
            negative == 0,
            "promotion_spend must not be negative",
            observed=negative,
            threshold=0,
            failing_rows=negative,
        )

    # Effectiveness must vary, or Step 7 has nothing to allocate between.
    variety = int(promotions["promotion_type"].nunique())
    suite.check(
        "promotion_types_varied",
        table,
        variety >= 3,
        "at least three promotion mechanics must appear",
        observed=variety,
        threshold=3,
    )


def check_referential_integrity(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    promotions: pd.DataFrame,
    suite: CheckSuite,
) -> None:
    """Foreign keys in gold must resolve."""
    known_products = set(products["product_id"])
    orphan_products = int((~sales["product_id"].isin(known_products)).sum())
    suite.check(
        "sales_product_fk",
        "sales_daily",
        orphan_products == 0,
        "every sales.product_id must exist in products",
        observed=orphan_products,
        threshold=0,
        failing_rows=orphan_products,
    )

    known_stores = set(stores["store_id"])
    orphan_stores = int((~sales["store_id"].isin(known_stores)).sum())
    suite.check(
        "sales_store_fk",
        "sales_daily",
        orphan_stores == 0,
        "every sales.store_id must exist in stores",
        observed=orphan_stores,
        threshold=0,
        failing_rows=orphan_stores,
    )

    if not promotions.empty:
        known_promotions = set(promotions["promotion_id"])
        referenced = sales["promotion_id"].dropna()
        orphan_promotions = int((~referenced.isin(known_promotions)).sum())
        suite.check(
            "sales_promotion_fk",
            "sales_daily",
            orphan_promotions == 0,
            "every non-null sales.promotion_id must exist in promotions",
            observed=orphan_promotions,
            threshold=0,
            failing_rows=orphan_promotions,
        )


def check_dimensions(
    products: pd.DataFrame,
    stores: pd.DataFrame,
    customers: pd.DataFrame,
    calendar: pd.DataFrame,
    relationships: pd.DataFrame,
    suite: CheckSuite,
) -> None:
    """Dimension uniqueness, coverage and leakage."""
    for name, frame, key in (
        ("products", products, "product_id"),
        ("stores", stores, "store_id"),
        ("customers", customers, "customer_id"),
    ):
        duplicates = int(frame[key].duplicated().sum())
        suite.check(
            f"{name}_key_unique",
            name,
            duplicates == 0,
            f"{key} must be unique",
            observed=duplicates,
            threshold=0,
            failing_rows=duplicates,
        )

    gaps = int(pd.to_datetime(calendar["date"]).diff().dt.days.dropna().ne(1).sum())
    suite.check(
        "calendar_contiguous",
        "calendar",
        gaps == 0,
        "calendar must contain every day with no gaps",
        observed=gaps,
        threshold=0,
        failing_rows=gaps,
    )

    festivals = int(calendar["festival_flag"].sum())
    suite.check(
        "calendar_has_festivals",
        "calendar",
        festivals > 0,
        "festival days must be present for seasonal scenarios to exist",
        observed=festivals,
        threshold=1,
    )

    # Latent simulation parameters must never reach the analytical tables.
    # This is the leakage guard from brief section 33: exposing true elasticity
    # or base demand would hand future models the answer.
    for name, frame in (("products", products), ("stores", stores)):
        leaked = [c for c in frame.columns if c.startswith("_")]
        suite.check(
            f"{name}_no_latent_columns",
            name,
            not leaked,
            f"latent simulation attributes must not be published: {leaked}",
            observed=len(leaked),
            threshold=0,
        )

    if not relationships.empty:
        substitutes = relationships[relationships["relationship_type"] == "substitute"]
        complements = relationships[relationships["relationship_type"] == "complement"]
        suite.check(
            "substitutes_positive",
            "product_relationships",
            bool((substitutes["cross_elasticity"] > 0).all()),
            "substitute cross-elasticities must be positive",
            observed=float(substitutes["cross_elasticity"].min()) if len(substitutes) else 0.0,
        )
        suite.check(
            "complements_negative",
            "product_relationships",
            bool((complements["cross_elasticity"] < 0).all()),
            "complement cross-elasticities must be negative",
            observed=float(complements["cross_elasticity"].max()) if len(complements) else 0.0,
        )


def check_pricing(pricing: pd.DataFrame, suite: CheckSuite) -> None:
    """Price path invariants."""
    table = "pricing"

    non_positive = int((pricing["regular_price"] <= 0).sum())
    suite.check(
        "regular_price_positive",
        table,
        non_positive == 0,
        "regular_price must be strictly positive",
        observed=non_positive,
        threshold=0,
        failing_rows=non_positive,
    )

    # Price must actually vary, or there is no signal to estimate elasticity
    # from. This is the check that would catch a generator regression silently
    # flattening the price paths.
    variation = pricing.groupby("product_id")["regular_price"].nunique()
    static_share = float((variation <= 1).mean())
    suite.check(
        "prices_vary",
        table,
        static_share < 0.10,
        "at least 90% of products must show price variation, otherwise "
        "elasticity is not identified",
        observed=round(static_share, 4),
        threshold=0.10,
    )

    if "price_change_reason" in pricing:
        reasons = set(pricing.loc[pricing["price_change_flag"], "price_change_reason"].unique())
        suite.check(
            "randomised_price_tests_present",
            table,
            "randomised_test" in reasons,
            "exogenous randomised price tests must exist; they are the clean "
            "identification subset for elasticity",
            observed=len(reasons),
        )


def run_all_checks(tables: dict[str, pd.DataFrame]) -> CheckSuite:
    """Run every invariant check against a loaded gold dataset."""
    suite = CheckSuite()

    sales = tables.get("sales_daily")
    if sales is not None and not sales.empty:
        check_sales(sales, suite)

    inventory = tables.get("inventory")
    if inventory is not None and not inventory.empty:
        check_inventory(inventory, suite)

    promotions = tables.get("promotions", pd.DataFrame())
    if not promotions.empty:
        check_promotions(promotions, suite)

    pricing = tables.get("pricing")
    if pricing is not None and not pricing.empty:
        check_pricing(pricing, suite)

    products = tables.get("products")
    stores = tables.get("stores")
    customers = tables.get("customers")
    calendar = tables.get("calendar")
    relationships = tables.get("product_relationships", pd.DataFrame())
    if (
        products is not None
        and stores is not None
        and customers is not None
        and calendar is not None
    ):
        check_dimensions(products, stores, customers, calendar, relationships, suite)
        if sales is not None and not sales.empty:
            check_referential_integrity(sales, products, stores, promotions, suite)

    return suite


def numeric_or_none(value: Any) -> float | None:
    """Coerce a numpy scalar to a plain float for JSON serialisation."""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None
