"""Data contracts for every gold table.

One :class:`~data.contracts.base.DataContract` per table, implementing the
validation rules in brief sections 6-9 and extending the same treatment to the
remaining tables.

Two conventions worth knowing when reading these:

* Contracts describe the **gold** layer, which the generator guarantees is
  clean. Bronze deliberately violates them - that is what makes the
  quality framework's checks meaningful rather than decorative.
* ``strict=False`` throughout, so a frame carrying extra columns still passes.
  Callers routinely project a subset or join extra context on, and a contract
  that rejected that would be fought rather than used.
"""

from __future__ import annotations

import pandera.pandas as pa

from app.schemas.domain import Channel, PromotionType, RelationshipType
from data.contracts.base import (
    DataContract,
    category_column,
    count_column,
    date_column,
    flag_column,
    id_column,
    money_column,
    percentage_column,
)

_CHANNELS = [c.value for c in Channel]
_PROMOTION_TYPES = [p.value for p in PromotionType]
_RELATIONSHIP_TYPES = [r.value for r in RelationshipType]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

SALES_CONTRACT = DataContract(
    name="sales_daily",
    description=(
        "Daily sales at product x store x date. `units` is what the till "
        "recorded, not what customers wanted - during a stockout the two differ."
    ),
    primary_key=("date", "product_id", "store_id"),
    foreign_keys={"product_id": "products", "store_id": "stores"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "date": date_column("Calendar day"),
            "product_id": id_column("FK -> products"),
            "store_id": id_column("FK -> stores"),
            "units": count_column("Observed units sold, censored by availability"),
            "regular_price": money_column("Undiscounted shelf price", positive=True),
            "selling_price": money_column("Price actually paid", positive=True),
            "discount_percentage": percentage_column("Discount depth, 0-100"),
            "revenue": money_column("units x selling_price"),
            "cost": money_column("units x unit_cost"),
            "gross_profit": pa.Column(
                float,
                coerce=True,
                description="revenue - cost. May be negative on a deep promotion.",
            ),
            "promotion_id": id_column("FK -> promotions; null when not promoted", nullable=True),
            "promotion_flag": flag_column("Convenience flag for promotion_id present"),
            "inventory_available": count_column("Opening + received that day"),
            "stockout_flag": flag_column("Demand exceeded availability"),
            "channel": category_column("Store channel", allowed=_CHANNELS),
        },
        checks=[
            # Cross-column rules Pandera can express as frame-level checks.
            # The revenue identity is the one that catches a broken join or a
            # partial write, and it is cheap enough to run whenever validation
            # is on at all.
            pa.Check(
                lambda df: (df["revenue"] - df["units"] * df["selling_price"]).abs() < 0.05,
                name="revenue_identity",
                error="revenue must equal units x selling_price",
            ),
            pa.Check(
                lambda df: (df["gross_profit"] - (df["revenue"] - df["cost"])).abs() < 0.05,
                name="gross_profit_identity",
                error="gross_profit must equal revenue - cost",
            ),
            pa.Check(
                lambda df: df["selling_price"] <= df["regular_price"] + 0.01,
                name="selling_price_not_above_regular",
                error="selling_price must not exceed regular_price",
            ),
        ],
    ),
)


PRICING_CONTRACT = DataContract(
    name="pricing",
    description="Daily price path per product x store, with change provenance.",
    primary_key=("date", "product_id", "store_id"),
    foreign_keys={"product_id": "products", "store_id": "stores"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "date": date_column("Calendar day"),
            "product_id": id_column("FK -> products"),
            "store_id": id_column("FK -> stores"),
            "regular_price": money_column("Shelf price", positive=True),
            "selling_price": money_column("After promotional discount", positive=True),
            "discount_percentage": percentage_column("Discount depth, 0-100"),
            "price_change_flag": flag_column("True on days the regular price moved"),
            "price_change_reason": category_column(
                "Why the price moved",
                allowed=["none", "scheduled", "cost_passthrough", "randomised_test"],
            ),
        },
        checks=[
            pa.Check(
                lambda df: df["selling_price"] <= df["regular_price"] + 0.01,
                name="selling_price_not_above_regular",
                error="selling_price must not exceed regular_price",
            ),
        ],
    ),
)


INVENTORY_CONTRACT = DataContract(
    name="inventory",
    description="Daily stock position. The reconciliation identity holds exactly.",
    primary_key=("date", "product_id", "store_id"),
    foreign_keys={"product_id": "products", "store_id": "stores"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "date": date_column("Calendar day"),
            "product_id": id_column("FK -> products"),
            "store_id": id_column("FK -> stores"),
            "opening_inventory": count_column("Stock at start of day"),
            "received_units": count_column("Deliveries landed"),
            "sold_units": count_column("Units sold; equals sales_daily.units"),
            "closing_inventory": count_column("opening + received - sold"),
            "inventory_days": pa.Column(
                float,
                nullable=True,
                coerce=True,
                description="Days of cover; null when there were no sales to divide by",
            ),
            "stockout_flag": flag_column("Demand exceeded availability"),
        },
        checks=[
            # Brief section 9. Stated with zero tolerance because the generator
            # constructs closing arithmetically - a mismatch is a bug, not
            # rounding. A tolerance here would hide exactly what it should catch.
            pa.Check(
                lambda df: (
                    df["closing_inventory"]
                    == df["opening_inventory"] + df["received_units"] - df["sold_units"]
                ),
                name="inventory_reconciliation",
                error="opening + received - sold must equal closing",
            ),
            pa.Check(
                lambda df: df["sold_units"] <= df["opening_inventory"] + df["received_units"],
                name="cannot_sell_more_than_available",
                error="sold_units must not exceed opening + received",
            ),
        ],
    ),
)


PROMOTIONS_CONTRACT = DataContract(
    name="promotions",
    description="Promotion events. One row per product x store x window.",
    primary_key=("promotion_id",),
    foreign_keys={"product_id": "products", "store_id": "stores"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "promotion_id": id_column("Unique event id"),
            "product_id": id_column("FK -> products"),
            "store_id": id_column("FK -> stores"),
            "promotion_type": category_column("Mechanic", allowed=_PROMOTION_TYPES),
            "start_date": date_column("First day, inclusive"),
            "end_date": date_column("Last day, inclusive"),
            "duration_days": pa.Column(int, coerce=True, checks=[pa.Check.gt(0)]),
            "discount_percentage": pa.Column(
                float,
                coerce=True,
                description="Depth; strictly inside (0, 100) for a real promotion",
                checks=[pa.Check.gt(0), pa.Check.lt(100)],
            ),
            "display_flag": flag_column("In-store display support"),
            "bundle_flag": flag_column("Bundled mechanic"),
            "promotion_channel": category_column(
                "Where it ran", allowed=["In-store", "Digital", "Both"]
            ),
            "region": category_column("Store region"),
            # Nullable because a point-in-time view masks these on forward-dated
            # rows: the schedule is knowable ahead, the realised spend is not.
            "promotion_units": count_column("Units sold during the event", nullable=True),
            "promotion_spend": money_column("Fixed + per-unit cost", nullable=True),
        },
        checks=[
            pa.Check(
                lambda df: df["start_date"] <= df["end_date"],
                name="promotion_dates_ordered",
                error="start_date must not fall after end_date",
            ),
        ],
    ),
)


TRADE_PROMOTIONS_CONTRACT = DataContract(
    name="trade_promotions",
    description="Retailer-level trade plans. ROI below 1.0 occurs by design.",
    primary_key=("trade_promo_id",),
    foreign_keys={"product_id": "products"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "trade_promo_id": id_column("Unique plan id"),
            "retailer": category_column("Retail partner"),
            "product_id": id_column("FK -> products"),
            "region": category_column("Region"),
            "start_date": date_column("Plan start"),
            "end_date": date_column("Plan end"),
            "planned_spend": money_column("Committed spend"),
            "actual_spend": money_column("Realised spend"),
            "expected_uplift": pa.Column(float, coerce=True, description="Fractional, planned"),
            "actual_uplift": pa.Column(float, coerce=True, description="Fractional, realised"),
            "margin": pa.Column(
                float, coerce=True, description="Product margin", checks=[pa.Check.in_range(0, 1)]
            ),
            # Intentionally unbounded below: a value-destroying promotion is a
            # finding, not a data error, and Step 7 needs those to exist.
            "roi": pa.Column(float, coerce=True, description="Incremental profit / actual spend"),
        },
        checks=[
            pa.Check(
                lambda df: df["start_date"] <= df["end_date"],
                name="trade_promo_dates_ordered",
                error="start_date must not fall after end_date",
            ),
        ],
    ),
)


COMPETITOR_PRICING_CONTRACT = DataContract(
    name="competitor_pricing",
    description="Competitor prices at product x competitor x date. Market-level.",
    primary_key=("date", "product_id", "competitor_id"),
    foreign_keys={"product_id": "products"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "date": date_column("Calendar day"),
            "product_id": id_column("FK -> products"),
            "competitor_id": id_column("Competitor identity"),
            "competitor_name": category_column("Competitor display name"),
            "competitor_product_id": id_column("Their SKU code"),
            "competitor_price": money_column("Their list price", positive=True),
            "competitor_discount": percentage_column("Their discount, 0-100"),
            "competitor_promotion_flag": flag_column("On promotion"),
            "competitor_effective_price": money_column(
                "After their discount - use this for analysis", positive=True
            ),
        },
    ),
)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

PRODUCTS_CONTRACT = DataContract(
    name="products",
    description="Product master. No latent simulation parameters may appear here.",
    primary_key=("product_id",),
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "product_id": id_column("Unique product id"),
            "product_name": category_column("Display name"),
            "brand": category_column("Brand"),
            "category": category_column("Category"),
            "subcategory": category_column("Subcategory"),
            "pack_size": category_column("Pack size"),
            "unit_cost": money_column("Cost per unit", positive=True),
            "base_price": money_column("Reference list price", positive=True),
            "launch_date": date_column("First day sellable"),
            "discontinue_date": pa.Column(
                "datetime64[ns]",
                nullable=True,
                coerce=True,
                description="Last day sellable; null when still active",
            ),
            "product_status": category_column(
                "Lifecycle state", allowed=["Active", "Launched", "Discontinued"]
            ),
        },
        checks=[
            pa.Check(
                lambda df: df["base_price"] > df["unit_cost"],
                name="price_above_cost",
                error="base_price must exceed unit_cost",
            ),
        ],
    ),
)


STORES_CONTRACT = DataContract(
    name="stores",
    description="Store master.",
    primary_key=("store_id",),
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "store_id": id_column("Unique store id"),
            "store_name": category_column("Display name"),
            "store_type": category_column("Format", allowed=["Flagship", "Standard"]),
            "channel": category_column("Channel", allowed=_CHANNELS),
            "region": category_column("Region"),
            "state": category_column("State"),
            "city": category_column("City"),
            "store_size_sqft": count_column("Floor area; 0 for e-commerce"),
            "opening_date": date_column("First trading day"),
        },
    ),
)


CUSTOMERS_CONTRACT = DataContract(
    name="customers",
    description="Customer master. Non-PII attributes only.",
    primary_key=("customer_id",),
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "customer_id": id_column("Unique customer id"),
            "segment": category_column("Value/Regular/Premium/Loyal/Occasional"),
            "region": category_column("Region"),
            "loyalty_tier": category_column("Bronze/Silver/Gold/Platinum"),
            "acquisition_channel": category_column("How they were acquired"),
            "customer_since": date_column("Tenure start"),
        },
    ),
)


CALENDAR_CONTRACT = DataContract(
    name="calendar",
    description="Date dimension with holiday, festival and financial calendar.",
    primary_key=("date",),
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "date": date_column("Calendar day"),
            "day_of_week": pa.Column(int, coerce=True, checks=[pa.Check.in_range(0, 6)]),
            "week_of_year": pa.Column(int, coerce=True, checks=[pa.Check.in_range(1, 53)]),
            "month": pa.Column(int, coerce=True, checks=[pa.Check.in_range(1, 12)]),
            "quarter": pa.Column(int, coerce=True, checks=[pa.Check.in_range(1, 4)]),
            "year": pa.Column(int, coerce=True),
            "weekend_flag": flag_column("Saturday or Sunday"),
            "holiday_flag": flag_column("Public holiday"),
            "festival_flag": flag_column("Festival window, including the run-up"),
            "season": category_column("Season"),
            "financial_month": pa.Column(int, coerce=True, checks=[pa.Check.in_range(1, 12)]),
            "financial_quarter": pa.Column(int, coerce=True, checks=[pa.Check.in_range(1, 4)]),
        },
    ),
)


PRODUCT_RELATIONSHIPS_CONTRACT = DataContract(
    name="product_relationships",
    description=(
        "Directed product relationships. cross_elasticity is "
        "d log(demand_a) / d log(price_b): positive => substitutes."
    ),
    primary_key=("product_a", "product_b"),
    foreign_keys={"product_a": "products", "product_b": "products"},
    schema=pa.DataFrameSchema(
        strict=False,
        coerce=True,
        columns={
            "product_a": id_column("Product whose demand responds"),
            "product_b": id_column("Product whose price moves"),
            "relationship_type": category_column("Kind", allowed=_RELATIONSHIP_TYPES),
            "relationship_strength": category_column(
                "Magnitude band", allowed=["strong", "moderate", "weak", "none"]
            ),
            "cross_elasticity": pa.Column(float, coerce=True, description="Signed coefficient"),
        },
        checks=[
            # The sign convention everything downstream depends on. Getting it
            # backwards would invert every assortment conclusion drawn from it.
            pa.Check(
                lambda df: (
                    ~((df["relationship_type"] == "substitute") & (df["cross_elasticity"] <= 0))
                ),
                name="substitutes_positive",
                error="substitute cross-elasticities must be positive",
            ),
            pa.Check(
                lambda df: (
                    ~((df["relationship_type"] == "complement") & (df["cross_elasticity"] >= 0))
                ),
                name="complements_negative",
                error="complement cross-elasticities must be negative",
            ),
            pa.Check(
                lambda df: df["product_a"] != df["product_b"],
                name="no_self_relationship",
                error="a product cannot relate to itself",
            ),
        ],
    ),
)


#: Every contract, keyed by the table it governs.
CONTRACTS: dict[str, DataContract] = {
    contract.name: contract
    for contract in (
        SALES_CONTRACT,
        PRICING_CONTRACT,
        INVENTORY_CONTRACT,
        PROMOTIONS_CONTRACT,
        TRADE_PROMOTIONS_CONTRACT,
        COMPETITOR_PRICING_CONTRACT,
        PRODUCTS_CONTRACT,
        STORES_CONTRACT,
        CUSTOMERS_CONTRACT,
        CALENDAR_CONTRACT,
        PRODUCT_RELATIONSHIPS_CONTRACT,
    )
}


def contract_for(table: str) -> DataContract | None:
    """Contract governing ``table``, or ``None`` if it has none.

    Returns ``None`` rather than raising: derived tables such as
    ``sales_weekly`` are legitimately uncontracted, and a caller validating
    opportunistically should not have to know which is which.
    """
    return CONTRACTS.get(table)
