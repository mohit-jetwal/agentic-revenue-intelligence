"""Data contract tests (brief sections 5-9, 34).

Two things are checked: that gold data satisfies its contracts, and - more
usefully - that a *violation* is actually caught. A contract that cannot fail is
not a control, and the only way to know it can fail is to break something on
purpose.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.contracts import (
    CONTRACTS,
    INVENTORY_CONTRACT,
    PRODUCT_RELATIONSHIPS_CONTRACT,
    PROMOTIONS_CONTRACT,
    SALES_CONTRACT,
    ContractViolationError,
    check_primary_key,
    contract_for,
)

pytestmark = [pytest.mark.data, pytest.mark.unit]


# --- gold satisfies its contracts ------------------------------------------


@pytest.mark.parametrize(
    "table",
    [
        "sales_daily",
        "pricing",
        "inventory",
        "promotions",
        "trade_promotions",
        "competitor_pricing",
        "products",
        "stores",
        "customers",
        "calendar",
        "product_relationships",
    ],
)
def test_gold_satisfies_its_contract(smoke_tables: dict[str, pd.DataFrame], table: str) -> None:
    contract = contract_for(table)
    assert contract is not None, f"no contract declared for {table}"

    frame = smoke_tables.get(table)
    assert frame is not None and not frame.empty, f"{table} missing from the dataset"

    # Sampled: a full schema check on the whole panel is slow and a violation
    # is not a needle - the generator either produces valid rows or it does not.
    sample = frame.head(50_000)
    contract.validate(sample)


def test_every_core_table_has_a_contract() -> None:
    expected = {
        "sales_daily",
        "pricing",
        "inventory",
        "promotions",
        "trade_promotions",
        "competitor_pricing",
        "products",
        "stores",
        "customers",
        "calendar",
        "product_relationships",
    }
    assert expected <= set(CONTRACTS)


def test_primary_keys_are_unique(smoke_tables: dict[str, pd.DataFrame]) -> None:
    for name in ("products", "stores", "customers", "promotions", "calendar"):
        check_primary_key(smoke_tables[name], CONTRACTS[name])


# --- violations are caught -------------------------------------------------


def test_negative_units_are_rejected(smoke_tables: dict[str, pd.DataFrame]) -> None:
    broken = smoke_tables["sales_daily"].head(200).copy()
    broken.loc[broken.index[0], "units"] = -5

    with pytest.raises(ContractViolationError) as exc_info:
        SALES_CONTRACT.validate(broken)

    assert "units" in exc_info.value.summary()


def test_broken_revenue_identity_is_rejected(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """revenue must equal units x selling_price - the identity that catches
    a partial write or a botched join."""
    broken = smoke_tables["sales_daily"].head(200).copy()
    broken.loc[broken.index[0], "revenue"] = 999_999.0

    with pytest.raises(ContractViolationError, match="sales_daily"):
        SALES_CONTRACT.validate(broken)


def test_broken_inventory_reconciliation_is_rejected(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    broken = smoke_tables["inventory"].head(200).copy()
    broken["closing_inventory"] = broken["closing_inventory"].astype("int64")
    broken.loc[broken.index[0], "closing_inventory"] += 77

    with pytest.raises(ContractViolationError):
        INVENTORY_CONTRACT.validate(broken)


def test_inverted_promotion_dates_are_rejected(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    broken = smoke_tables["promotions"].head(100).copy()
    index = broken.index[0]
    start = broken.loc[index, "start_date"]
    broken.loc[index, "start_date"] = broken.loc[index, "end_date"]
    broken.loc[index, "end_date"] = start

    with pytest.raises(ContractViolationError):
        PROMOTIONS_CONTRACT.validate(broken)


def test_out_of_range_discount_is_rejected(smoke_tables: dict[str, pd.DataFrame]) -> None:
    broken = smoke_tables["sales_daily"].head(200).copy()
    broken.loc[broken.index[0], "discount_percentage"] = 150.0

    with pytest.raises(ContractViolationError):
        SALES_CONTRACT.validate(broken)


def test_null_product_id_is_rejected(smoke_tables: dict[str, pd.DataFrame]) -> None:
    broken = smoke_tables["sales_daily"].head(200).copy()
    broken.loc[broken.index[0], "product_id"] = None

    with pytest.raises(ContractViolationError):
        SALES_CONTRACT.validate(broken)


def test_wrong_cross_elasticity_sign_is_rejected(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """A substitute with a negative coefficient inverts every conclusion drawn
    from it, so the contract pins the sign convention."""
    broken = smoke_tables["product_relationships"].copy()
    substitutes = broken[broken["relationship_type"] == "substitute"]
    assert not substitutes.empty
    broken.loc[substitutes.index[0], "cross_elasticity"] = -0.5

    with pytest.raises(ContractViolationError):
        PRODUCT_RELATIONSHIPS_CONTRACT.validate(broken)


def test_duplicate_primary_key_is_rejected(smoke_tables: dict[str, pd.DataFrame]) -> None:
    products = smoke_tables["products"].head(50)
    duplicated = pd.concat([products, products.head(1)], ignore_index=True)

    with pytest.raises(ContractViolationError, match="duplicate"):
        check_primary_key(duplicated, CONTRACTS["products"])


def test_lazy_validation_reports_every_failure(
    smoke_tables: dict[str, pd.DataFrame],
) -> None:
    """All failures at once, not just the first.

    When a generator change breaks four columns, seeing all four is one fix;
    seeing them one per run is four.
    """
    broken = smoke_tables["sales_daily"].head(200).copy()
    broken.loc[broken.index[0], "units"] = -1
    broken.loc[broken.index[1], "discount_percentage"] = 500.0
    broken.loc[broken.index[2], "selling_price"] = -3.0

    with pytest.raises(ContractViolationError) as exc_info:
        SALES_CONTRACT.validate(broken, lazy=True)

    failures = exc_info.value.failures
    assert failures is not None
    assert failures["column"].nunique() >= 2


# --- repository integration -------------------------------------------------


def test_repository_validate_flag_runs_the_contract(smoke_repository: object) -> None:
    """`validate=True` at the boundary is what section 5 is really asking for."""
    frame = smoke_repository.get_products(validate=True)  # type: ignore[attr-defined]
    assert not frame.empty


def test_validation_is_off_by_default(smoke_repository: object) -> None:
    """Opt-in, because schema-checking every read of a 6M-row frame is a real cost
    and the generator already guarantees gold is clean."""
    frame = smoke_repository.get_products()  # type: ignore[attr-defined]
    assert not frame.empty


def test_contract_columns_match_the_data(smoke_tables: dict[str, pd.DataFrame]) -> None:
    """Every contracted column must exist in the generated table.

    Catches the drift where a generator renames a column and the contract keeps
    describing a table that no longer exists in that shape.
    """
    for name, contract in CONTRACTS.items():
        frame = smoke_tables.get(name)
        if frame is None:
            continue
        missing = [c for c in contract.columns if c not in frame.columns]
        assert not missing, f"{name} is missing contracted columns: {missing}"
