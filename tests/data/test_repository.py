"""LocalDataRepository over the generated dataset.

Confirms the acceptance criterion from brief section 36 - that the data is
reachable through the repository abstraction - and the two properties the
abstraction exists to guarantee: filters push down, and ground truth is
unreachable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data.generation.pipeline import GenerationResult
from data.repositories.base import DataAccessError, DatasetNotFoundError
from data.repositories.local import LocalDataRepository

pytestmark = [pytest.mark.data, pytest.mark.integration]


@pytest.fixture(scope="module")
def repository(smoke_result: GenerationResult) -> LocalDataRepository:
    return LocalDataRepository(
        parquet_root=smoke_result.root / "gold",
        max_result_rows=200_000,
    )


# --- discovery -------------------------------------------------------------


def test_lists_expected_tables(repository: LocalDataRepository) -> None:
    tables = set(repository.list_tables())
    assert {
        "products",
        "stores",
        "customers",
        "calendar",
        "sales_daily",
        "pricing",
        "inventory",
        "promotions",
        "competitor_pricing",
    } <= tables


def test_ground_truth_is_not_discoverable(repository: LocalDataRepository) -> None:
    """The guarantee that keeps hidden parameters out of future models."""
    tables = repository.list_tables()
    assert "latent_demand" not in tables
    assert not any("elasticity" in t or "ground_truth" in t for t in tables)


def test_health_check_reports_the_dataset_version(repository: LocalDataRepository) -> None:
    healthy, detail = repository.health_check()
    assert healthy
    assert "v1.0-smoke" in detail


def test_dataset_version_includes_config_hash(repository: LocalDataRepository) -> None:
    """Every ToolResult stamps this, so a recommendation traces to exact data."""
    version = repository.dataset_version()
    assert version.startswith("v1.0-smoke+")


def test_describe_table_returns_schema(repository: LocalDataRepository) -> None:
    schema = repository.describe_table("sales_daily")
    assert list(schema.columns) == ["name", "type", "comment"]
    assert "units" in set(schema["name"])


def test_unknown_table_raises_with_guidance(repository: LocalDataRepository) -> None:
    with pytest.raises(DatasetNotFoundError, match="generate-data"):
        repository.describe_table("no_such_table")


# --- typed reads -----------------------------------------------------------


def test_get_products_filters_by_category(repository: LocalDataRepository) -> None:
    everything = repository.get_products()
    category = str(everything["category"].iloc[0])
    filtered = repository.get_products(category=category)

    assert not filtered.empty
    assert set(filtered["category"]) == {category}
    assert len(filtered) < len(everything)


def test_get_stores_filters_by_region(repository: LocalDataRepository) -> None:
    stores = repository.get_stores()
    region = str(stores["region"].iloc[0])
    filtered = repository.get_stores(region=region)
    assert set(filtered["region"]) == {region}


def test_get_sales_filters_by_product_and_date(repository: LocalDataRepository) -> None:
    products = repository.get_products()
    product_id = str(products["product_id"].iloc[0])

    start = date(2024, 3, 1)
    end = date(2024, 3, 31)
    sales = repository.get_sales(product_ids=[product_id], start_date=start, end_date=end)

    assert not sales.empty
    assert set(sales["product_id"]) == {product_id}
    dates = pd.to_datetime(sales["date"]).dt.date
    assert dates.min() >= start
    assert dates.max() <= end


def test_get_sales_filters_by_region_via_join(repository: LocalDataRepository) -> None:
    stores = repository.get_stores()
    region = str(stores["region"].iloc[0])
    region_stores = set(stores.loc[stores["region"] == region, "store_id"])

    sales = repository.get_sales(
        region=region, start_date=date(2024, 1, 1), end_date=date(2024, 1, 15)
    )
    assert not sales.empty
    assert set(sales["store_id"]) <= region_stores


def test_get_sales_projects_requested_columns(repository: LocalDataRepository) -> None:
    sales = repository.get_sales(
        columns=["date", "product_id", "units"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )
    assert list(sales.columns) == ["date", "product_id", "units"]


@pytest.mark.parametrize(
    "column",
    [
        "units; DROP TABLE products",
        "units FROM products--",
        "*, (SELECT 1)",
        "units)",
        "1=1",
        "",
    ],
)
def test_column_names_must_be_plain_identifiers(
    repository: LocalDataRepository, column: str
) -> None:
    """Column names cannot be bound as parameters, so they are validated.

    From Step 13 an agent chooses these column lists, which makes this the
    difference between a projection and an injection point.
    """
    with pytest.raises(DataAccessError, match="invalid column name"):
        repository.get_sales(columns=["date", column])


def test_valid_column_names_still_pass(repository: LocalDataRepository) -> None:
    frame = repository.get_sales(
        columns=["units", "revenue"], start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
    )
    assert list(frame.columns) == ["units", "revenue"]


def test_get_promotions_uses_overlap_not_containment(
    repository: LocalDataRepository,
) -> None:
    """A promotion running through the window must be returned.

    Containment semantics would silently drop promotions that started earlier -
    exactly the ones most likely to matter for a mid-period question.
    """
    promotions = repository.get_promotions()
    assert not promotions.empty

    sample = promotions.iloc[0]
    start = pd.Timestamp(sample["start_date"]).date()
    end = pd.Timestamp(sample["end_date"]).date()
    if end <= start:
        pytest.skip("degenerate promotion window")

    # A window strictly inside the promotion: containment would return nothing.
    inner_start = start + timedelta(days=1)
    inner_end = min(inner_start, end)
    overlapping = repository.get_promotions(start_date=inner_start, end_date=inner_end)
    assert sample["promotion_id"] in set(overlapping["promotion_id"])


def test_get_product_relationships_matches_either_side(
    repository: LocalDataRepository,
) -> None:
    relationships = repository.get_product_relationships()
    assert not relationships.empty

    product_id = str(relationships["product_b"].iloc[0])
    filtered = repository.get_product_relationships(product_ids=[product_id])
    assert not filtered.empty
    assert ((filtered["product_a"] == product_id) | (filtered["product_b"] == product_id)).all()


def test_get_inventory_and_competitor_prices_return_rows(
    repository: LocalDataRepository,
) -> None:
    inventory = repository.get_inventory(start_date=date(2024, 5, 1), end_date=date(2024, 5, 7))
    assert not inventory.empty

    competitor = repository.get_competitor_prices(
        start_date=date(2024, 5, 1), end_date=date(2024, 5, 7)
    )
    assert not competitor.empty


def test_calendar_is_not_truncated_by_the_row_cap(
    repository: LocalDataRepository,
) -> None:
    """The default cap would silently clip three years of dates."""
    calendar = repository.get_calendar()
    assert len(calendar) == 1096


# --- ad-hoc SQL ------------------------------------------------------------


def test_execute_query_runs_read_only_sql(repository: LocalDataRepository) -> None:
    frame = repository.execute_query(
        "SELECT region, COUNT(*) AS n FROM stores GROUP BY region ORDER BY region"
    )
    assert not frame.empty
    assert set(frame.columns) == {"region", "n"}


def test_execute_query_supports_cte(repository: LocalDataRepository) -> None:
    frame = repository.execute_query(
        "WITH t AS (SELECT category FROM products) SELECT COUNT(*) AS n FROM t"
    )
    assert int(frame["n"].iloc[0]) > 0


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE products",
        "DELETE FROM products",
        "UPDATE products SET base_price = 0",
        "INSERT INTO products VALUES (1)",
        "CREATE TABLE evil AS SELECT 1",
    ],
)
def test_execute_query_rejects_writes(repository: LocalDataRepository, statement: str) -> None:
    with pytest.raises(DataAccessError):
        repository.execute_query(statement)


def test_execute_query_rejects_write_hidden_in_a_cte(
    repository: LocalDataRepository,
) -> None:
    """Starting with WITH must not be a way past the guard."""
    with pytest.raises(DataAccessError):
        repository.execute_query("WITH x AS (SELECT 1) DELETE FROM products")


def test_execute_query_enforces_the_row_cap(repository: LocalDataRepository) -> None:
    frame = repository.execute_query("SELECT * FROM sales_daily", max_rows=25)
    assert len(frame) == 25


def test_execute_query_cannot_reach_ground_truth(
    repository: LocalDataRepository,
) -> None:
    """No amount of SQL creativity reaches the hidden parameters.

    Only gold tables are registered as views, so the name does not resolve.
    """
    with pytest.raises(DataAccessError):
        repository.execute_query("SELECT * FROM latent_demand")
