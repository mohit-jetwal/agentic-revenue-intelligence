"""Point-in-time access and the truncation guard (brief sections 10-11, 32).

Covers the two repository behaviours Step 3 added: filters that narrow correctly
and loudly, and an as-of cut that respects each table's availability class.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.repositories.availability import (
    TABLE_AVAILABILITY,
    Availability,
    availability_of,
    clamp_window,
)
from data.repositories.base import ResultTruncatedError
from data.repositories.local import LocalDataRepository
from data.repositories.point_in_time import PointInTimeView

pytestmark = [pytest.mark.data, pytest.mark.integration]


# --- the truncation guard ---------------------------------------------------


def test_hitting_the_row_cap_raises(smoke_result: object) -> None:
    """Silent truncation is the bug this guards against.

    A feature computed over a truncated panel is not obviously wrong - the lags
    simply stop early and the frame looks well-formed. Section 32 forbids
    returning incorrect results silently.
    """
    tiny = LocalDataRepository(
        parquet_root=smoke_result.root / "gold",  # type: ignore[attr-defined]
        max_result_rows=100,
    )
    with pytest.raises(ResultTruncatedError) as exc_info:
        tiny.get_sales()

    assert "100" in str(exc_info.value)
    assert "max_rows" in str(exc_info.value)


def test_explicit_max_rows_opts_into_truncation(smoke_result: object) -> None:
    """A caller asking for a bounded peek gets one, without an exception."""
    tiny = LocalDataRepository(
        parquet_root=smoke_result.root / "gold",  # type: ignore[attr-defined]
        max_result_rows=100,
    )
    frame = tiny.get_sales(max_rows=50)
    assert len(frame) == 50


def test_a_query_under_the_cap_is_fine(smoke_repository: LocalDataRepository) -> None:
    frame = smoke_repository.get_products()
    assert 0 < len(frame) < smoke_repository.max_result_rows


# --- filters (section 10) ---------------------------------------------------


def test_date_filter_narrows(smoke_repository: LocalDataRepository) -> None:
    frame = smoke_repository.get_sales(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31))
    dates = pd.to_datetime(frame["date"]).dt.date
    assert dates.min() >= date(2024, 3, 1)
    assert dates.max() <= date(2024, 3, 31)


def test_product_filter_narrows(smoke_repository: LocalDataRepository) -> None:
    products = smoke_repository.get_products()["product_id"].head(2).tolist()
    frame = smoke_repository.get_sales(product_ids=products)
    assert set(frame["product_id"]) <= set(products)


def test_store_filter_narrows(smoke_repository: LocalDataRepository) -> None:
    stores = smoke_repository.get_stores()["store_id"].head(3).tolist()
    frame = smoke_repository.get_sales(store_ids=stores)
    assert set(frame["store_id"]) <= set(stores)


def test_combined_filters_narrow(smoke_repository: LocalDataRepository) -> None:
    products = smoke_repository.get_products()["product_id"].head(3).tolist()
    stores = smoke_repository.get_stores()["store_id"].head(3).tolist()

    frame = smoke_repository.get_sales(
        product_ids=products,
        store_ids=stores,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 29),
    )
    if frame.empty:
        pytest.skip("sampled products and stores are not co-listed in this window")

    assert set(frame["product_id"]) <= set(products)
    assert set(frame["store_id"]) <= set(stores)
    dates = pd.to_datetime(frame["date"]).dt.date
    assert dates.min() >= date(2024, 1, 1)
    assert dates.max() <= date(2024, 2, 29)


# --- availability classification --------------------------------------------


def test_every_gold_table_is_classified(smoke_repository: LocalDataRepository) -> None:
    """An unclassified table defaults to OBSERVED, which is safe but lossy.

    Better to notice here than to wonder later why a table stopped appearing
    past the as-of date.
    """
    unclassified = [
        table
        for table in smoke_repository.list_tables()
        if table not in TABLE_AVAILABILITY and not table.startswith("sales_")
    ]
    assert not unclassified, f"tables with no availability class: {unclassified}"


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("sales_daily", Availability.OBSERVED),
        ("inventory", Availability.OBSERVED),
        ("competitor_pricing", Availability.OBSERVED),
        ("trade_promotions", Availability.OBSERVED),
        ("calendar", Availability.KNOWN_IN_ADVANCE),
        ("promotions", Availability.KNOWN_IN_ADVANCE),
        ("pricing", Availability.KNOWN_IN_ADVANCE),
        ("products", Availability.STATIC),
        ("stores", Availability.STATIC),
    ],
)
def test_availability_classes_are_as_intended(table: str, expected: Availability) -> None:
    """Pin the classification, because it is the leak-shaped decision.

    Moving a table into KNOWN_IN_ADVANCE silently permits future data, so the
    change should have to be made deliberately and visibly.
    """
    assert availability_of(table) is expected


def test_unknown_tables_default_to_observed() -> None:
    """Forgetting to classify costs signal, never correctness."""
    assert availability_of("some_new_table_nobody_classified") is Availability.OBSERVED


def test_clamp_window_only_cuts_observed_tables() -> None:
    as_of = date(2024, 6, 30)
    far = date(2025, 12, 31)

    _, observed_end = clamp_window("sales_daily", None, far, as_of)
    assert observed_end == as_of

    _, planned_end = clamp_window("promotions", None, far, as_of)
    assert planned_end == far

    _, no_as_of = clamp_window("sales_daily", None, far, None)
    assert no_as_of == far


# --- the view ---------------------------------------------------------------


def test_as_of_returns_a_view(smoke_repository: LocalDataRepository) -> None:
    view = smoke_repository.as_of(date(2024, 6, 30))
    assert isinstance(view, PointInTimeView)
    assert view.as_of_date == date(2024, 6, 30)


def test_view_qualifies_the_dataset_version(smoke_view: PointInTimeView) -> None:
    """Two feature sets from the same data at different as-of dates are
    different artifacts, and the version has to say so."""
    version = smoke_view.dataset_version()
    assert "@" in version
    assert str(smoke_view.as_of_date) in version


def test_view_rebases_rather_than_nesting(smoke_view: PointInTimeView) -> None:
    """Re-freezing must not stack views into an ambiguous chain."""
    rebased = smoke_view.as_of(date(2024, 1, 31))
    assert rebased.as_of_date == date(2024, 1, 31)
    assert rebased.repository is smoke_view.repository


def test_view_is_a_repository(smoke_view: PointInTimeView) -> None:
    """Anything accepting a repository accepts a view - including, eventually,
    the Databricks implementation."""
    from data.repositories.base import DataRepository

    assert isinstance(smoke_view, DataRepository)


def test_static_dimensions_pass_through(smoke_view: PointInTimeView) -> None:
    view_products = smoke_view.get_products()
    direct = smoke_view.repository.get_products()
    assert len(view_products) == len(direct)
