"""Databricks repository: Unity Catalog Gold tables via Databricks SQL.

Stage 2 implementation of :class:`~data.repositories.base.DataRepository`.

Declared now, deliberately, even though it raises throughout. Two reasons:

1. It proves the interface is satisfiable by a warehouse-backed store, not just
   by local files. An abstraction is only worth having if a second
   implementation can honour it; writing the signatures now surfaces any place
   where the ABC accidentally leaked a local-only assumption.
2. It makes the migration path concrete and reviewable rather than aspirational.

Notes for Stage 2 implementation:

* Connect with ``databricks-sql-connector`` against a SQL Warehouse, or use
  ``databricks-connect`` where a Spark session is genuinely needed. The
  warehouse is the right default - these are filtered reads, not distributed
  jobs, and a warehouse avoids cluster spin-up latency on the agent's critical
  path.
* Authenticate with a service principal via a Databricks secret scope. Never a
  personal access token in configuration.
* ``dataset_version()`` should return the Delta table version
  (``DESCRIBE HISTORY``), giving every recommendation an exact, reproducible
  data reference.
* Access is governed by Unity Catalog grants on the service principal:
  ``SELECT`` on the Gold schema only. The application does not enforce
  read-only access; the catalog does.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from data.repositories.base import DataRepository

_STAGE = (
    "Stage 2 (Databricks). Set APP__ENVIRONMENT=databricks only after the "
    "Stage 2 production implementation steps are complete."
)


class DatabricksDataRepository(DataRepository):
    """Reads Gold Delta tables in Unity Catalog through Databricks SQL."""

    def __init__(
        self,
        *,
        host: str,
        token: str,
        warehouse_id: str,
        catalog: str,
        schema: str,
        max_result_rows: int = 100_000,
        query_timeout_seconds: int = 30,
    ) -> None:
        self.host = host
        self._token = token
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        self.schema = schema
        self.max_result_rows = max_result_rows
        self.query_timeout_seconds = query_timeout_seconds

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"DatabricksDataRepository.{method}() belongs to {_STAGE}")

    def _table(self, name: str) -> str:
        """Fully-qualified Unity Catalog table name."""
        return f"{self.catalog}.{self.schema}.{name}"

    # -- dimensions ---------------------------------------------------------

    def get_products(
        self,
        *,
        product_ids: list[str] | None = None,
        category: str | None = None,
        brand: str | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_products")

    def get_stores(
        self,
        *,
        store_ids: list[str] | None = None,
        region: str | None = None,
        channel: str | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_stores")

    def get_customers(
        self,
        *,
        customer_ids: list[str] | None = None,
        segment: str | None = None,
        region: str | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_customers")

    def get_calendar(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_calendar")

    def get_product_relationships(
        self,
        *,
        product_ids: list[str] | None = None,
        relationship_type: str | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_product_relationships")

    # -- facts --------------------------------------------------------------

    def get_sales(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        channel: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_sales")

    def get_pricing(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_pricing")

    def get_inventory(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_inventory")

    def get_promotions(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        promotion_type: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_promotions")

    def get_trade_promotions(
        self,
        *,
        product_ids: list[str] | None = None,
        retailer: str | None = None,
        region: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_trade_promotions")

    def get_competitor_prices(
        self,
        *,
        product_ids: list[str] | None = None,
        competitor_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("get_competitor_prices")

    # -- generic ------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        raise self._not_yet("execute_query")

    # -- introspection ------------------------------------------------------

    def list_tables(self) -> list[str]:
        raise self._not_yet("list_tables")

    def describe_table(self, table: str) -> pd.DataFrame:
        raise self._not_yet("describe_table")

    def dataset_version(self) -> str:
        raise self._not_yet("dataset_version")

    def health_check(self) -> tuple[bool, str]:
        return False, f"Databricks repository not implemented. {_STAGE}"
