"""Local repository: DuckDB over Parquet.

Stage 1 implementation of :class:`~data.repositories.base.DataRepository`.

Why DuckDB rather than SQLite for the analytical side: every model here runs
full-column aggregations over the sales fact (group by product, by week, by
region). SQLite is row-oriented and will scan the whole row to read two columns;
DuckDB is columnar and vectorised, reads only what the query touches, and queries
Parquet files in place without an import step. At the 1M-10M row scale the brief
asks for, that is the difference between a model fitting in seconds and in
minutes.

SQLite still has a job in this project - application state (investigations,
traces, feedback) - which is transactional, small, and exactly what it is good
at. Two engines, one job each.

Method bodies are filled in at Stage 1 Step 3, once Step 2 has generated the
datasets. They are declared now so the seam is real: the container can construct
this class today and the interface is provably satisfiable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from data.repositories.base import DataRepository

_STEP = "Stage 1 Step 3 (local database / repository)"


class LocalDataRepository(DataRepository):
    """Reads Gold-layer Parquet datasets through an embedded DuckDB engine."""

    def __init__(
        self,
        parquet_root: Path,
        *,
        duckdb_path: Path | None = None,
        max_result_rows: int = 100_000,
        query_timeout_seconds: int = 30,
    ) -> None:
        self.parquet_root = parquet_root
        #: ``None`` means an in-memory DuckDB that reads Parquet directly.
        self.duckdb_path = duckdb_path
        self.max_result_rows = max_result_rows
        self.query_timeout_seconds = query_timeout_seconds

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(
            f"LocalDataRepository.{method}() is implemented in {_STEP}. "
            f"Run data generation (Step 2) first; expected Parquet root: "
            f"{self.parquet_root}"
        )

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
        """Discover datasets by looking for Parquet files under the root.

        Implemented now (rather than deferred) so ``GET /health`` can report
        honestly whether data has been generated yet.
        """
        if not self.parquet_root.exists():
            return []
        names = {
            p.stem if p.is_file() else p.name
            for p in self.parquet_root.iterdir()
            if p.name.endswith(".parquet") or p.is_dir()
        }
        return sorted(names)

    def describe_table(self, table: str) -> pd.DataFrame:
        raise self._not_yet("describe_table")

    def dataset_version(self) -> str:
        raise self._not_yet("dataset_version")

    def health_check(self) -> tuple[bool, str]:
        if not self.parquet_root.exists():
            return False, f"parquet root not found: {self.parquet_root} (run Step 2)"
        tables = self.list_tables()
        if not tables:
            return False, f"no datasets under {self.parquet_root} (run Step 2)"
        return True, f"{len(tables)} datasets under {self.parquet_root}"
