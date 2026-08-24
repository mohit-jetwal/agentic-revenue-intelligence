"""Data access interface - seam 2 of the dev/prod boundary.

Every ML model, every analytical tool and every agent reads data through this
interface and nothing else. No module outside ``data/repositories`` may import
``duckdb``, open a Parquet file, or hold a Databricks connection.

That restriction is what makes section 44 of the brief achievable: migrating to
Databricks means writing one new implementation of this ABC, not editing the
models. If a model ever calls ``pd.read_parquet`` directly, the migration turns
into a rewrite - which is precisely the failure this design exists to prevent.

The return type is ``pandas.DataFrame`` throughout. That is a deliberate
narrowing: it means the Databricks implementation must collect Spark results to
pandas before returning. Acceptable because every consumer here is a
statistical/ML model operating on a filtered slice (one category, one region, a
date window), not a full-table scan. Aggregations that *should* stay distributed
belong in the Gold layer or in a dedicated SQL tool, not in a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd


class DataAccessError(RuntimeError):
    """Raised when the underlying store cannot satisfy a request."""


class DatasetNotFoundError(DataAccessError):
    """Raised when a requested table/dataset does not exist."""


class DataRepository(ABC):
    """Read-only access to analytical (Gold-layer) datasets.

    Read-only by design: agents investigate and recommend, they do not write
    back to the warehouse. Any write path (feedback, investigation history) goes
    through application state, not this interface.

    All ``get_*`` methods share the same optional filter arguments. Filters are
    pushed down to the engine rather than applied in pandas afterwards, so a
    narrow query stays cheap regardless of table size.
    """

    # -- dimensions ---------------------------------------------------------

    @abstractmethod
    def get_products(
        self,
        *,
        product_ids: list[str] | None = None,
        category: str | None = None,
        brand: str | None = None,
    ) -> pd.DataFrame:
        """Product master. One row per product."""

    @abstractmethod
    def get_stores(
        self,
        *,
        store_ids: list[str] | None = None,
        region: str | None = None,
        channel: str | None = None,
    ) -> pd.DataFrame:
        """Store master. One row per store."""

    @abstractmethod
    def get_customers(
        self,
        *,
        customer_ids: list[str] | None = None,
        segment: str | None = None,
        region: str | None = None,
    ) -> pd.DataFrame:
        """Customer master. Non-PII attributes only (segment, region, tier)."""

    @abstractmethod
    def get_calendar(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Date dimension: week, month, quarter, holiday and festival flags."""

    @abstractmethod
    def get_product_relationships(
        self,
        *,
        product_ids: list[str] | None = None,
        relationship_type: str | None = None,
    ) -> pd.DataFrame:
        """Known substitute/complement pairs, used to scope cross-price work.

        Restricting the cross-price search space matters: with N products a
        naive matrix is O(N^2) regressions, most of them noise.
        """

    # -- facts --------------------------------------------------------------

    @abstractmethod
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
        """Daily sales fact at product x store x date grain."""

    @abstractmethod
    def get_pricing(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Regular and selling price by product x store x date."""

    @abstractmethod
    def get_inventory(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Daily inventory position and stockout flags.

        Needed to separate a demand decline from an availability-driven one -
        the distinction the Root Cause agent must be able to make.
        """

    @abstractmethod
    def get_promotions(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        promotion_type: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Promotion events with type, depth, spend and mechanics flags."""

    @abstractmethod
    def get_trade_promotions(
        self,
        *,
        product_ids: list[str] | None = None,
        retailer: str | None = None,
        region: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Trade promotion plans: planned/actual spend, expected/actual uplift, ROI."""

    @abstractmethod
    def get_competitor_prices(
        self,
        *,
        product_ids: list[str] | None = None,
        competitor_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Competitor price and promotion observations by product x date."""

    # -- generic ------------------------------------------------------------

    @abstractmethod
    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Run a validated, read-only analytical query.

        This exists for the Text-to-SQL tool, which needs an escape hatch the
        typed ``get_*`` methods cannot provide. It is **not** a general backdoor:
        the caller is responsible for having passed the SQL through the guardrail
        validator first (allowlisted schemas, read-only, row and time limits).
        Implementations must additionally enforce the row cap themselves - never
        trust the caller to have done it.
        """

    # -- introspection ------------------------------------------------------

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Available analytical tables. Used for schema discovery."""

    @abstractmethod
    def describe_table(self, table: str) -> pd.DataFrame:
        """Column names, types and comments for one table.

        Column comments are what a Text-to-SQL agent actually reasons over, so
        implementations should surface them where the platform provides them
        (Unity Catalog comments in Stage 2).
        """

    @abstractmethod
    def dataset_version(self) -> str:
        """Identifier for the current data snapshot.

        Stamped onto every :class:`~app.schemas.tool_contract.ToolResult` so a
        recommendation can be traced to the exact data it was computed from.
        In Stage 2 this is the Delta table version.
        """

    def health_check(self) -> tuple[bool, str]:
        """Cheap liveness probe used by ``GET /health``.

        Concrete rather than abstract because a default of "try to list tables"
        is correct for every implementation; override only if something cheaper
        exists.
        """
        try:
            tables = self.list_tables()
        except Exception as exc:  # noqa: BLE001 - health checks must not raise
            return False, f"{type(self).__name__}: {exc}"
        return True, f"{len(tables)} tables available"
