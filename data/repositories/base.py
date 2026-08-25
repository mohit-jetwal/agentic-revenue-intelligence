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
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from data.repositories.point_in_time import PointInTimeView


class DataAccessError(RuntimeError):
    """Raised when the underlying store cannot satisfy a request."""


class DatasetNotFoundError(DataAccessError):
    """Raised when a requested table/dataset does not exist."""


class ResultTruncatedError(DataAccessError):
    """Raised when a query hit the row cap and results were cut short.

    Silence is the dangerous outcome here. A feature computed over a truncated
    panel is not obviously wrong - the lags and rolling windows simply stop
    early, the frame looks well-formed, and the model trains on a quietly
    mutilated history. Brief section 32 is explicit that incorrect results must
    not be returned silently, so hitting the cap is an error, not a shrug.

    Callers who genuinely want a bounded peek pass ``max_rows`` explicitly and
    are then opting into truncation with their eyes open.
    """

    def __init__(self, table: str, limit: int) -> None:
        super().__init__(
            f"Query on {table!r} returned exactly the {limit:,}-row cap, so results "
            f"were almost certainly truncated. Narrow the filters, or pass an "
            f"explicit max_rows to accept a bounded result, or raise "
            f"DATA__MAX_RESULT_ROWS."
        )
        self.table = table
        self.limit = limit


class DataRepository(ABC):
    """Read-only access to analytical (Gold-layer) datasets.

    Read-only by design: agents investigate and recommend, they do not write
    back to the warehouse. Any write path (feedback, investigation history) goes
    through application state, not this interface.

    All ``get_*`` methods share the same optional filter arguments. Filters are
    pushed down to the engine rather than applied in pandas afterwards, so a
    narrow query stays cheap regardless of table size.

    **Point-in-time access.** Time-series methods accept ``as_of_date``, which
    restricts observed data to what was knowable on that date. Which tables that
    restricts, and which are legitimately knowable in advance, is decided by
    :mod:`data.repositories.availability`.

    Prefer :meth:`as_of` over the keyword. Both apply the same rule, but a
    keyword can be forgotten on one call in one feature builder and leak the
    future with nothing to catch it, whereas a view has no method that would
    return future data at all. Feature builders in ``features/`` accept only a
    view, for exactly that reason.
    """

    def as_of(self, as_of_date: date) -> PointInTimeView:
        """Return a view of this repository frozen at ``as_of_date``.

        Every read through the view is restricted to what was knowable then.
        This is the interface feature engineering and model training should use;
        the raw repository is for ad-hoc analysis and for serving current state.
        """
        from data.repositories.point_in_time import PointInTimeView

        return PointInTimeView(self, as_of_date)

    # -- dimensions ---------------------------------------------------------

    @abstractmethod
    def get_products(
        self,
        *,
        product_ids: list[str] | None = None,
        category: str | None = None,
        brand: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Product master. One row per product."""

    @abstractmethod
    def get_stores(
        self,
        *,
        store_ids: list[str] | None = None,
        region: str | None = None,
        channel: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Store master. One row per store."""

    @abstractmethod
    def get_customers(
        self,
        *,
        customer_ids: list[str] | None = None,
        segment: str | None = None,
        region: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Customer master. Non-PII attributes only (segment, region, tier)."""

    @abstractmethod
    def get_calendar(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Date dimension: week, month, quarter, holiday and festival flags.

        Deliberately has no ``as_of_date``: holidays and the financial calendar
        are known years ahead, so cutting them at an as-of date would remove
        information a planner genuinely has.
        """

    @abstractmethod
    def get_product_relationships(
        self,
        *,
        product_ids: list[str] | None = None,
        relationship_type: str | None = None,
        validate: bool = False,
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
        as_of_date: date | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        validate: bool = False,
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
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Regular and selling price by product x store x date.

        Classified known-in-advance: a price file is set before its effective
        date, so ``as_of_date`` does not cut it. See
        :mod:`data.repositories.availability`.
        """

    @abstractmethod
    def get_inventory(
        self,
        *,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
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
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Promotion events with type, depth, spend and mechanics flags.

        Classified known-in-advance, because promotion mechanics are agreed with
        retailers weeks ahead. A forward-dated row's *actuals* - realised spend
        and units - are still unknowable and are nulled beyond ``as_of_date``.
        """

    @abstractmethod
    def get_trade_promotions(
        self,
        *,
        product_ids: list[str] | None = None,
        retailer: str | None = None,
        region: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Trade promotion plans: planned/actual spend, expected/actual uplift, ROI.

        Classified observed despite being planned: the table is dominated by
        after-the-fact columns (actual spend, actual uplift, realised ROI), and
        letting those past the as-of date would leak the very outcome a model is
        trying to predict.
        """

    @abstractmethod
    def get_competitor_prices(
        self,
        *,
        product_ids: list[str] | None = None,
        competitor_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Competitor price and promotion observations by product x date.

        Observed, not planned. We may know our own future price list; we never
        know a rival's until it hits the shelf.
        """

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
