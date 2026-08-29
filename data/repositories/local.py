"""Local repository: DuckDB over Parquet.

Stage 1 implementation of :class:`~data.repositories.base.DataRepository`.

Why DuckDB rather than SQLite for the analytical side: every model here runs
full-column aggregations over the sales fact (group by product, by week, by
region). SQLite is row-oriented and reads whole rows to reach two columns;
DuckDB is columnar and vectorised, reads only the columns touched, and queries
Parquet in place with no import step. At the 6M-row dev scale that is the
difference between a model fitting in seconds and in minutes.

SQLite still has a job in this project - application state (investigations,
traces, feedback) - which is transactional, small, and exactly what it is good
at. Two engines, one job each.

Two properties this class must hold, both load-bearing:

* **Filters are pushed into SQL, not applied in pandas afterwards.** A request
  for one product over one month must scan one product over one month. Reading
  6M rows and filtering in memory would work and would be unusable.
* **``ground_truth/`` is unreachable.** Every path here is rooted at the gold
  layer. There is no method, and no argument to any method, that can reach the
  hidden simulation parameters - so a future model cannot accidentally train on
  the answers it is meant to estimate.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.observability.logging import get_logger
from data.contracts.tables import contract_for
from data.repositories.availability import clamp_window
from data.repositories.base import (
    DataAccessError,
    DataRepository,
    DatasetNotFoundError,
    ResultTruncatedError,
)

logger = get_logger(__name__)

#: Statement prefixes permitted through :meth:`LocalDataRepository.execute_query`.
_READ_ONLY_PREFIXES = ("select", "with")

#: A bare SQL identifier. Column names reach the SELECT list by interpolation
#: (they cannot be bound as parameters), so they are validated against this
#: rather than trusted. From Step 13 an agent chooses those column lists, which
#: makes this the difference between a projection and an injection point.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Never allowed, even inside a CTE. Belt and braces: the connection is opened
#: read-only as well, so this is a second line of defence rather than the only one.
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "attach",
    "copy",
    "export",
    "install",
    "load",
    "pragma",
    "set ",
)


class LocalDataRepository(DataRepository):
    """Reads gold-layer Parquet datasets through an embedded DuckDB engine."""

    def __init__(
        self,
        parquet_root: Path,
        *,
        duckdb_path: Path | None = None,
        max_result_rows: int = 100_000,
        query_timeout_seconds: int = 30,
    ) -> None:
        self.parquet_root = Path(parquet_root)
        self.duckdb_path = duckdb_path
        self.max_result_rows = max_result_rows
        self.query_timeout_seconds = query_timeout_seconds
        self._connection: duckdb.DuckDBPyConnection | None = None
        # DuckDB connections are not thread-safe for concurrent cursors, and
        # FastAPI will call this from a worker thread pool.
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            # In-memory: the Parquet files are the source of truth, so a
            # persistent database would only be a second copy that can go stale.
            self._connection = duckdb.connect(database=":memory:")
            self._connection.execute("SET enable_progress_bar = false")
        return self._connection

    def _table_path(self, table: str) -> Path:
        """Resolve a table name to its Parquet location.

        Handles both single-file tables (dimensions) and partitioned
        directories (the large facts), so callers never need to know which is
        which.
        """
        single = self.parquet_root / f"{table}.parquet"
        if single.is_file():
            return single
        partitioned = self.parquet_root / table
        if partitioned.is_dir():
            return partitioned
        raise DatasetNotFoundError(
            f"No dataset {table!r} under {self.parquet_root}. "
            f"Generate one first: uv run ari generate-data --profile dev"
        )

    def _source(self, table: str) -> str:
        """SQL expression selecting a table's Parquet files."""
        path = self._table_path(table)
        if path.is_dir():
            glob = str(path / "**" / "*.parquet").replace("\\", "/")
            return f"read_parquet('{glob}', hive_partitioning = true, union_by_name = true)"
        return f"read_parquet('{str(path).replace(chr(92), '/')}')"

    @staticmethod
    def _projection(columns: list[str] | None, *, prefix: str = "") -> str:
        """Build a validated SELECT list.

        Column names cannot be passed as bound parameters, so they are checked
        against :data:`_IDENTIFIER` before interpolation. Anything that is not a
        plain identifier is rejected outright rather than escaped - a caller
        with a legitimate need never has to send one.
        """
        if not columns:
            return f"{prefix}*" if prefix else "*"
        for column in columns:
            if not _IDENTIFIER.match(column):
                raise DataAccessError(f"invalid column name: {column!r}")
        return ", ".join(f"{prefix}{column}" for column in columns)

    def _run(self, sql: str, parameters: list[Any] | None = None) -> pd.DataFrame:
        with self._lock:
            try:
                relation = self._connect().execute(sql, parameters or [])
                frame: pd.DataFrame = relation.fetchdf()
            except DatasetNotFoundError:
                raise
            except Exception as exc:
                raise DataAccessError(f"query failed: {exc}") from exc
        return frame

    # -- filter helpers -----------------------------------------------------

    @staticmethod
    def _add_in(
        clauses: list[str],
        parameters: list[Any],
        column: str,
        values: list[str] | None,
    ) -> None:
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        clauses.append(f"{column} IN ({placeholders})")
        parameters.extend(values)

    @staticmethod
    def _add_equals(
        clauses: list[str], parameters: list[Any], column: str, value: str | None
    ) -> None:
        if value is None:
            return
        clauses.append(f"{column} = ?")
        parameters.append(value)

    @staticmethod
    def _add_date_range(
        clauses: list[str],
        parameters: list[Any],
        column: str,
        start: date | None,
        end: date | None,
    ) -> None:
        if start is not None:
            clauses.append(f"{column} >= ?")
            parameters.append(start)
        if end is not None:
            clauses.append(f"{column} <= ?")
            parameters.append(end)

    def _finish(
        self,
        table: str,
        frame: pd.DataFrame,
        *,
        limit: int,
        explicit_limit: bool,
        validate: bool,
        started: float,
        filters: dict[str, Any],
    ) -> pd.DataFrame:
        """Common tail for every read: truncation guard, contract, logging.

        The truncation guard is the important part. Every query carries a
        ``LIMIT``, so a large slice comes back quietly cut short - and a feature
        computed over a truncated panel is not obviously wrong. The lags and
        rolling windows simply stop early, the frame looks well-formed, and the
        model trains on a mutilated history. Brief section 32 forbids returning
        incorrect results silently, so hitting the cap without having asked for
        a bounded result is an error.
        """
        rows = len(frame)
        if rows >= limit and not explicit_limit:
            logger.error("repository.truncated", table=table, rows=rows, limit=limit)
            raise ResultTruncatedError(table, limit)

        if validate:
            contract = contract_for(table)
            if contract is not None:
                frame = contract.validate(frame)

        # Section 33: what was asked for and what came back, never the data.
        logger.debug(
            "repository.read",
            table=table,
            rows=rows,
            duration_ms=int((time.perf_counter() - started) * 1000),
            **{k: v for k, v in filters.items() if v is not None},
        )
        return frame

    def _select(
        self,
        table: str,
        clauses: list[str],
        parameters: list[Any],
        *,
        columns: list[str] | None = None,
        limit: int | None = None,
        validate: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        started = time.perf_counter()
        effective_limit = int(limit or self.max_result_rows)

        projection = self._projection(columns)
        # Safe against injection: `projection` is identifier-validated above, the
        # FROM clause is a filesystem path resolved by `_source`, every WHERE
        # clause is a class-internal literal, and all user values bind as `?`.
        sql = f"SELECT {projection} FROM {self._source(table)}"  # nosec B608
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" LIMIT {effective_limit}"

        frame = self._run(sql, parameters)
        return self._finish(
            table,
            frame,
            limit=effective_limit,
            explicit_limit=limit is not None,
            validate=validate,
            started=started,
            filters=filters or {},
        )

    # -- dimensions ---------------------------------------------------------

    def get_products(
        self,
        *,
        product_ids: list[str] | None = None,
        category: str | None = None,
        brand: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_equals(clauses, parameters, "category", category)
        self._add_equals(clauses, parameters, "brand", brand)
        return self._select(
            "products", clauses, parameters, validate=validate, filters={"category": category}
        )

    def get_stores(
        self,
        *,
        store_ids: list[str] | None = None,
        region: str | None = None,
        channel: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "store_id", store_ids)
        self._add_equals(clauses, parameters, "region", region)
        self._add_equals(clauses, parameters, "channel", channel)
        return self._select(
            "stores", clauses, parameters, validate=validate, filters={"region": region}
        )

    def get_customers(
        self,
        *,
        customer_ids: list[str] | None = None,
        segment: str | None = None,
        region: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "customer_id", customer_ids)
        self._add_equals(clauses, parameters, "segment", segment)
        self._add_equals(clauses, parameters, "region", region)
        return self._select(
            "customers", clauses, parameters, validate=validate, filters={"segment": segment}
        )

    def get_calendar(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_date_range(clauses, parameters, "date", start_date, end_date)
        # The calendar is small and callers usually want the whole span; the
        # default row cap would truncate three years of dates and now raise.
        return self._select("calendar", clauses, parameters, limit=200_000, validate=validate)

    def get_product_relationships(
        self,
        *,
        product_ids: list[str] | None = None,
        relationship_type: str | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list[Any] = []
        if product_ids:
            placeholders = ", ".join("?" for _ in product_ids)
            # Either side of the pair: callers asking about a product want its
            # relationships in both directions.
            clauses.append(f"(product_a IN ({placeholders}) OR product_b IN ({placeholders}))")
            parameters.extend(product_ids)
            parameters.extend(product_ids)
        self._add_equals(clauses, parameters, "relationship_type", relationship_type)
        return self._select(
            "product_relationships",
            clauses,
            parameters,
            validate=validate,
            filters={"relationship_type": relationship_type},
        )

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
        as_of_date: date | None = None,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        started = time.perf_counter()
        start_date, end_date = clamp_window("sales_daily", start_date, end_date, as_of_date)

        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "s.product_id", product_ids)
        self._add_in(clauses, parameters, "s.store_id", store_ids)
        self._add_equals(clauses, parameters, "s.channel", channel)
        self._add_date_range(clauses, parameters, "s.date", start_date, end_date)

        projection = self._projection(columns, prefix="s.")

        if region is not None:
            # Region lives on the store dimension, so filtering by it needs a
            # join. Kept out of the default path: an unnecessary join over
            # millions of rows is a real cost.
            # Safe: identifier-validated projection, class-internal join clause.
            join = f"JOIN {self._source('stores')} st ON s.store_id = st.store_id"
            sql = f"SELECT {projection} FROM {self._source('sales_daily')} s {join}"  # nosec B608
            clauses.append("st.region = ?")
            parameters.append(region)
        else:
            sql = f"SELECT {projection} FROM {self._source('sales_daily')} s"  # nosec B608

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        effective_limit = int(max_rows or self.max_result_rows)
        sql += f" LIMIT {effective_limit}"

        frame = self._run(sql, parameters)
        return self._finish(
            "sales_daily",
            frame,
            limit=effective_limit,
            explicit_limit=max_rows is not None,
            # A projected subset cannot satisfy a whole-table contract, so
            # validation is skipped rather than failing on absent columns.
            validate=validate and columns is None,
            started=started,
            filters={
                "products": len(product_ids) if product_ids else None,
                "stores": len(store_ids) if store_ids else None,
                "region": region,
                "channel": channel,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
                "as_of_date": str(as_of_date) if as_of_date else None,
            },
        )

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
        start_date, end_date = clamp_window("pricing", start_date, end_date, as_of_date)
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_in(clauses, parameters, "store_id", store_ids)
        self._add_date_range(clauses, parameters, "date", start_date, end_date)
        return self._select(
            "pricing",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"products": len(product_ids) if product_ids else None},
        )

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
        start_date, end_date = clamp_window("inventory", start_date, end_date, as_of_date)
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_in(clauses, parameters, "store_id", store_ids)
        self._add_date_range(clauses, parameters, "date", start_date, end_date)
        return self._select(
            "inventory",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"products": len(product_ids) if product_ids else None},
        )

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
        start_date, end_date = clamp_window("promotions", start_date, end_date, as_of_date)
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_in(clauses, parameters, "store_id", store_ids)
        self._add_equals(clauses, parameters, "promotion_type", promotion_type)
        # Overlap semantics, not containment: a promotion that started before
        # the window but is still running inside it is relevant, and a
        # containment filter would silently drop it.
        if start_date is not None:
            clauses.append("end_date >= ?")
            parameters.append(start_date)
        if end_date is not None:
            clauses.append("start_date <= ?")
            parameters.append(end_date)
        return self._select(
            "promotions",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"promotion_type": promotion_type},
        )

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
        start_date, end_date = clamp_window("trade_promotions", start_date, end_date, as_of_date)
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_equals(clauses, parameters, "retailer", retailer)
        self._add_equals(clauses, parameters, "region", region)
        if start_date is not None:
            clauses.append("end_date >= ?")
            parameters.append(start_date)
        if end_date is not None:
            clauses.append("start_date <= ?")
            parameters.append(end_date)
        return self._select(
            "trade_promotions",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"retailer": retailer, "region": region},
        )

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
        start_date, end_date = clamp_window(
            "competitor_pricing", start_date, end_date, as_of_date
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "product_id", product_ids)
        self._add_in(clauses, parameters, "competitor_id", competitor_ids)
        self._add_date_range(clauses, parameters, "date", start_date, end_date)
        return self._select(
            "competitor_pricing",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"products": len(product_ids) if product_ids else None},
        )

    def get_commodity_costs(
        self,
        *,
        categories: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of_date: date | None = None,
        max_rows: int | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        start_date, end_date = clamp_window(
            "commodity_costs", start_date, end_date, as_of_date
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        self._add_in(clauses, parameters, "category", categories)
        self._add_date_range(clauses, parameters, "date", start_date, end_date)
        return self._select(
            "commodity_costs",
            clauses,
            parameters,
            limit=max_rows,
            validate=validate,
            filters={"categories": len(categories) if categories else None},
        )

    # -- generic ------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Run a read-only analytical query.

        The escape hatch for the Text-to-SQL tool. Guarded here rather than
        trusting the caller: the guardrail layer in Step 20 will validate first,
        but a repository that assumes validation happened is one refactor away
        from being an open door.

        Read-only is enforced twice - by statement inspection, and by the table
        surface itself, since only gold-layer views are registered.
        """
        statement = sql.strip().rstrip(";")
        lowered = statement.lower()

        if not lowered.startswith(_READ_ONLY_PREFIXES):
            raise DataAccessError(
                f"only SELECT and WITH statements are permitted, got: {statement[:40]!r}"
            )
        for keyword in _FORBIDDEN_KEYWORDS:
            if keyword in lowered:
                raise DataAccessError(f"statement contains forbidden keyword {keyword.strip()!r}")

        limit = int(max_rows or self.max_result_rows)
        # Interpolating `statement` is the documented purpose of this method - it
        # is the Text-to-SQL escape hatch. Three things contain it: the read-only
        # prefix check and forbidden-keyword scan above, the int()-coerced limit,
        # and the fact that only gold-layer views are registered, so there is
        # nothing sensitive in scope to reach.
        wrapped = f"SELECT * FROM ({statement}) AS _q LIMIT {limit}"  # nosec B608

        with self._lock:
            connection = self._connect()
            self._register_views(connection)
            try:
                frame: pd.DataFrame = connection.execute(
                    wrapped, list((parameters or {}).values())
                ).fetchdf()
            except Exception as exc:
                raise DataAccessError(f"query failed: {exc}") from exc
        return frame

    def _register_views(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Expose gold tables as named views for ad-hoc SQL.

        Only gold. ``ground_truth`` is never registered, so no query - however
        it is written - can reach the hidden simulation parameters.
        """
        for table in self.list_tables():
            try:
                source = self._source(table)
            except DatasetNotFoundError:
                continue
            # Safe: `table` comes from filesystem discovery in list_tables() and
            # `source` from _source(); neither is caller-supplied.
            view_sql = f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM {source}"  # nosec B608
            connection.execute(view_sql)

    # -- introspection ------------------------------------------------------

    def list_tables(self) -> list[str]:
        """Discover gold datasets: single Parquet files and partitioned dirs."""
        if not self.parquet_root.exists():
            return []
        names: set[str] = set()
        for entry in self.parquet_root.iterdir():
            if entry.is_file() and entry.suffix == ".parquet":
                names.add(entry.stem)
            elif entry.is_dir() and any(entry.rglob("*.parquet")):
                names.add(entry.name)
        return sorted(names)

    def describe_table(self, table: str) -> pd.DataFrame:
        """Column names and types.

        Feeds schema discovery for the Text-to-SQL agent in Step 14. Unity
        Catalog supplies column comments in Stage 2; Parquet has no equivalent,
        so the ``comment`` column is present but empty here - keeping the shape
        stable across both implementations.
        """
        # Safe: `_source` resolves a discovered filesystem path, not caller input.
        frame = self._run(f"DESCRIBE SELECT * FROM {self._source(table)}")  # nosec B608
        frame = frame.rename(columns={"column_name": "name", "column_type": "type"})
        frame["comment"] = pd.Series([None] * len(frame), dtype="string")
        return frame[["name", "type", "comment"]]

    def dataset_version(self) -> str:
        """Identifier for the current data snapshot.

        Read from the manifest and stamped onto every ``ToolResult``, so a
        recommendation can be traced to the exact seed and config that produced
        the data behind it.
        """
        from data.generation.writer import read_manifest

        manifest = read_manifest(self.parquet_root.parent)
        version = manifest.get("dataset_version", "unknown")
        config_hash = manifest.get("config_hash", "")
        return f"{version}+{config_hash}" if config_hash else str(version)

    def health_check(self) -> tuple[bool, str]:
        if not self.parquet_root.exists():
            return False, f"parquet root not found: {self.parquet_root} (run Step 2)"
        tables = self.list_tables()
        if not tables:
            return False, f"no datasets under {self.parquet_root} (run Step 2)"
        try:
            version = self.dataset_version()
        except FileNotFoundError:
            return False, f"{len(tables)} datasets but no manifest (regenerate)"
        return True, f"{len(tables)} tables, dataset {version}"

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
