"""A repository frozen at a point in time.

The central defence against feature leakage.

Every model in Steps 4-11 is temporal, and a single leaked future value produces
a model that backtests beautifully and fails in production. The failure is
particularly nasty because nothing else in the system catches it: the frame is
well-formed, the metrics are excellent, and the error only surfaces when real
money is behind the recommendation.

An ``as_of_date`` keyword on each method would express the same intent, but it
can be forgotten - one omission in one feature builder, silently training on the
future. A view cannot be forgotten in the same way, because it has no method that
would return future observed data at all. Feature builders in ``features/``
therefore accept a :class:`PointInTimeView` rather than a bare repository, which
turns "remember to pass as_of" into "you physically cannot reach that row".

Which tables the cut applies to is decided by
:mod:`data.repositories.availability`, not here.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.availability import (
    Availability,
    actuals_columns_of,
    availability_of,
    clamp_window,
    date_column_of,
)
from data.repositories.base import DataRepository

logger = get_logger(__name__)


class PointInTimeView(DataRepository):
    """Read-only view of a repository restricted to one as-of date.

    Implements :class:`~data.repositories.base.DataRepository`, so anything that
    accepts a repository accepts a view - including, eventually, the Databricks
    implementation. The wrapping is transparent to callers.
    """

    def __init__(self, repository: DataRepository, as_of_date: date) -> None:
        self._repository = repository
        self.as_of_date = as_of_date
        logger.debug("point_in_time.view_created", as_of_date=str(as_of_date))

    def __repr__(self) -> str:
        return f"PointInTimeView({type(self._repository).__name__}, as_of={self.as_of_date})"

    @property
    def repository(self) -> DataRepository:
        """The underlying repository.

        Exposed for diagnostics and for the leakage tests, which need to compare
        a view's output against an unrestricted read. Production code should not
        reach through it - doing so discards the guarantee the view exists for.
        """
        return self._repository

    def as_of(self, as_of_date: date) -> PointInTimeView:
        """Re-freeze at a different date.

        Rebased on the underlying repository rather than nested, so views cannot
        stack into a chain whose effective cut is hard to reason about.
        """
        return PointInTimeView(self._repository, as_of_date)

    # -- helpers ------------------------------------------------------------

    def _window(
        self, table: str, start_date: date | None, end_date: date | None
    ) -> tuple[date | None, date | None]:
        return clamp_window(table, start_date, end_date, self.as_of_date)

    def _mask_actuals(self, table: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Null after-the-fact columns on forward-dated rows.

        A known-in-advance table may still carry actuals. The promotion schedule
        for next month is knowable; the spend it will eventually book is not.
        Returning the row for its mechanics while blanking its outcomes is the
        honest representation of what a planner actually holds.
        """
        columns = [c for c in actuals_columns_of(table) if c in frame.columns]
        if not columns or frame.empty:
            return frame

        date_column = date_column_of(table)
        if date_column not in frame.columns:
            return frame

        future = pd.to_datetime(frame[date_column]).dt.date > self.as_of_date
        if not bool(future.any()):
            return frame

        masked = frame.copy()
        masked.loc[future, columns] = pd.NA
        logger.debug(
            "point_in_time.actuals_masked",
            table=table,
            columns=columns,
            rows=int(future.sum()),
        )
        return masked

    # -- dimensions (static: passed through) --------------------------------

    def get_products(self, **kwargs: Any) -> pd.DataFrame:
        return self._repository.get_products(**kwargs)

    def get_stores(self, **kwargs: Any) -> pd.DataFrame:
        return self._repository.get_stores(**kwargs)

    def get_customers(self, **kwargs: Any) -> pd.DataFrame:
        return self._repository.get_customers(**kwargs)

    def get_product_relationships(self, **kwargs: Any) -> pd.DataFrame:
        return self._repository.get_product_relationships(**kwargs)

    def get_calendar(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        # Known in advance: holidays and the financial calendar are not cut.
        return self._repository.get_calendar(
            start_date=start_date, end_date=end_date, validate=validate
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
        start, end = self._window("sales_daily", start_date, end_date)
        return self._repository.get_sales(
            product_ids=product_ids,
            store_ids=store_ids,
            region=region,
            channel=channel,
            start_date=start,
            end_date=end,
            columns=columns,
            max_rows=max_rows,
            validate=validate,
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
        start, end = self._window("pricing", start_date, end_date)
        return self._repository.get_pricing(
            product_ids=product_ids,
            store_ids=store_ids,
            start_date=start,
            end_date=end,
            max_rows=max_rows,
            validate=validate,
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
        start, end = self._window("inventory", start_date, end_date)
        return self._repository.get_inventory(
            product_ids=product_ids,
            store_ids=store_ids,
            start_date=start,
            end_date=end,
            max_rows=max_rows,
            validate=validate,
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
        start, end = self._window("promotions", start_date, end_date)
        frame = self._repository.get_promotions(
            product_ids=product_ids,
            store_ids=store_ids,
            promotion_type=promotion_type,
            start_date=start,
            end_date=end,
            max_rows=max_rows,
            validate=validate,
        )
        return self._mask_actuals("promotions", frame)

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
        start, end = self._window("trade_promotions", start_date, end_date)
        return self._repository.get_trade_promotions(
            product_ids=product_ids,
            retailer=retailer,
            region=region,
            start_date=start,
            end_date=end,
            max_rows=max_rows,
            validate=validate,
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
        start, end = self._window("competitor_pricing", start_date, end_date)
        return self._repository.get_competitor_prices(
            product_ids=product_ids,
            competitor_ids=competitor_ids,
            start_date=start,
            end_date=end,
            max_rows=max_rows,
            validate=validate,
        )

    # -- generic ------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Ad-hoc SQL. **Not** restricted by the as-of date.

        Deliberately not clamped, because rewriting arbitrary SQL to add a date
        predicate is not something to attempt by string manipulation - it would
        be wrong on the first subquery or window function it met, and wrong
        silently.

        The consequence is that this method is unsafe for feature engineering,
        and the honest response is to say so rather than pretend otherwise.
        Feature builders use the typed ``get_*`` methods. Step 14's Text-to-SQL
        agent, which is what this exists for, answers questions about the past
        rather than assembling training features.
        """
        logger.warning(
            "point_in_time.unrestricted_query",
            as_of_date=str(self.as_of_date),
            reason="execute_query bypasses the as-of cut; not for feature building",
        )
        return self._repository.execute_query(sql, parameters, max_rows=max_rows)

    # -- introspection ------------------------------------------------------

    def list_tables(self) -> list[str]:
        return self._repository.list_tables()

    def describe_table(self, table: str) -> pd.DataFrame:
        return self._repository.describe_table(table)

    def dataset_version(self) -> str:
        """Dataset version, qualified by the as-of date.

        Two feature sets built from the same data at different as-of dates are
        different artifacts, and the version string has to say so - otherwise
        MLflow in Step 12 would record them as identical inputs.
        """
        return f"{self._repository.dataset_version()}@{self.as_of_date.isoformat()}"

    def health_check(self) -> tuple[bool, str]:
        healthy, detail = self._repository.health_check()
        return healthy, f"{detail} (as-of {self.as_of_date})"

    def availability(self, table: str) -> Availability:
        """Classification governing this table, for diagnostics and tests."""
        return availability_of(table)
