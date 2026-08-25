"""When is each table's data actually knowable?

This module answers the question that makes point-in-time correctness correct,
rather than merely defensive.

The naive reading of "as-of date" is *clamp everything to D*. That is wrong, and
wrong in a way that would cripple forecasting. A demand model predicting next
week legitimately knows next week's promotion calendar - promotions are planned
and committed weeks ahead - and it certainly knows next week's public holidays.
Clamping those to D means the model learns a promotion ran only after it ended,
which is not the information a planner actually has.

The opposite error is worse: letting *observed* data past D leaks the future and
produces a model that backtests beautifully and fails in production.

So availability is a **per-table property**, and each table is classified here:

* ``OBSERVED`` - recorded after the fact. Sales, inventory, competitor prices.
  Visible only up to and including the as-of date.
* ``KNOWN_IN_ADVANCE`` - planned and committed before it happens. The calendar,
  the promotion schedule, the price file. Visible over the full horizon.
* ``STATIC`` - no meaningful time axis. Dimensions and reference data.

``KNOWN_IN_ADVANCE`` is the dangerous class, because it is the one that lets data
past the as-of date. Two safeguards apply. First, membership is deliberately
small and each entry is justified below. Second, a forward-dated row of an
otherwise-planned table can still carry columns that are *actuals* - a future
promotion's realised spend and units are not knowable - so those columns are
listed in :data:`ACTUALS_COLUMNS` and nulled beyond the as-of date.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum


class Availability(StrEnum):
    """How a table's rows become knowable over time."""

    #: Recorded after the event. Never visible beyond the as-of date.
    OBSERVED = "observed"
    #: Planned and committed ahead of time. Visible over the full horizon.
    KNOWN_IN_ADVANCE = "known_in_advance"
    #: No time dimension worth cutting on.
    STATIC = "static"


#: Table -> availability class. Every gold table must appear here; an unlisted
#: table is treated as OBSERVED by :func:`availability_of`, because guessing
#: wrong in that direction costs a little signal, and guessing wrong in the
#: other direction silently leaks the future.
TABLE_AVAILABILITY: dict[str, Availability] = {
    # -- observed -----------------------------------------------------------
    # Sales are recorded when they happen.
    "sales_daily": Availability.OBSERVED,
    "sales_transactions": Availability.OBSERVED,
    "sales_weekly": Availability.OBSERVED,
    "sales_monthly": Availability.OBSERVED,
    # Stock positions are counted, not planned.
    "inventory": Availability.OBSERVED,
    # Competitor prices are *observed* by shelf audit or scraping. We may know
    # our own price list in advance; we never know theirs.
    "competitor_pricing": Availability.OBSERVED,
    # Trade promotion rows carry actual spend, actual uplift and realised ROI.
    # Even though the plan exists ahead of time, this table is dominated by
    # after-the-fact columns, so the safe classification is observed.
    "trade_promotions": Availability.OBSERVED,
    # Input cost indices are published with a lag.
    "commodity_costs": Availability.OBSERVED,
    # -- known in advance ---------------------------------------------------
    # Holidays, festivals and the financial calendar are known years out.
    "calendar": Availability.KNOWN_IN_ADVANCE,
    # Promotion mechanics are agreed with retailers weeks ahead. A planner on
    # date D genuinely knows what is scheduled for D+14 - which is exactly why
    # `days_until_promotion_end` is a legitimate feature (brief section 18).
    "promotions": Availability.KNOWN_IN_ADVANCE,
    # Price files are set ahead of their effective date. A pricing manager on D
    # knows the list price that takes effect next month.
    "pricing": Availability.KNOWN_IN_ADVANCE,
    # -- static -------------------------------------------------------------
    "products": Availability.STATIC,
    "stores": Availability.STATIC,
    "customers": Availability.STATIC,
    "product_relationships": Availability.STATIC,
}

#: Columns that are actuals living on an otherwise-planned table. Beyond the
#: as-of date these are unknowable and get nulled, so a feature builder cannot
#: read next month's realised promotional spend off a row it was allowed to see
#: for its *schedule*.
ACTUALS_COLUMNS: dict[str, tuple[str, ...]] = {
    "promotions": ("promotion_spend", "promotion_units"),
}

#: The date column each table is cut on. Event tables span a window rather than
#: sitting on a single day, so they are cut on their start.
DATE_COLUMN: dict[str, str] = {
    "promotions": "start_date",
    "trade_promotions": "start_date",
}


def availability_of(table: str) -> Availability:
    """Classification for ``table``, defaulting to the safe class.

    An unknown table is treated as ``OBSERVED``. That default is deliberate: a
    new table added in a later step is invisible past the as-of date until
    someone consciously classifies it, so the failure mode of forgetting is lost
    signal rather than silent leakage.
    """
    return TABLE_AVAILABILITY.get(table, Availability.OBSERVED)


def date_column_of(table: str) -> str:
    """Name of the column an as-of cut applies to."""
    return DATE_COLUMN.get(table, "date")


def actuals_columns_of(table: str) -> tuple[str, ...]:
    """Columns to null beyond the as-of date on a known-in-advance table."""
    return ACTUALS_COLUMNS.get(table, ())


def clamp_window(
    table: str,
    start_date: date | None,
    end_date: date | None,
    as_of_date: date | None,
) -> tuple[date | None, date | None]:
    """Narrow a requested date window to what was knowable at ``as_of_date``.

    The single place the as-of rule is applied. Both the ``as_of_date`` keyword
    on the repository methods and :class:`~data.repositories.point_in_time.PointInTimeView`
    route through here, so the two cannot drift apart and start disagreeing
    about what the future is.
    """
    if as_of_date is None:
        return start_date, end_date

    if availability_of(table) is not Availability.OBSERVED:
        # Planned and static data is knowable ahead; the caller's own window
        # stands.
        return start_date, end_date

    capped_end = as_of_date if end_date is None else min(end_date, as_of_date)
    return start_date, capped_end
