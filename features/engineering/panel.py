"""The shift discipline - one place, so it cannot be forgotten.

Every temporal feature in this project passes through :func:`shifted_group`.
That is deliberate and it is the single most important line of defence against
leakage inside feature engineering.

The rule: a feature used to predict day *t* must be computed from data strictly
before *t*. A 7-day rolling mean that includes today's sales is not a predictor,
it is a partial answer - and a model given it will look excellent in backtest
and be useless in production, because at prediction time today's sales do not
exist yet.

The mistake is easy to make and invisible once made. ``df.groupby(key).rolling(7)``
reads perfectly well and silently includes the current row. Doing the shift in
one shared helper, rather than at each of the twenty call sites, means it is
right everywhere or wrong everywhere - and the tests pin it as right.

The panel grain throughout is ``(product_id, store_id, date)``, and every
function here assumes rows are sorted within that grain. :func:`prepare_panel`
guarantees it.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: The keys identifying one time series within the panel. Lags and rolling
#: windows are computed within a group, never across - otherwise the last day of
#: one product-store would leak into the first day of the next.
PANEL_KEYS: tuple[str, str] = ("product_id", "store_id")
DATE_KEY = "date"


#: Columns that are arithmetic functions of the target on the same row.
#: ``revenue = units x selling_price``, so a model given revenue and price can
#: recover units exactly - it is the target wearing a hat. Dropped from every
#: feature panel by default; see :func:`drop_target_derived`.
TARGET_DERIVED: tuple[str, ...] = (
    "revenue",
    "cost",
    "gross_profit",
    "sold_units",
    "closing_inventory",
    "inventory_days",
    "promotion_units",
)


def as_bool(series: pd.Series) -> pd.Series:
    """Coerce a possibly-null object series to a clean boolean column.

    Shifting or left-joining a bool column yields object dtype with ``NaN``.
    Pandas 3 removes the implicit downcast on ``fillna``, so the cast is made
    explicit here rather than repeated - with a deprecation warning - at every
    call site.
    """
    return series.astype(object).where(series.notna(), False).astype(bool)


def drop_target_derived(frame: pd.DataFrame, *, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Remove columns that encode the target on the same row.

    Done centrally rather than left to each consumer. Expecting every future
    model to remember that ``revenue`` is really ``units`` in disguise is how
    one of them eventually forgets, and a model that leaks its target reports
    excellent metrics right up until it is deployed.
    """
    present = [c for c in (*TARGET_DERIVED, *extra) if c in frame.columns]
    return frame.drop(columns=present) if present else frame


def prepare_panel(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str] = PANEL_KEYS,
    date_column: str = DATE_KEY,
) -> pd.DataFrame:
    """Normalise a frame into a sorted panel ready for temporal features.

    Sorting is not cosmetic. ``groupby().shift()`` operates on row order, so an
    unsorted frame produces lags that are silently wrong - pointing at whatever
    row happened to precede, which may be a later date.
    """
    missing = [c for c in (*keys, date_column) if c not in frame.columns]
    if missing:
        raise KeyError(f"panel is missing required columns: {missing}")

    panel = frame.copy()
    panel[date_column] = pd.to_datetime(panel[date_column])
    return panel.sort_values([*keys, date_column]).reset_index(drop=True)


def shifted_group(
    panel: pd.DataFrame,
    column: str,
    *,
    periods: int = 1,
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.Series:
    """``column`` shifted forward by ``periods`` within each panel group.

    ``periods=1`` yields "the value as at the previous day", which is the most
    recent value genuinely known when predicting today.

    Every temporal primitive builds on this. Nothing in ``features/`` should
    call ``groupby().shift()`` directly - going through here is what makes the
    shift auditable in one place.
    """
    if periods < 1:
        raise ValueError(
            f"periods must be >= 1 to stay point-in-time correct, got {periods}. "
            f"A zero or negative shift would let the current or a future value "
            f"into a feature used to predict the current value."
        )
    return panel.groupby(list(keys), observed=True, sort=False)[column].shift(periods)


def rolling_on_shifted(
    panel: pd.DataFrame,
    column: str,
    *,
    window: int,
    statistic: str = "mean",
    min_periods: int | None = None,
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.Series:
    """Rolling statistic over the window *ending yesterday*.

    Shifts by one period first, then rolls. That ordering is the whole point: a
    plain ``rolling(7)`` includes the current row, so ``rolling_7_sales`` would
    contain one seventh of the very number the model is trying to predict.

    ``min_periods`` defaults to 1, so the start of a series produces a partial
    average rather than ``NaN``. That is a deliberate trade: a new product-store
    would otherwise have no usable history for its first month, and dropping
    those rows biases training toward established pairs.
    """
    grouped = panel.groupby(list(keys), observed=True, sort=False)[column]
    shifted = grouped.shift(1)
    rolling = shifted.groupby([panel[k] for k in keys], observed=True, sort=False).rolling(
        window=window, min_periods=min_periods or 1
    )

    result = getattr(rolling, statistic)()
    # `.rolling()` on a groupby prepends the group keys to the index; drop them
    # so the result aligns positionally with the panel it came from.
    return result.reset_index(level=list(range(len(keys))), drop=True).sort_index()


def expanding_on_shifted(
    panel: pd.DataFrame,
    column: str,
    *,
    statistic: str = "mean",
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.Series:
    """Expanding statistic over all history *before* the current row.

    Used for "historical average price" and similar: everything known so far,
    excluding today.
    """
    grouped = panel.groupby(list(keys), observed=True, sort=False)[column]
    shifted = grouped.shift(1)
    expanding = shifted.groupby([panel[k] for k in keys], observed=True, sort=False).expanding(
        min_periods=1
    )

    result = getattr(expanding, statistic)()
    return result.reset_index(level=list(range(len(keys))), drop=True).sort_index()


def days_since_flag(
    panel: pd.DataFrame,
    flag_column: str,
    *,
    keys: Sequence[str] = PANEL_KEYS,
    date_column: str = DATE_KEY,
) -> pd.Series:
    """Days since ``flag_column`` was last true, counting only prior days.

    Shifts first, so a row where the flag is true today reports the gap since
    the *previous* occurrence rather than zero. Reporting zero would encode
    today's flag value into a feature that is supposed to summarise the past -
    a subtle leak, since the flag itself is often what a model is predicting
    around.

    ``NaN`` before the first occurrence, which is honest: "never happened yet"
    is not the same as "happened a very long time ago", and callers decide how
    to fill it.
    """
    shifted_flag = as_bool(shifted_group(panel, flag_column, periods=1, keys=keys))
    dates = panel[date_column]

    # Forward-fill the date of the most recent prior occurrence within each group.
    marked = dates.where(shifted_flag)
    last_seen = marked.groupby([panel[k] for k in keys], observed=True, sort=False).ffill()

    return (dates - last_seen).dt.days


def rolling_count_of_flag(
    panel: pd.DataFrame,
    flag_column: str,
    *,
    window: int,
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.Series:
    """How many times a flag was true in the window ending yesterday."""
    numeric = flag_column + "__numeric"
    working = panel.assign(**{numeric: panel[flag_column].astype(float)})
    return rolling_on_shifted(working, numeric, window=window, statistic="sum", keys=keys)


def pct_change_on_shifted(
    panel: pd.DataFrame,
    column: str,
    *,
    periods: int = 1,
    keys: Sequence[str] = PANEL_KEYS,
) -> pd.Series:
    """Percentage change against the value ``periods`` days ago.

    Uses the *current* value against a past one, so this is legitimate only for
    columns knowable at prediction time - price, for instance, which is set in
    advance. Applying it to sales would compare today's demand to last week's
    and hand the model the answer.
    """
    previous = shifted_group(panel, column, periods=periods, keys=keys)
    return (panel[column] - previous) / previous.replace(0.0, pd.NA)
