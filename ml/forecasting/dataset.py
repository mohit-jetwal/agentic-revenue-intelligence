"""The horizon dataset: the piece that makes this a forecaster and not a nowcast.

Step 4's baseline predicts units at date *D* using features at *D*, including
``lag_1_units``. That is legitimate for a historical counterfactual, where
yesterday is known. It is invalid for forecasting: standing at as-of *T*
predicting *T+30*, *T+29*'s sales do not exist yet.

So a training row here is **(origin t, horizon step h, target = units at t+h)**,
and every feature is placed by asking one question - *is this knowable at t?*
Step 3's availability classes already answer it:

===========================  ============  ========================================
Feature family               Sourced at    Why that is legitimate
===========================  ============  ========================================
lags, rollings, dynamics     origin ``t``  ``sales_daily`` is OBSERVED
competitor price and gap     origin ``t``  ``competitor_pricing`` is OBSERVED
price/promotion history      origin ``t``  derived from OBSERVED sales
calendar, festival, season   target        ``calendar`` is KNOWN_IN_ADVANCE
planned promotion            target        ``promotions`` is KNOWN_IN_ADVANCE
planned price                target        ``pricing`` is KNOWN_IN_ADVANCE
product/store attributes     either        STATIC
``horizon_step`` itself      -             lets one model span every horizon
===========================  ============  ========================================

Target-side columns carry an ``h_`` prefix, so the split is visible in the
feature names and in the importance table rather than being a fact you have to
remember.

**Train/serve symmetry is the property that matters most here.** Training reads
target-side features from history; inference reads them from a future scaffold.
Two code paths computing what must be one thing is the classic source of silent
skew, so both go through :func:`target_side_features` and a test asserts they
agree row-for-row. Note in particular that planned price comes from the
``pricing`` table in *both* paths, never from ``sales_daily`` - the sales table
is OBSERVED and simply does not exist over a forecast horizon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from data.repositories.point_in_time import PointInTimeView
from features.engineering import FeatureEngineer, FeatureRequest
from features.engineering.demand import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    build_demand_features,
    mask_censored,
)
from features.engineering.entity import drop_high_cardinality
from features.engineering.promotion import add_promotion_features, expand_promotion_calendar
from features.engineering.temporal import (
    add_festival_proximity,
    add_time_features,
    join_calendar_features,
)
from ml.baseline.training import EXCLUDED_FROM_FEATURES, SUPPLY_FEATURES
from ml.forecasting.config import ForecastConfig
from ml.forecasting.sampling import SeriesSample

logger = get_logger(__name__)

TARGET = "units"
ORIGIN_DATE = "origin_date"
TARGET_DATE = "target_date"
HORIZON_STEP = "horizon_step"
KEYS = ("product_id", "store_id")

#: Prefix marking a column as sourced from the *target* date rather than the
#: origin. Visible in feature importance, which is where a mistake would
#: otherwise hide.
TARGET_PREFIX = "h_"

#: Never features here, on top of what Step 4 already excludes.
#:
#: ``time_index`` is anchored to the *frame's own minimum*
#: (``features/engineering/temporal.py:81``), so the same calendar date gets a
#: different value at training and at serving time - a silent train/serve skew
#: that no amount of leakage testing on the training frame would reveal.
#:
#: ``year`` cannot be extrapolated: every forecast is for a year the model has
#: either seen (and will overfit to) or never seen (and cannot place).
FORECAST_EXCLUDED: frozenset[str] = frozenset(
    {
        "time_index",
        "year",
        "financial_year",
        ORIGIN_DATE,
        TARGET_DATE,
        "date",
        TARGET,
        "units_uncensored",
        "promotion_id",
        "price_change_reason",
        # Hive partition key (`part=202301`). A storage-layout artifact that a
        # tree will happily split on as a coarse date proxy - which both leaks
        # calendar position into the origin side and cannot be reproduced for a
        # future date that has no partition yet.
        "part",
        # Supply columns Step 4's SUPPLY_FEATURES does not name, because its
        # panel never carried them. `received_units` is inbound replenishment;
        # `sold_units_lag_1` is yesterday's sales re-derived from the inventory
        # table, which both duplicates `lag_1_units` and reintroduces the
        # censoring the masking above exists to remove.
        "received_units",
        "sold_units_lag_1",
    }
)

#: Calendar columns worth carrying to the target date.
_CALENDAR_COLUMNS = (
    "holiday_flag", "festival_flag", "season", "weekend_flag",
    "financial_month", "financial_quarter",
)

#: Planned-price columns from the ``pricing`` table (KNOWN_IN_ADVANCE).
_PLANNED_PRICE_COLUMNS = ("regular_price", "selling_price", "discount_percentage")


@dataclass(frozen=True)
class HorizonDataset:
    """One row per (series, origin, horizon step), ready to fit."""

    frame: pd.DataFrame
    feature_names: list[str]
    target: str = TARGET
    #: Rows dropped by each filter, so the cost of every exclusion is visible.
    excluded: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - sklearn convention
        return self.frame[self.feature_names]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]

    def origins(self) -> pd.Series:
        return self.frame[ORIGIN_DATE]

    def describe(self) -> str:
        return (
            f"{len(self.frame):,} rows | {len(self.feature_names)} features | "
            f"origins {self.frame[ORIGIN_DATE].min()}..{self.frame[ORIGIN_DATE].max()} | "
            f"h {self.frame[HORIZON_STEP].min()}-{self.frame[HORIZON_STEP].max()}"
        )


# -- origin side ------------------------------------------------------------


def build_history(
    repository: DataRepository,
    config: ForecastConfig,
    sample: SeriesSample,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Build the historical panel the origin-side features come from.

    Deliberately does **not** reuse ``ml.baseline.pipeline.build_panel``: that
    function flattens a pair sample into independent product and store filters,
    loading roughly seven times the requested series. See
    :mod:`ml.forecasting.sampling`.
    """
    calendar = repository.get_calendar()
    available = pd.to_datetime(calendar["date"]).dt.date
    start = start_date or available.min()
    end = end_date or available.max()

    view = repository.as_of(end)
    engineer = FeatureEngineer(view)

    started = time.perf_counter()
    panel = engineer.build(
        FeatureRequest(
            start_date=start,
            end_date=end,
            product_ids=sample.product_ids,
            store_ids=sample.store_ids,
            # Inventory is loaded so `stockout_flag` is available for censoring
            # decisions; the columns themselves are dropped from the feature set
            # further down.
            inventory=True,
            promotion=True,
            include_promotion_spend=False,
            warmup_days=config.origins.warmup_days,
        )
    )
    # The step that makes `n_series` mean what it says.
    panel = sample.restrict(panel)

    if config.target_handling.mask_censored_lags and "stockout_flag" in panel.columns:
        panel = _recompute_demand_uncensored(panel, config)

    logger.info(
        "forecast.history_built",
        rows=len(panel),
        columns=len(panel.columns),
        series=len(sample),
        seconds=round(time.perf_counter() - started, 1),
    )
    return panel


def _recompute_demand_uncensored(panel: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Rebuild the demand block from censoring-masked sales.

    ``mask_censored`` exists in Step 3 (``features/engineering/demand.py``) and
    has never been used - its docstring defers the decision to the model. Step 5
    is the model that should take it.

    Without this, a stockout depresses ``lag_7``/``rolling_28`` for the next
    eight weeks, so the model learns a supply failure as a demand signal through
    the back door even though the stockout rows themselves were excluded from
    training. The masked series carries NaN on stockout days, and the rolling
    helpers skip NaN rather than treating it as zero - which is the difference
    between "we do not know what demand was" and "demand was nothing".
    """
    masked = mask_censored(panel, column=TARGET, stockout_column="stockout_flag")

    demand_columns = [
        c for c in masked.columns
        if c.startswith(("lag_", "rolling_"))
        or c in {"demand_momentum", "demand_volatility", "demand_trend_28"}
    ]
    working = masked.drop(columns=demand_columns, errors="ignore")

    rebuilt = build_demand_features(
        working.rename(columns={TARGET: "_observed", f"{TARGET}_uncensored": TARGET}),
        column=TARGET,
        lags=DEFAULT_LAGS,
        windows=DEFAULT_WINDOWS,
        keys=KEYS,
    )
    # Put the observed target back; only the *features* are built on the masked
    # series. The target must stay observed, because that is what actually sold.
    rebuilt[TARGET] = rebuilt.pop("_observed")
    return rebuilt


def origin_side_features(panel: pd.DataFrame) -> list[str]:
    """Columns knowable at the origin date."""
    return [
        column
        for column in panel.columns
        if column not in EXCLUDED_FROM_FEATURES
        and column not in SUPPLY_FEATURES
        and column not in FORECAST_EXCLUDED
        and not column.startswith(TARGET_PREFIX)
        # Calendar and promotion state at the *origin* say nothing useful about
        # the target date and would be confused with their h_ counterparts.
        and column not in _ORIGIN_CALENDAR_NOISE
    ]


#: Origin-date calendar/promotion columns. Dropped rather than kept: their
#: target-date counterparts carry the same information about the day being
#: forecast, and keeping both invites the model to read the origin's calendar as
#: if it were the target's.
_ORIGIN_CALENDAR_NOISE: frozenset[str] = frozenset(
    {
        "day_of_week", "week_of_year", "month", "quarter", "day_of_month",
        "day_of_year", "weekend_flag", "holiday_flag", "festival_flag", "season",
        "financial_month", "financial_quarter",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
        "days_to_festival", "days_since_festival",
        "promotion_flag", "promotion_discount", "promotion_duration",
        "days_into_promotion", "days_until_promotion_end", "days_to_next_promotion",
        "promotion_type", "display_flag", "bundle_flag", "promotion_intensity",
        "promotion_spend", "promotion_units",
    }
)


# -- target side (shared by training and inference) -------------------------


def target_side_features(
    view: PointInTimeView,
    pairs: pd.DataFrame,
    dates: pd.Series | pd.DatetimeIndex,
    *,
    include_promotion_spend: bool = False,
) -> pd.DataFrame:
    """Features for the date being forecast, from KNOWN_IN_ADVANCE tables only.

    **The single most important function for train/serve consistency.** Training
    calls it over historical target dates; inference calls it over future ones.
    Because it is one function reading one set of tables, the two cannot drift -
    and :func:`tests.forecasting.test_leakage` asserts they produce identical
    vectors for the same (series, date).

    Reads ``calendar``, ``promotions`` and ``pricing``. It deliberately does not
    touch ``sales_daily``: that table is OBSERVED, so over a real forecast
    horizon it is empty, and using it in training would create a feature the
    serving path could never reproduce.
    """
    dates = pd.to_datetime(pd.Series(list(dates))).drop_duplicates().sort_values()
    if dates.empty or pairs.empty:
        return pd.DataFrame()

    grid = pairs[list(KEYS)].merge(pd.DataFrame({"date": dates}), how="cross")

    start = dates.min().date()
    end = dates.max().date()

    calendar = view.get_calendar(start_date=start, end_date=end)
    grid = add_time_features(grid, cyclical=True)
    grid = join_calendar_features(grid, calendar, columns=_CALENDAR_COLUMNS)
    grid = add_festival_proximity(grid, calendar)

    promotions = view.get_promotions(
        product_ids=pairs["product_id"].unique().tolist(),
        store_ids=pairs["store_id"].unique().tolist(),
        start_date=start - timedelta(days=120),
        end_date=end + timedelta(days=120),
        max_rows=5_000_000,
    )
    if not promotions.empty:
        grid = add_promotion_features(
            grid, promotions, keys=KEYS, include_spend=include_promotion_spend
        )

    pricing = view.get_pricing(
        product_ids=pairs["product_id"].unique().tolist(),
        store_ids=pairs["store_id"].unique().tolist(),
        start_date=start,
        end_date=end,
        max_rows=20_000_000,
    )
    if not pricing.empty:
        columns = ["date", *KEYS, *(c for c in _PLANNED_PRICE_COLUMNS if c in pricing.columns)]
        planned = pricing[columns].copy()
        planned["date"] = pd.to_datetime(planned["date"])
        grid = grid.merge(planned, on=["date", *KEYS], how="left")

    grid = drop_high_cardinality(grid)
    # Dropped *before* prefixing, so the exclusion list is checked against the
    # real column name. Prefixing first would let `year` through as `h_year`.
    grid = grid.drop(
        columns=[c for c in grid.columns if c in FORECAST_EXCLUDED and c != "date"],
        errors="ignore",
    )

    # Prefix everything that is not a key, so the origin/target split is legible
    # downstream.
    renames = {
        column: f"{TARGET_PREFIX}{column}"
        for column in grid.columns
        if column not in {"date", *KEYS}
    }
    return grid.rename(columns=renames).rename(columns={"date": TARGET_DATE})


# -- the self-join ----------------------------------------------------------


def build_horizon_dataset(
    history: pd.DataFrame,
    view: PointInTimeView,
    config: ForecastConfig,
    sample: SeriesSample,
    *,
    seed: int | None = None,
    horizon_features_from_target: bool = False,
) -> HorizonDataset:
    """Turn a historical panel into (origin, horizon, target) training rows.

    ``horizon_features_from_target`` is a **deliberate defect switch**, used by
    one test to plant the exact bug this design exists to prevent - origin-side
    features read from the target row. It makes the leakage test falsifiable;
    without it, a test that always passes proves nothing. It is never true in
    production.
    """
    if history.empty:
        return HorizonDataset(frame=history, feature_names=[])

    rng = np.random.default_rng(seed if seed is not None else config.sampling.seed)
    working = history.copy()
    working["date"] = pd.to_datetime(working["date"])

    excluded: dict[str, int] = {"panel_rows": len(working)}

    # -- choose origins ----------------------------------------------------
    all_dates = np.sort(working["date"].unique())
    # A stride over the *calendar*, not over row position, so every series uses
    # the same origin dates and the folds line up across series.
    origin_dates = set(all_dates[:: config.origins.stride_days])

    origins = working[working["date"].isin(origin_dates)].copy()
    if not config.target_handling.exclude_stockout_origins and "stockout_flag" in origins:
        pass  # kept deliberately; see the module docstring for target vs origin
    elif "stockout_flag" in origins:
        origins = origins[~origins["stockout_flag"].astype(bool)]

    # A row is only usable as an origin once its longest lag is defined,
    # otherwise the model trains on rows whose history is mostly NaN.
    if "lag_364_units" in origins.columns:
        before = len(origins)
        origins = origins[origins["lag_364_units"].notna()]
        excluded["origins_without_history"] = before - len(origins)

    origin_features = origin_side_features(working)
    origins = origins[[*KEYS, "date", *origin_features]].rename(columns={"date": ORIGIN_DATE})
    excluded["origins"] = len(origins)

    # -- draw horizon steps ------------------------------------------------
    # Random rather than a fixed grid. With a fixed grid the model's splits on
    # `horizon_step` are piecewise-constant, which shows up as a visible
    # staircase in the daily forecast path - and the path is a deliverable.
    repeats = config.origins.horizons_per_origin
    expanded = origins.loc[origins.index.repeat(repeats)].reset_index(drop=True)
    expanded[HORIZON_STEP] = rng.integers(1, config.max_horizon + 1, size=len(expanded))
    expanded[TARGET_DATE] = expanded[ORIGIN_DATE] + pd.to_timedelta(
        expanded[HORIZON_STEP], unit="D"
    )
    # The same (origin, h) drawn twice carries no extra information.
    expanded = expanded.drop_duplicates(subset=[*KEYS, ORIGIN_DATE, HORIZON_STEP])

    # -- attach the target -------------------------------------------------
    target_columns = [*KEYS, "date", TARGET]
    if "stockout_flag" in working.columns:
        target_columns.append("stockout_flag")
    targets = working[target_columns].rename(
        columns={"date": TARGET_DATE, "stockout_flag": "target_stockout_flag"}
    )
    frame = expanded.merge(targets, on=[*KEYS, TARGET_DATE], how="inner")
    excluded["after_target_join"] = len(frame)

    # -- target-side features ----------------------------------------------
    source_date = ORIGIN_DATE if horizon_features_from_target else TARGET_DATE
    target_side = target_side_features(
        view, sample.pairs, frame[source_date].drop_duplicates()
    )
    if not target_side.empty:
        if horizon_features_from_target:
            # The planted defect: join target-side features on the ORIGIN date
            # while labelling them as the target's.
            target_side = target_side.rename(columns={TARGET_DATE: ORIGIN_DATE})
            frame = frame.merge(target_side, on=[*KEYS, ORIGIN_DATE], how="left")
        else:
            frame = frame.merge(target_side, on=[*KEYS, TARGET_DATE], how="left")

    # -- censored targets ---------------------------------------------------
    if config.target_handling.exclude_stockout_targets and "target_stockout_flag" in frame:
        before = len(frame)
        frame = frame[~frame["target_stockout_flag"].astype(bool)]
        excluded["stockout_targets"] = before - len(frame)
    frame = frame.drop(columns=["target_stockout_flag"], errors="ignore")

    before = len(frame)
    frame = frame[frame[TARGET].notna()]
    excluded["missing_target"] = before - len(frame)

    feature_names = [
        column
        for column in frame.columns
        if column not in FORECAST_EXCLUDED and column not in KEYS
    ]
    frame = frame.sort_values([ORIGIN_DATE, *KEYS, HORIZON_STEP]).reset_index(drop=True)
    excluded["retained"] = len(frame)

    dataset = HorizonDataset(frame=frame, feature_names=feature_names, excluded=excluded)
    logger.info("forecast.horizon_dataset_built", **excluded, features=len(feature_names))
    return dataset


# -- inference scaffold -----------------------------------------------------


def latest_known_date(view: PointInTimeView) -> date:
    """Last date for which KNOWN_IN_ADVANCE data actually exists.

    Not a theoretical bound. The generated dataset stops on 2025-12-31 for the
    calendar, the promotion schedule and the price plan alike, so a horizon
    reaching past it has no target-side features at all. Forecasting anyway
    would mean assuming "no promotion planned", which biases those days low and
    is exactly the kind of quiet fabrication the brief forbids.
    """
    calendar = view.get_calendar()
    return pd.to_datetime(calendar["date"]).dt.date.max()


def build_future_scaffold(
    view: PointInTimeView,
    history: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    as_of: date,
    horizon_days: int,
) -> pd.DataFrame:
    """Rows for dates that have not happened yet.

    Deliberately not inside :class:`FeatureEngineer`. That class starts from
    clamped ``sales_daily`` and left-joins everything onto it, so it structurally
    cannot emit a row for a date with no sales - and bending it to do so would
    put Step 3's leakage guarantees at risk for the sake of one caller.

    Origin-side features are taken from the single most recent history row per
    series (the as-of row); target-side features come from
    :func:`target_side_features`, the same function training uses.
    """
    if history.empty or pairs.empty:
        return pd.DataFrame()

    working = history.copy()
    working["date"] = pd.to_datetime(working["date"])
    as_of_ts = pd.Timestamp(as_of)

    at_origin = working[working["date"] <= as_of_ts]
    if at_origin.empty:
        return pd.DataFrame()
    # The as-of row per series: everything the model knows at forecast time.
    at_origin = (
        at_origin.sort_values("date")
        .groupby(list(KEYS), as_index=False, observed=True)
        .tail(1)
    )

    origin_features = origin_side_features(working)
    origins = at_origin[[*KEYS, "date", *origin_features]].rename(
        columns={"date": ORIGIN_DATE}
    )

    steps = pd.DataFrame({HORIZON_STEP: range(1, horizon_days + 1)})
    frame = origins.merge(steps, how="cross")
    frame[TARGET_DATE] = as_of_ts + pd.to_timedelta(frame[HORIZON_STEP], unit="D")

    target_side = target_side_features(view, pairs, frame[TARGET_DATE].drop_duplicates())
    if not target_side.empty:
        frame = frame.merge(target_side, on=[*KEYS, TARGET_DATE], how="left")

    return frame.sort_values([*KEYS, HORIZON_STEP]).reset_index(drop=True)
