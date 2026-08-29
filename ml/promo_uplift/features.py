"""Pre-treatment covariates (brief sections 5, 12, 25).

Every column this module produces must satisfy one rule: **it was determined
before the promotion started.** A causal adjustment set exists to close back-door
paths, and a variable measured after treatment closes nothing - it either blocks
the causal path being measured, or opens a new one.

Three failure modes this module is built to prevent.

**Mediator adjustment.** ``selling_price`` and ``discount_percentage`` are
*consequences* of the promotion, not causes of it. Conditioning on them holds the
price cut fixed across arms, which removes the largest channel through which a
promotion works. The estimate that survives is the mechanic alone, reported as
though it were the whole effect. This is the single most likely way to get a
plausible-looking wrong number here, and it is why those columns are in
:data:`POST_TREATMENT_FEATURES` rather than merely omitted.

**Rolling-window contamination.** A trailing 7-day mean computed on day five of a
promotion contains four promoted days. The covariate then carries the treatment
effect, the outcome model explains the outcome using it, and the estimated effect
shrinks toward zero. Every trailing feature here is therefore anchored at the
**event start**, not at the row's own date - see :func:`anchor_dates`.

**Collider adjustment.** ``stockout_flag`` is caused by the promotion (demand
outruns the reorder policy) *and* correlated with demand. Conditioning on it
opens a path that was closed. It is excluded from the adjustment set entirely,
even though it is used to filter rows.

What is *deliberately* included is the confounder itself: the category's seasonal
position on the date. In the platform generator, promotion timing is drawn with
weights ``exp(targeting * 2 * seasonal)``, so seasonality is the mechanism by
which treatment and outcome are related without one causing the other. An
adjustment set that omitted it would leave the main back-door path wide open.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import InsufficientPrePeriodError
from ml.promo_uplift.treatment import DATE, KEYS, RowRole

logger = get_logger(__name__)

#: Feature version, bumped when the covariate set changes. Travels on every
#: result so two estimates can be told apart when the adjustment set moved.
COVARIATE_VERSION = "promo_uplift_covariates_v1"

#: Never covariates. Each is either the outcome, a function of the outcome, or a
#: consequence of treatment. The distinction matters for *why* each is here:
#:
#: * outcome and its arithmetic: ``units``, ``revenue``, ``cost``, ``gross_profit``
#: * mediators of treatment: ``selling_price``, ``discount_percentage``,
#:   ``promotion_*`` - conditioning on these measures the mechanic alone
#: * colliders: ``stockout_flag`` - caused by treatment, correlated with demand
#: * bookkeeping: roles, ids, ground truth
POST_TREATMENT_FEATURES: frozenset[str] = frozenset(
    {
        "units",
        "revenue",
        "cost",
        "gross_profit",
        "selling_price",
        "discount_percentage",
        "promotion_id",
        "promotion_flag",
        "promotion_type",
        "promotion_spend",
        "promotion_units",
        "promotion_intensity",
        "promotion_duration",
        "days_into_promotion",
        "days_until_promotion_end",
        "display_flag",
        "bundle_flag",
        "stockout_flag",
        "inventory_available",
        "opening_inventory",
        "closing_inventory",
        "inventory_days",
        "sold_units",
        "treatment",
        "role",
        "control_origin",
        "_days_to_event",
        "_anchor_date",
        # Simulation truth, from both generators.
        "latent_units",
        "mean_demand",
        "lost_units",
        "observed_units",
        "true_lambda_untreated",
        "true_lambda_treated",
        "true_effect_units",
        "true_uplift_pct",
        "true_segment_uplift",
    }
)

#: Identifier columns: carried through the frame but never fitted on. A model
#: given raw ids memorises listings instead of learning what makes them
#: promotable, and it cannot generalise to a listing it has not seen.
IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"date", "product_id", "store_id", "promotion_id"})

_TRAILING_WINDOWS: tuple[int, ...] = (7, 14, 28, 56)
_LAGS: tuple[int, ...] = (1, 7, 14, 28)


@dataclass
class CovariateFrame:
    """Covariates ``X``, treatment ``t``, outcome ``y``, and their provenance."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    outcome: str = "units"
    #: Groups of features, for the report and for interpretability output.
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    version: str = COVARIATE_VERSION

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[list(self.feature_names)]

    @property
    def t(self) -> np.ndarray:
        return self.frame["treatment"].to_numpy(dtype=bool)

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.outcome].to_numpy(dtype=float)

    def numeric_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.feature_names if n not in self.categorical_names)


def anchor_dates(panel: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """The date each row's trailing covariates are measured as of.

    For a **control** row, its own date. For a **treated** row, the date the
    event started - not the row's own date. On day five of a promotion the
    previous four days are already treated, so a covariate anchored at the row
    would contain the effect being estimated. Anchoring at the event start gives
    every row of one promotion a single pre-treatment information set, which is
    also what makes those rows comparable to each other.

    Note there is **no subtraction of a day here**. The trailing statistics in
    :func:`_trailing_features` are computed on ``shift(1)``, so the value indexed
    at date ``D`` already covers ``D-1`` and earlier. Subtracting a day again
    would shift the window twice and quietly discard the most recent - and most
    informative - day of history. The exclusion of the current day lives in one
    place, and this is not it.
    """
    anchor = pd.to_datetime(panel[DATE])

    if events.empty or "promotion_id" not in panel.columns:
        return anchor

    starts = pd.to_datetime(events.set_index("promotion_id")["start_date"])

    treated = panel["role"] == RowRole.TREATED if "role" in panel.columns else pd.Series(
        False, index=panel.index
    )
    mapped = panel["promotion_id"].map(starts)
    return anchor.where(~(treated & mapped.notna()), mapped)


def build_covariates(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: PromoUpliftConfig | None = None,
    history: pd.DataFrame | None = None,
) -> CovariateFrame:
    """Build the adjustment set.

    ``history`` is the panel to compute trailing statistics from, which may be
    wider than the analysis panel - a promotion at the very start of the
    requested window still has history before it. When omitted, ``panel`` is used
    and rows without enough history are dropped rather than filled: a covariate
    imputed with a column mean is not "the demand this listing was running at",
    it is the average listing wearing that listing's name.
    """
    settings = config or get_promo_uplift_config()
    source = panel if history is None else history

    trailing = _trailing_features(source, config=settings)
    working = panel.copy()
    working[DATE] = pd.to_datetime(working[DATE])
    working["_anchor_date"] = anchor_dates(working, events)

    # Join trailing statistics at the anchor date, not the row date. The
    # trailing frame is indexed by the date its statistics are valid *as of*, so
    # this is a plain key merge rather than an interval lookup.
    merged = working.merge(
        trailing,
        left_on=["product_id", "store_id", "_anchor_date"],
        right_on=["product_id", "store_id", DATE],
        how="left",
        suffixes=("", "_trailing"),
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_trailing")])

    calendar = _calendar_features(merged)
    # Static attributes are already columns of the panel; only their *names* are
    # collected. Concatenating a copy would duplicate the column, and a
    # duplicated name makes `frame[name]` return a DataFrame rather than a
    # Series - which fails far from here, in the estimator.
    static_names = _static_feature_names(merged)
    merged = pd.concat([merged, calendar], axis=1)

    groups = {
        "demand_history": tuple(trailing.columns.difference([*KEYS, DATE])),
        "calendar_and_season": tuple(calendar.columns),
        "static": static_names,
    }
    feature_names = tuple(
        name for group in groups.values() for name in group
        if name not in POST_TREATMENT_FEATURES and name not in IDENTIFIER_COLUMNS
    )

    complete = _drop_incomplete(merged, feature_names, settings)

    categorical = tuple(
        name for name in feature_names if str(complete[name].dtype) in {"object", "category"}
    )
    for name in categorical:
        complete[name] = complete[name].astype("category")

    _assert_no_post_treatment(feature_names)

    logger.info(
        "promo_uplift.covariates_built",
        rows=len(complete),
        features=len(feature_names),
        categorical=len(categorical),
    )
    return CovariateFrame(
        frame=complete.reset_index(drop=True),
        feature_names=feature_names,
        categorical_names=categorical,
        outcome=settings.target,
        groups=groups,
    )


def _trailing_features(
    source: pd.DataFrame, *, config: PromoUpliftConfig
) -> pd.DataFrame:
    """Demand and price history, valid as of each date.

    Every statistic is computed over rows strictly *before* the date it is
    indexed at, so joining on the anchor date can never reach forward. Built with
    ``shift(1)`` inside each listing rather than by filtering per row: one
    vectorised pass over the panel instead of one window per observation.
    """
    frame = source[[DATE, *KEYS, "units"]].copy()
    frame[DATE] = pd.to_datetime(frame[DATE])
    if "regular_price" in source.columns:
        frame["regular_price"] = source["regular_price"].to_numpy()
    if "promotion_flag" in source.columns:
        frame["_promoted"] = source["promotion_flag"].astype(bool).to_numpy()
    frame = frame.sort_values([*KEYS, DATE])

    grouped = frame.groupby(list(KEYS), observed=True, sort=False)
    # Shift once, then roll. Rolling on the shifted series guarantees the window
    # excludes the current row even when dates are irregular.
    units_prior = grouped["units"].shift(1)
    out = frame[[DATE, *KEYS]].copy()

    for lag in _LAGS:
        out[f"demand_lag_{lag}"] = grouped["units"].shift(lag)

    prior = units_prior.groupby([frame[k] for k in KEYS], observed=True, sort=False)
    for window in _TRAILING_WINDOWS:
        out[f"demand_mean_{window}"] = prior.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(w // 2, 3)).mean()
        )
    for window in (14, 28):
        out[f"demand_std_{window}"] = prior.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(w // 2, 3)).std()
        )

    # Level and shape, separated. The level says how big this listing is; the
    # ratios say whether it is currently running hot or cold, which is what a
    # merchandiser reacts to when deciding to promote.
    out["demand_momentum_7_28"] = out["demand_mean_7"] / out["demand_mean_28"].replace(0, np.nan)
    out["demand_volatility"] = out["demand_std_28"] / out["demand_mean_28"].replace(0, np.nan)
    out["demand_log_level"] = np.log1p(out["demand_mean_28"])

    if "regular_price" in frame.columns:
        price_prior = grouped["regular_price"].shift(1)
        out["regular_price_lag_1"] = price_prior
        price_group = price_prior.groupby([frame[k] for k in KEYS], observed=True, sort=False)
        rolling_price = price_group.transform(
            lambda s: s.rolling(56, min_periods=14).mean()
        )
        out["price_vs_trailing_mean"] = price_prior / rolling_price.replace(0, np.nan)

    if "_promoted" in frame.columns:
        promo_prior = grouped["_promoted"].shift(1).astype(float)
        promo_group = promo_prior.groupby([frame[k] for k in KEYS], observed=True, sort=False)
        # Prior promotion intensity: the strongest single predictor of being
        # promoted again, and a confounder because heavily promoted listings are
        # also different in demand.
        out["promo_share_28"] = promo_group.transform(
            lambda s: s.rolling(28, min_periods=7).mean()
        )
        out["promo_share_90"] = promo_group.transform(
            lambda s: s.rolling(90, min_periods=21).mean()
        )
        out["days_since_promotion"] = _days_since(frame, promo_prior)

    return out.reset_index(drop=True)


def _days_since(frame: pd.DataFrame, promoted_prior: pd.Series) -> pd.Series:
    """Days since the last promoted day, counted from prior rows only.

    A cumulative-max of the row index where the flag was set, forward-filled
    within each listing - one pass, no per-row search.
    """
    position = pd.Series(np.arange(len(frame)), index=frame.index)
    last = position.where(promoted_prior.fillna(0) > 0)
    last = last.groupby([frame[k] for k in KEYS], observed=True, sort=False).ffill()
    return (position - last).astype(float)


def _calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Date-determined covariates, including the seasonal confounder.

    Harmonics rather than an empirical seasonal index estimated from the data.
    An empirical index fitted on control rows would be contaminated by exactly
    the selection it is meant to correct - promotions cluster at seasonal peaks,
    so the control rows under-represent those peaks and the estimated curve is
    flattened there.

    Two harmonics: one annual cycle plus a half-year term, enough for a single
    peak with an asymmetric shape. More would start fitting holiday timing, which
    belongs to the calendar flags rather than to a smooth seasonal term.
    """
    dates = pd.to_datetime(panel[DATE])
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * day_of_year / 365.25

    features = pd.DataFrame(index=panel.index)
    features["day_of_week"] = dates.dt.dayofweek.astype(int)
    features["is_weekend"] = dates.dt.dayofweek.isin((5, 6)).astype(int)
    features["month"] = dates.dt.month.astype(int)
    features["season_sin_1"] = np.sin(angle)
    features["season_cos_1"] = np.cos(angle)
    features["season_sin_2"] = np.sin(2.0 * angle)
    features["season_cos_2"] = np.cos(2.0 * angle)
    # Linear time, so a trend common to treated and control periods is adjusted
    # for rather than absorbed into the effect. Promotions are not uniformly
    # distributed across the window, so an untreated drift would otherwise load
    # onto treatment.
    features["time_index"] = (dates - dates.min()).dt.days.astype(float)

    for flag in ("holiday_flag", "festival_flag"):
        if flag in panel.columns:
            features[flag] = panel[flag].astype(bool).astype(int)
    return features


def _static_feature_names(panel: pd.DataFrame) -> tuple[str, ...]:
    """Listing attributes that do not change over the analysis window.

    Returns names, not a frame: these columns are already present, and the
    caller only needs to know which of them belong in the adjustment set.
    """
    return tuple(
        c
        for c in ("category", "region", "channel", "brand", "store_segment", "store_tier")
        if c in panel.columns
    )


def _drop_incomplete(
    frame: pd.DataFrame, feature_names: tuple[str, ...], config: PromoUpliftConfig
) -> pd.DataFrame:
    """Drop rows whose covariates are not fully determined.

    Dropped, not imputed. A listing three weeks old has no 56-day trailing mean,
    and filling it with the panel average asserts that this listing runs at the
    average rate - a claim nobody made and one the estimator would then treat as
    evidence. The rows that survive are the rows the adjustment set actually
    covers, and the count of what was lost is reported.
    """
    required = [
        name for name in feature_names
        if name.startswith("demand_") or name.startswith("promo_share")
    ]
    if not required:
        return frame

    complete = frame.dropna(subset=required).copy()
    lost = len(frame) - len(complete)
    if lost:
        logger.info(
            "promo_uplift.incomplete_covariates_dropped",
            dropped=lost,
            share=round(lost / max(len(frame), 1), 4),
        )
    if complete.empty:
        raise InsufficientPrePeriodError(
            f"no row has a complete covariate set; the panel needs at least "
            f"{config.controls.pre_period_days} days of history before the first "
            f"promotion for the trailing windows to be defined",
            required_days=config.controls.pre_period_days,
        )
    treated_left = int((complete["role"] == RowRole.TREATED).sum()) if "role" in complete else 0
    if treated_left == 0:
        raise InsufficientPrePeriodError(
            "every treated row was dropped for incomplete pre-treatment history; "
            "the promotions in this window start too close to the beginning of "
            "the available data",
            required_days=config.controls.pre_period_days,
        )
    return complete


def _assert_no_post_treatment(feature_names: tuple[str, ...]) -> None:
    """Fail loudly if a post-treatment column reached the adjustment set.

    A runtime check rather than a code review convention. This is the failure
    that produces a confident, precise, wrong answer - and unlike a crash, it
    leaves no trace in the output.
    """
    leaked = sorted(set(feature_names) & POST_TREATMENT_FEATURES)
    if leaked:
        raise AssertionError(
            f"post-treatment columns reached the adjustment set: {leaked}. "
            f"These are consequences of the promotion; conditioning on them "
            f"blocks the causal path being estimated"
        )


__all__ = [
    "COVARIATE_VERSION",
    "IDENTIFIER_COLUMNS",
    "POST_TREATMENT_FEATURES",
    "CovariateFrame",
    "anchor_dates",
    "build_covariates",
]
