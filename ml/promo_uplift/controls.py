"""Control construction (brief sections 4, 6, 16).

A control observation has one job: to stand in for what the treated observation
*would* have been. Every way of choosing controls is a claim about comparability,
and the claim is usually the weakest link in a causal estimate - weaker than the
estimator, weaker than the covariates, weaker than the sample size.

Two pools, because they fail differently.

**Within-series controls** - unpromoted days from the same product in the same
store. Product identity, store identity, shelf position, local competition and
customer mix are all held fixed exactly rather than adjusted for, which removes
whole families of confounders without modelling any of them. The failure mode is
temporal: the control days are at a different point in the season, and if that
difference is what drove the promotion decision it is also what drives the gap.
Hence the ``same_series_window_days`` fence - close in time, so the seasonal
position is similar.

**Cross-sectional controls** - never-promoted listings in the same category and
region. Contemporaneous, so seasonality and any market-wide shock are shared. The
failure mode is compositional: a listing that never gets promoted is usually
different in kind - slower, more niche, worse distribution - and those
differences are exactly what the propensity model then has to carry.

Neither is sufficient alone, which is why both are built and why the balance
diagnostics in :mod:`ml.promo_uplift.diagnostics` are not optional decoration.

**What is deliberately excluded from both pools**: washout rows. They are
depressed *by* the treatment, so using them as controls deflates the baseline
and inflates uplift. That is the single most common way a pull-forward effect
gets converted into apparent incrementality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import NoControlGroupError
from ml.promo_uplift.treatment import DATE, KEYS, AnalysisFrame, RowRole

logger = get_logger(__name__)


@dataclass
class ControlPool:
    """Treated rows and the controls judged comparable to them."""

    frame: pd.DataFrame
    treated_rows: int
    control_rows: int
    #: Control rows by origin, so the report can say where comparability came
    #: from rather than presenting one undifferentiated pool.
    within_series_rows: int
    cross_sectional_rows: int
    dropped: dict[str, int]
    warnings: list[str]

    @property
    def treatment(self) -> np.ndarray:
        return self.frame["treatment"].to_numpy(dtype=bool)

    def summary(self) -> str:
        return (
            f"{self.treated_rows:,} treated vs {self.control_rows:,} control "
            f"({self.within_series_rows:,} within-series, "
            f"{self.cross_sectional_rows:,} cross-sectional)"
        )


def days_to_nearest_event(
    panel: pd.DataFrame, events: pd.DataFrame
) -> pd.Series:
    """Days from each row to the nearest qualifying event on the same listing.

    Zero inside an event, positive outside. ``NaN`` where the listing has no
    events at all, which is what distinguishes a never-treated series from one
    that is merely far from its own promotions.

    Two ``merge_asof`` passes rather than an interval join: one backward to the
    most recent event end, one forward to the next event start. Both are single
    sorted passes, where a cross join of rows against events would be
    quadratic in the worst case.
    """
    if events.empty:
        return pd.Series(np.nan, index=panel.index)

    left = panel[[DATE, *KEYS]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values(DATE)

    ends = events[[*KEYS, "end_date"]].copy()
    ends["end_date"] = pd.to_datetime(ends["end_date"])
    ends = ends.sort_values("end_date")

    starts = events[[*KEYS, "start_date"]].copy()
    starts["start_date"] = pd.to_datetime(starts["start_date"])
    starts = starts.sort_values("start_date")

    backward = pd.merge_asof(
        left, ends, left_on=DATE, right_on="end_date",
        by=list(KEYS), direction="backward", allow_exact_matches=True,
    )
    forward = pd.merge_asof(
        left, starts, left_on=DATE, right_on="start_date",
        by=list(KEYS), direction="forward", allow_exact_matches=True,
    )

    after = (backward[DATE] - backward["end_date"]).dt.days
    before = (forward["start_date"] - forward[DATE]).dt.days
    nearest = pd.concat([after.clip(lower=0), before.clip(lower=0)], axis=1).min(
        axis=1, skipna=True
    )

    # A row in the *middle* of an event has no earlier end and no later start,
    # so both merges return NaN - which reads as "this listing was never
    # promoted" and would put it in the cross-sectional control pool. A third
    # backward merge, on the event START, catches those rows: an event that
    # began on or before this date and has not ended by it contains it.
    inside = pd.merge_asof(
        left,
        starts.rename(columns={"start_date": "_open_start"}),
        left_on=DATE,
        right_on="_open_start",
        by=list(KEYS),
        direction="backward",
        allow_exact_matches=True,
    )
    open_event = inside["_open_start"].notna() & (
        backward["end_date"].isna() | (backward["end_date"] < inside["_open_start"])
    )
    nearest = nearest.where(~open_event.to_numpy(), 0.0)

    values = np.full(len(panel), np.nan, dtype=float)
    values[backward["_row"].to_numpy()] = nearest.to_numpy(dtype=float)
    return pd.Series(values, index=panel.index, dtype=float)


def build_control_pool(
    analysis: AnalysisFrame, *, config: PromoUpliftConfig | None = None
) -> ControlPool:
    """Assemble the comparison set for the pooled estimators.

    Returns treated rows plus eligible controls in one frame with a boolean
    ``treatment`` column - the shape every estimator in this package consumes.
    """
    settings = config or get_promo_uplift_config()
    rule = settings.controls

    panel = analysis.frame
    events = analysis.events
    dropped: dict[str, int] = {}
    warnings: list[str] = list(analysis.warnings)

    treated = panel[panel["role"] == RowRole.TREATED].copy()
    candidates = panel[panel["role"] == RowRole.CONTROL].copy()
    dropped["not_control_role"] = len(panel) - len(treated) - len(candidates)

    if treated.empty:
        raise NoControlGroupError(
            "no treated rows survive the treatment definition, so there is "
            "nothing to compare against",
            treated_rows=0,
            control_rows=len(candidates),
        )

    gap = days_to_nearest_event(candidates, events)
    candidates["_days_to_event"] = gap

    # Within-series: close enough in time that the seasonal position is similar.
    within = candidates["_days_to_event"].notna() & (
        candidates["_days_to_event"] <= rule.same_series_window_days
    )
    # Cross-sectional: listings with no qualifying events of their own, matched
    # on category and region to the treated set and restricted to the treated
    # period so the comparison is contemporaneous.
    cross = pd.Series(False, index=candidates.index)
    if rule.use_cross_sectional_controls:
        cross = _cross_sectional_mask(candidates, treated)

    eligible = within | cross
    dropped["control_too_far_in_time"] = int((~eligible).sum())

    selected = candidates[eligible].copy()
    selected["control_origin"] = np.where(
        within[eligible].to_numpy(), "within_series", "cross_sectional"
    )
    treated["control_origin"] = "treated"

    pool = pd.concat([treated, selected], ignore_index=True)
    pool = pool.sort_values([*KEYS, DATE]).reset_index(drop=True)
    pool["treatment"] = pool["role"] == RowRole.TREATED

    within_rows = int((selected["control_origin"] == "within_series").sum())
    cross_rows = int((selected["control_origin"] == "cross_sectional").sum())

    _check_sufficiency(
        treated_rows=len(treated),
        control_rows=len(selected),
        rule=rule,
    )

    if cross_rows == 0 and rule.use_cross_sectional_controls:
        warnings.append(
            "no never-treated listings were available in the same category and "
            "region, so every control is a different day of the same listing; "
            "any effect shared by all listings at that time is not separable "
            "from the promotion"
        )
    if within_rows == 0:
        warnings.append(
            "no within-series controls; the comparison rests entirely on other "
            "listings, so product-level differences must be carried by the "
            "covariates rather than held fixed by design"
        )

    logger.info(
        "promo_uplift.control_pool_built",
        treated=len(treated),
        control=len(selected),
        within_series=within_rows,
        cross_sectional=cross_rows,
    )
    return ControlPool(
        frame=pool,
        treated_rows=len(treated),
        control_rows=len(selected),
        within_series_rows=within_rows,
        cross_sectional_rows=cross_rows,
        dropped=dropped,
        warnings=warnings,
    )


def _cross_sectional_mask(candidates: pd.DataFrame, treated: pd.DataFrame) -> pd.Series:
    """Control rows from never-treated listings in a treated category and region.

    ``_days_to_event`` is NaN exactly when a listing has no qualifying events,
    which is the definition of never-treated. Reusing it avoids a second pass
    over the event table.
    """
    never_treated = candidates["_days_to_event"].isna()

    strata = [c for c in ("category", "region") if c in candidates.columns]
    if not strata:
        # Without category or region there is no defensible stratum, so the
        # cross-sectional pool would be "every other listing" - which is not a
        # comparison group, it is a population average.
        return pd.Series(False, index=candidates.index)

    treated_strata = set(map(tuple, treated[strata].drop_duplicates().to_numpy()))
    in_stratum = pd.Series(
        [tuple(row) in treated_strata for row in candidates[strata].to_numpy()],
        index=candidates.index,
    )

    start = pd.to_datetime(treated[DATE]).min()
    end = pd.to_datetime(treated[DATE]).max()
    contemporaneous = candidates[DATE].between(start, end)

    return never_treated & in_stratum & contemporaneous


def _check_sufficiency(*, treated_rows: int, control_rows: int, rule: object) -> None:
    """Refuse rather than return an estimate no data supports.

    The thresholds are not statistical power calculations - they are floors
    below which the estimate is arithmetic rather than inference. Five treated
    rows and twenty controls will happily produce a point estimate and a
    confidence interval, and both will be meaningless.
    """
    min_control = getattr(rule, "min_control_rows", 30)
    min_treated = getattr(rule, "min_treated_rows", 5)

    if control_rows < min_control:
        raise NoControlGroupError(
            f"only {control_rows} eligible control rows, below the minimum of "
            f"{min_control}. Widen controls.same_series_window_days, enable "
            f"cross-sectional controls, or request a longer date range",
            treated_rows=treated_rows,
            control_rows=control_rows,
            required_control_rows=min_control,
        )
    if treated_rows < min_treated:
        raise NoControlGroupError(
            f"only {treated_rows} treated rows, below the minimum of "
            f"{min_treated}; an effect estimated from this many observations "
            f"would be dominated by the noise on any one of them",
            treated_rows=treated_rows,
            control_rows=control_rows,
        )


def event_controls(
    analysis: AnalysisFrame,
    promotion_id: str,
    *,
    config: PromoUpliftConfig | None = None,
) -> pd.DataFrame:
    """Control rows for one specific event, for per-event and DiD estimates.

    Narrower than the pooled selection: only the same listing, only within the
    configured window either side of *this* event. A per-event estimate that
    borrowed controls from across the whole panel would not be a statement about
    this promotion.
    """
    settings = config or get_promo_uplift_config()
    window = settings.controls.same_series_window_days

    event = analysis.events[analysis.events["promotion_id"] == promotion_id]
    if event.empty:
        return analysis.frame.iloc[:0]

    row = event.iloc[0]
    start = pd.to_datetime(row["start_date"])
    end = pd.to_datetime(row["end_date"])

    panel = analysis.frame
    same_listing = (
        (panel["product_id"] == row["product_id"])
        & (panel["store_id"] == row["store_id"])
        & (panel["role"] == RowRole.CONTROL)
    )
    in_window = (panel[DATE] >= start - pd.Timedelta(days=window)) & (
        panel[DATE] <= end + pd.Timedelta(days=window)
    )
    return panel[same_listing & in_window]


__all__ = [
    "ControlPool",
    "build_control_pool",
    "days_to_nearest_event",
    "event_controls",
]
