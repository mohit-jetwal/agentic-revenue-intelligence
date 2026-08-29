"""Treatment definition and the analysis frame (brief sections 4, 6).

This module answers one question and it is the question the whole step turns on:
**which rows were treated, and over what window?**

Three decisions live here.

**Treatment is the whole promotion event, not the promotion flag.** In the
platform generator a promotion moves demand through two channels - the mechanic's
own response curve, and the price cut acting through own-price elasticity. At a
20% depth the price channel is the larger of the two. So the counterfactual is
"no promotion at all, therefore no discount either", and anything that holds the
discount fixed across arms is measuring the smaller half and calling it the
effect.

**There are three windows, not two.** The event window gives *gross* uplift. The
washout window after it carries pull-forward: shoppers loaded their pantry and
buy less for a week or two. Net incrementality is the sum, and it is the only one
of the two that Step 8 can responsibly allocate budget against. Reporting gross
alone is how promotions get renewed that never paid back.

**Rows in the washout are neither treated nor control.** They are depressed *by*
the treatment, so counting them as controls would deflate the comparison baseline
and inflate uplift - the exact error the pull-forward term exists to expose. They
get their own role and are excluded from both arms of the gross estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import TreatmentDefinitionError

logger = get_logger(__name__)

KEYS: tuple[str, ...] = ("product_id", "store_id")
DATE = "date"


class RowRole(StrEnum):
    """What part a row plays in the comparison."""

    #: Inside a qualifying promotion window.
    TREATED = "treated"
    #: Inside the pull-forward window after an event. Neither arm.
    WASHOUT = "washout"
    #: Eligible to serve as a comparison observation.
    CONTROL = "control"
    #: Promoted, but the event failed the treatment definition (too shallow, too
    #: short, wrong mechanic). Excluded from *both* arms - a sub-threshold
    #: promotion is not a clean control either.
    EXCLUDED = "excluded"


@dataclass
class AnalysisFrame:
    """A panel labelled with treatment roles, plus the events behind it."""

    frame: pd.DataFrame
    events: pd.DataFrame
    #: Row counts dropped at each stage, for the report and the quality checks.
    excluded: dict[str, int] = field(default_factory=dict)
    #: Non-fatal findings, carried onto every result produced from this frame.
    warnings: list[str] = field(default_factory=list)

    @property
    def treated(self) -> pd.DataFrame:
        return self.frame[self.frame["role"] == RowRole.TREATED]

    @property
    def control(self) -> pd.DataFrame:
        return self.frame[self.frame["role"] == RowRole.CONTROL]

    @property
    def treated_rows(self) -> int:
        return int((self.frame["role"] == RowRole.TREATED).sum())

    @property
    def control_rows(self) -> int:
        return int((self.frame["role"] == RowRole.CONTROL).sum())

    def summary(self) -> str:
        counts = self.frame["role"].value_counts()
        parts = ", ".join(f"{role} {counts.get(role, 0):,}" for role in RowRole)
        return f"{len(self.frame):,} rows ({parts}) across {len(self.events):,} events"


def extract_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse promoted rows into one row per promotion event.

    Grouping on ``promotion_id`` rather than detecting contiguous blocks. The id
    is the authoritative event boundary in both the gold ``sales`` table and the
    synthetic generator; inferring blocks from adjacency would silently merge two
    back-to-back promotions into one and average away the difference between
    them.
    """
    if "promotion_id" not in panel.columns:
        raise TreatmentDefinitionError(
            "panel has no promotion_id column; treatment cannot be defined"
        )

    promoted = panel[panel["promotion_id"].notna()]
    if promoted.empty:
        return pd.DataFrame(
            columns=[
                "promotion_id", "product_id", "store_id", "start_date", "end_date",
                "duration_days", "discount_depth", "promotion_type", "observed_days",
            ]
        )

    aggregations: dict[str, tuple[str, str]] = {
        "product_id": ("product_id", "first"),
        "store_id": ("store_id", "first"),
        "start_date": (DATE, "min"),
        "end_date": (DATE, "max"),
        "observed_days": (DATE, "count"),
    }
    if "discount_percentage" in promoted.columns:
        aggregations["discount_depth"] = ("discount_percentage", "max")
    if "promotion_type" in promoted.columns:
        aggregations["promotion_type"] = ("promotion_type", "first")
    for column in ("display_flag", "bundle_flag"):
        if column in promoted.columns:
            aggregations[column] = (column, "max")
    if "promotion_spend" in promoted.columns:
        aggregations["promotion_spend"] = ("promotion_spend", "max")

    events = promoted.groupby("promotion_id", sort=True).agg(**aggregations).reset_index()

    if "discount_depth" in events.columns:
        events["discount_depth"] = events["discount_depth"] / 100.0
    else:
        events["discount_depth"] = 0.0
    if "promotion_type" not in events.columns:
        events["promotion_type"] = None

    # Calendar span, not the number of rows observed. A promotion whose middle
    # days are missing from the panel still ran for its full duration, and using
    # the row count would let a data gap shorten an event.
    events["duration_days"] = (
        (pd.to_datetime(events["end_date"]) - pd.to_datetime(events["start_date"])).dt.days + 1
    )
    return events


def qualify_events(
    events: pd.DataFrame, *, config: PromoUpliftConfig | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the configured treatment definition to the event table.

    Returns the events *with* a ``qualifies`` column rather than a filtered
    table. Disqualified events still matter: their rows must be excluded from the
    control arm too, because a 3% discount is not a clean "no promotion"
    observation even though it fails the treatment threshold.
    """
    settings = config or get_promo_uplift_config()
    rule = settings.treatment

    if events.empty:
        return events.assign(qualifies=pd.Series(dtype=bool)), {}

    working = events.copy()
    reasons: dict[str, int] = {}

    deep_enough = working["discount_depth"] >= rule.min_discount_depth
    long_enough = working["duration_days"] >= rule.min_duration_days
    qualifies = deep_enough & long_enough
    reasons["too_shallow"] = int((~deep_enough).sum())
    reasons["too_short"] = int((~long_enough).sum())

    if rule.include_types:
        right_type = working["promotion_type"].isin(list(rule.include_types))
        reasons["wrong_type"] = int((~right_type).sum())
        qualifies &= right_type

    if rule.require_price_reduction:
        has_cut = working["discount_depth"] > 0.0
        reasons["no_price_cut"] = int((~has_cut).sum())
        qualifies &= has_cut

    working["qualifies"] = qualifies

    if not qualifies.any():
        raise TreatmentDefinitionError(
            f"no promotion event satisfies the treatment definition "
            f"({rule.describe()}); every one of {len(working):,} events was "
            f"filtered out, so there is nothing to estimate an effect for",
            affected_rows=len(working),
        )
    return working, reasons


def build_analysis_frame(
    panel: pd.DataFrame,
    *,
    config: PromoUpliftConfig | None = None,
    events: pd.DataFrame | None = None,
) -> AnalysisFrame:
    """Label every row with its treatment role.

    The panel is expected at ``date x product_id x store_id`` grain with at least
    ``units``, ``promotion_id`` and ``stockout_flag``. Both the gold ``sales``
    table and :mod:`ml.promo_uplift.synthetic` satisfy that, which is deliberate:
    the validation data goes through the same code path as production data, so a
    bug here cannot hide in one and not the other.
    """
    settings = config or get_promo_uplift_config()

    working = panel.copy()
    working[DATE] = pd.to_datetime(working[DATE])
    working = working.sort_values([*KEYS, DATE]).reset_index(drop=True)

    _check_grain(working)

    event_table = extract_events(working) if events is None else events.copy()
    event_table, filter_reasons = qualify_events(event_table, config=settings)

    qualifying = set(event_table.loc[event_table["qualifies"], "promotion_id"])
    disqualified = set(event_table.loc[~event_table["qualifies"], "promotion_id"])

    promotion = working["promotion_id"]
    role = pd.Series(RowRole.CONTROL, index=working.index, dtype=object)
    role[promotion.isin(disqualified)] = RowRole.EXCLUDED
    role[promotion.isin(qualifying)] = RowRole.TREATED

    washout = _washout_mask(
        working,
        event_table[event_table["qualifies"]],
        washout_days=settings.treatment.washout_days,
    )
    # Order matters: a washout window that runs into the next promotion is
    # treated, not washout. The next event's own effect dominates, and calling
    # those days "recovery from the previous promotion" would attribute one
    # promotion's lift to another's payback.
    role[washout & (role == RowRole.CONTROL)] = RowRole.WASHOUT

    working["role"] = pd.Categorical(role, categories=list(RowRole))
    working["treatment"] = working["role"] == RowRole.TREATED

    excluded = {
        "total_rows": len(working),
        "washout": int((working["role"] == RowRole.WASHOUT).sum()),
        "sub_threshold_promotion": int((working["role"] == RowRole.EXCLUDED).sum()),
        **{f"events_{reason}": count for reason, count in filter_reasons.items() if count},
    }

    warnings = _treatment_warnings(working, event_table)

    frame = AnalysisFrame(
        frame=working,
        events=event_table[event_table["qualifies"]].reset_index(drop=True),
        excluded=excluded,
        warnings=warnings,
    )
    logger.info(
        "promo_uplift.analysis_frame_built",
        rows=len(working),
        treated=frame.treated_rows,
        control=frame.control_rows,
        events=len(frame.events),
    )
    return frame


def _check_grain(panel: pd.DataFrame) -> None:
    """Refuse a panel with duplicate keys.

    A duplicated product-store-day double-counts one observation, which inflates
    the effective sample size and narrows every confidence interval. Worse, if
    the duplicates disagree on ``promotion_id`` the treatment indicator is
    genuinely ambiguous for that row and the estimand is undefined.
    """
    duplicates = panel.duplicated(subset=[DATE, *KEYS]).sum()
    if duplicates:
        raise TreatmentDefinitionError(
            f"{duplicates:,} duplicate (date, product_id, store_id) rows; the "
            f"treatment indicator is ambiguous where they disagree",
            affected_rows=int(duplicates),
        )


def _washout_mask(
    panel: pd.DataFrame, events: pd.DataFrame, *, washout_days: int
) -> pd.Series:
    """Rows falling within ``washout_days`` after a qualifying event ends.

    Implemented as a backward ``merge_asof`` on the event end date per listing:
    for each row, find the most recent event that ended on or before it, then
    test the gap. One sorted pass instead of an interval join, which pandas does
    not do efficiently and which would dominate runtime at panel scale.
    """
    if washout_days <= 0 or events.empty:
        return pd.Series(False, index=panel.index)

    ends = events[["product_id", "store_id", "end_date"]].copy()
    ends["end_date"] = pd.to_datetime(ends["end_date"])
    ends = ends.sort_values("end_date").rename(columns={"end_date": "_last_event_end"})

    left = panel[[DATE, *KEYS]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values(DATE)

    merged = pd.merge_asof(
        left,
        ends,
        left_on=DATE,
        right_on="_last_event_end",
        by=list(KEYS),
        direction="backward",
        allow_exact_matches=True,
    )
    gap = (merged[DATE] - merged["_last_event_end"]).dt.days
    inside = gap.between(1, washout_days, inclusive="both").fillna(False)

    mask = pd.Series(False, index=panel.index)
    mask.iloc[merged.loc[inside.to_numpy(), "_row"].to_numpy()] = True
    return mask


def _treatment_warnings(panel: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    """Findings that do not block an estimate but must travel with it."""
    warnings: list[str] = []

    if "promotion_flag" in panel.columns:
        flagged = panel["promotion_flag"].astype(bool)
        identified = panel["promotion_id"].notna()
        orphan_flags = int((flagged & ~identified).sum())
        if orphan_flags:
            warnings.append(
                f"{orphan_flags:,} rows are flagged as promoted but carry no "
                f"promotion_id; they cannot be attributed to an event and are "
                f"treated as control, which biases uplift downward"
            )

    if "discount_percentage" in panel.columns:
        treated = panel["role"] == RowRole.TREATED
        zero_depth = int((treated & (panel["discount_percentage"] <= 0)).sum())
        if zero_depth:
            warnings.append(
                f"{zero_depth:,} treated rows record no discount; if these are "
                f"display-only mechanics that is correct, otherwise the "
                f"promotion depth feed has gaps"
            )

    if not events.empty:
        qualifying = events[events["qualifies"]] if "qualifies" in events else events
        if len(qualifying) < 30:
            warnings.append(
                f"only {len(qualifying)} qualifying events; segment-level "
                f"estimates will be noisy and some may not be estimable at all"
            )

    return warnings


def event_window_rows(
    frame: AnalysisFrame,
    promotion_id: str,
    *,
    include_washout: bool = False,
) -> pd.DataFrame:
    """Rows belonging to one promotion, optionally including its washout.

    ``include_washout`` is the switch between the two estimands: gross uplift
    over the event window, and net incrementality once pull-forward has been
    paid back.
    """
    event = frame.events[frame.events["promotion_id"] == promotion_id]
    if event.empty:
        return frame.frame.iloc[:0]

    row = event.iloc[0]
    end = pd.to_datetime(row["end_date"])
    if include_washout:
        end = end + pd.Timedelta(days=int(frame.frame.attrs.get("washout_days", 0)))

    panel = frame.frame
    return panel[
        (panel["product_id"] == row["product_id"])
        & (panel["store_id"] == row["store_id"])
        & (panel[DATE] >= pd.to_datetime(row["start_date"]))
        & (panel[DATE] <= end)
    ]


def treated_period(events: pd.DataFrame) -> tuple[date, date]:
    """First and last day covered by any qualifying event."""
    if events.empty:
        raise TreatmentDefinitionError("no qualifying events, so there is no treated period")
    start = pd.to_datetime(events["start_date"]).min().date()
    end = pd.to_datetime(events["end_date"]).max().date()
    return start, end


__all__ = [
    "AnalysisFrame",
    "RowRole",
    "build_analysis_frame",
    "event_window_rows",
    "extract_events",
    "qualify_events",
    "treated_period",
]
