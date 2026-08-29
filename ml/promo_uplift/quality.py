"""Data-quality checks on the causal grain (brief section 7).

Same shape as :mod:`ml.forecasting.quality` - PASS / WARN / FAIL, each check
carrying why it matters - but the checks are different because the failure modes
are different. A forecasting model degrades gracefully when its inputs are
imperfect: a few missing days cost accuracy and the error is visible in the
backtest. A causal estimate does not degrade, it *misleads*, and nothing in the
output says so.

The instruction that shapes this module is section 7's: **do not silently fix
causal data problems.** A missing promotion record is not a null to impute. It
means a treated day is sitting in the control group, deflating the comparison
baseline and inflating uplift - and imputing it would convert a visible data gap
into an invisible bias. So these checks report, and the pipeline attaches what
they found to every estimate.

Each check states the *direction* of the bias where it is knowable. "Uplift is
overstated" is actionable; "data quality issue" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config

logger = get_logger(__name__)


class Status(StrEnum):
    PASS = "PASS"  # nosec B105 - a check verdict, not a credential
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    """One check, its verdict, and why a causal analyst cares."""

    name: str
    status: Status
    detail: str
    why: str
    value: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL


@dataclass
class QualityReport:
    """Every check, with the verdicts that block an estimate."""

    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    @property
    def passed(self) -> bool:
        return not self.failures

    def messages(self) -> list[str]:
        """Warning strings to attach to a result."""
        return [f"{c.name}: {c.detail}" for c in self.warnings + self.failures]

    def render(self) -> str:
        lines = [
            "| Check | Status | Detail | Why it matters |",
            "|---|---|---|---|",
        ]
        lines.extend(
            f"| {c.name} | {c.status} | {c.detail} | {c.why} |" for c in self.checks
        )
        counts = pd.Series([c.status for c in self.checks]).value_counts()
        lines.append("")
        lines.append(
            f"**{counts.get(Status.PASS, 0)} passed, "
            f"{counts.get(Status.WARN, 0)} warnings, "
            f"{counts.get(Status.FAIL, 0)} failures**"
        )
        return "\n".join(lines)


def check_panel(
    panel: pd.DataFrame, *, config: PromoUpliftConfig | None = None
) -> QualityReport:
    """Run every check against a promotion analysis panel."""
    settings = config or get_promo_uplift_config()
    checks: list[Check] = [
        _rows(panel),
        _duplicate_grain(panel),
        _missing_dates(panel),
        _negative_units(panel),
        _missing_treatment_label(panel),
        _promotion_without_discount(panel),
        _discount_without_promotion(panel),
        _overlapping_promotions(panel),
        _invalid_prices(panel),
        _stockout_share(panel, settings),
        _differential_censoring(panel, settings),
        _pre_period_history(panel, settings),
        _control_availability(panel, settings),
        _always_treated_series(panel),
    ]
    report = QualityReport(checks=checks)
    logger.info(
        "promo_uplift.quality_checked",
        checks=len(checks),
        failures=len(report.failures),
        warnings=len(report.warnings),
    )
    return report


def _rows(panel: pd.DataFrame) -> Check:
    n = len(panel)
    status = Status.PASS if n >= 1000 else Status.WARN if n >= 100 else Status.FAIL
    return Check(
        "rows",
        status,
        f"{n:,} rows",
        "Causal estimates on a few hundred rows have intervals wide enough to "
        "contain any business conclusion.",
        float(n),
    )


def _duplicate_grain(panel: pd.DataFrame) -> Check:
    keys = ["date", "product_id", "store_id"]
    if not all(k in panel.columns for k in keys):
        return Check("duplicate_grain", Status.FAIL, "grain columns missing", "")
    duplicates = int(panel.duplicated(subset=keys).sum())
    return Check(
        "duplicate_grain",
        Status.PASS if duplicates == 0 else Status.FAIL,
        f"{duplicates:,} duplicate product-store-days",
        "A duplicated row double-counts one observation, narrowing every "
        "interval. Where duplicates disagree on promotion status the treatment "
        "indicator is ambiguous and the estimand is undefined.",
        float(duplicates),
    )


def _missing_dates(panel: pd.DataFrame) -> Check:
    if "date" not in panel.columns:
        return Check("missing_dates", Status.FAIL, "no date column", "")
    grouped = panel.groupby(["product_id", "store_id"], observed=True)["date"]
    span = (grouped.max() - grouped.min()).dt.days + 1
    observed = grouped.count()
    missing = int((span - observed).clip(lower=0).sum())
    share = missing / max(float(span.sum()), 1.0)
    status = Status.PASS if share < 0.01 else Status.WARN if share < 0.05 else Status.FAIL
    return Check(
        "missing_dates",
        status,
        f"{missing:,} missing days ({share:.2%} of spans)",
        "A gap inside a promotion shortens its measured window; a gap in the "
        "pre-period corrupts the trailing covariates that the whole adjustment "
        "rests on.",
        share,
    )


def _negative_units(panel: pd.DataFrame) -> Check:
    if "units" not in panel.columns:
        return Check("negative_units", Status.FAIL, "no units column", "")
    negative = int((panel["units"] < 0).sum())
    return Check(
        "negative_units",
        Status.PASS if negative == 0 else Status.FAIL,
        f"{negative:,} rows with negative units",
        "Returns booked as negative sales land in whichever arm the date falls "
        "in, and a return usually follows a promotion - so they subtract from "
        "the treated arm and understate uplift.",
        float(negative),
    )


def _missing_treatment_label(panel: pd.DataFrame) -> Check:
    if "promotion_flag" not in panel.columns or "promotion_id" not in panel.columns:
        return Check(
            "missing_treatment_label",
            Status.WARN,
            "promotion_flag or promotion_id absent",
            "Without both, promoted rows cannot be attributed to an event.",
        )
    orphans = int((panel["promotion_flag"].astype(bool) & panel["promotion_id"].isna()).sum())
    share = orphans / max(len(panel), 1)
    status = Status.PASS if orphans == 0 else Status.WARN if share < 0.01 else Status.FAIL
    return Check(
        "missing_treatment_label",
        status,
        f"{orphans:,} flagged rows with no promotion_id",
        "These are promoted days that land in the CONTROL group, carrying their "
        "promotional lift with them. That raises the comparison baseline, so "
        "uplift is UNDERSTATED. Not imputable - a guessed event id would "
        "attribute the days to the wrong promotion.",
        share,
    )


def _promotion_without_discount(panel: pd.DataFrame) -> Check:
    if "promotion_flag" not in panel.columns or "discount_percentage" not in panel.columns:
        return Check("promotion_without_discount", Status.WARN, "columns absent", "")
    promoted = panel["promotion_flag"].astype(bool)
    zero = int((promoted & (panel["discount_percentage"] <= 0)).sum())
    share = zero / max(int(promoted.sum()), 1)
    status = Status.PASS if share < 0.05 else Status.WARN
    return Check(
        "promotion_without_discount",
        status,
        f"{zero:,} promoted rows with no discount ({share:.1%})",
        "Legitimate for display or bundle mechanics. If these are price "
        "promotions with a missing depth feed, they fall below the treatment "
        "threshold and are dropped from both arms.",
        share,
    )


def _discount_without_promotion(panel: pd.DataFrame) -> Check:
    if "promotion_flag" not in panel.columns or "discount_percentage" not in panel.columns:
        return Check("discount_without_promotion", Status.WARN, "columns absent", "")
    unpromoted = ~panel["promotion_flag"].astype(bool)
    discounted = int((unpromoted & (panel["discount_percentage"] > 0)).sum())
    share = discounted / max(int(unpromoted.sum()), 1)
    status = Status.PASS if share < 0.01 else Status.WARN if share < 0.05 else Status.FAIL
    return Check(
        "discount_without_promotion",
        status,
        f"{discounted:,} unpromoted rows carry a discount ({share:.1%})",
        "Control rows sold below list. They are cheaper than a true no-promotion "
        "day, so they sell more, which raises the comparison baseline and "
        "UNDERSTATES uplift.",
        share,
    )


def _overlapping_promotions(panel: pd.DataFrame) -> Check:
    keys = ["date", "product_id", "store_id"]
    if "promotion_id" not in panel.columns or not all(k in panel.columns for k in keys):
        return Check("overlapping_promotions", Status.WARN, "columns absent", "")
    promoted = panel[panel["promotion_id"].notna()]
    overlaps = int(promoted.groupby(keys, observed=True)["promotion_id"].nunique().gt(1).sum())
    return Check(
        "overlapping_promotions",
        Status.PASS if overlaps == 0 else Status.FAIL,
        f"{overlaps:,} product-store-days under two promotions",
        "Uplift cannot be attributed between simultaneous promotions. The "
        "treatment is not a single well-defined intervention, so the effect it "
        "identifies is undefined rather than merely imprecise.",
        float(overlaps),
    )


def _invalid_prices(panel: pd.DataFrame) -> Check:
    columns = [c for c in ("regular_price", "selling_price") if c in panel.columns]
    if not columns:
        return Check("invalid_prices", Status.WARN, "no price columns", "")
    invalid = int(sum((panel[c] <= 0).sum() for c in columns))
    above = 0
    if len(columns) == 2:
        above = int((panel["selling_price"] > panel["regular_price"] * 1.001).sum())
    total = invalid + above
    return Check(
        "invalid_prices",
        Status.PASS if total == 0 else Status.WARN,
        f"{invalid:,} non-positive, {above:,} selling above regular",
        "Price is the largest channel through which a promotion moves demand. A "
        "corrupted price makes the discount depth wrong, which puts events on "
        "the wrong side of the treatment threshold.",
        float(total),
    )


def _stockout_share(panel: pd.DataFrame, config: PromoUpliftConfig) -> Check:
    if "stockout_flag" not in panel.columns:
        return Check("stockout_share", Status.WARN, "no stockout_flag", "")
    share = float(panel["stockout_flag"].astype(bool).mean())
    status = Status.PASS if share < 0.05 else Status.WARN if share < 0.20 else Status.FAIL
    _ = config
    return Check(
        "stockout_share",
        status,
        f"{share:.1%} of rows censored",
        "Censored rows record what was available, not what was wanted. At a high "
        "share the estimand narrows to a small in-stock subset that may not "
        "resemble the promotions being asked about.",
        share,
    )


def _differential_censoring(panel: pd.DataFrame, config: PromoUpliftConfig) -> Check:
    if "stockout_flag" not in panel.columns or "promotion_flag" not in panel.columns:
        return Check("differential_censoring", Status.WARN, "columns absent", "")
    promoted = panel["promotion_flag"].astype(bool)
    if not promoted.any() or promoted.all():
        return Check("differential_censoring", Status.WARN, "one arm is empty", "")

    stockout = panel["stockout_flag"].astype(bool)
    treated_rate = float(stockout[promoted].mean())
    control_rate = float(stockout[~promoted].mean())
    gap = treated_rate - control_rate
    threshold = config.stockouts.differential_censoring_warn_pp
    status = Status.PASS if abs(gap) <= threshold else Status.WARN
    return Check(
        "differential_censoring",
        status,
        f"treated {treated_rate:.1%} vs control {control_rate:.1%} (gap {gap:+.1%})",
        "THE critical stockout check. Promotions raise demand, demand outruns "
        "replenishment, so treated rows censor more. Excluding them then drops "
        "the highest-demand promotion days and UNDERSTATES uplift. A positive "
        "gap means the exclusion is selective, not incidental.",
        gap,
    )


def _pre_period_history(panel: pd.DataFrame, config: PromoUpliftConfig) -> Check:
    if "date" not in panel.columns:
        return Check("pre_period_history", Status.FAIL, "no date column", "")
    grouped = panel.groupby(["product_id", "store_id"], observed=True)["date"]
    span = (grouped.max() - grouped.min()).dt.days + 1
    required = config.controls.pre_period_days
    short = int((span < required).sum())
    share = short / max(len(span), 1)
    status = Status.PASS if share < 0.05 else Status.WARN if share < 0.25 else Status.FAIL
    return Check(
        "pre_period_history",
        status,
        f"{short:,} of {len(span):,} listings have under {required} days",
        "Every covariate is measured before treatment. A listing without enough "
        "history has no trailing demand, no prior promotion frequency and no "
        "pre-trend - so it is dropped, and the estimate applies to the "
        "established listings that remain.",
        share,
    )


def _control_availability(panel: pd.DataFrame, config: PromoUpliftConfig) -> Check:
    if "promotion_flag" not in panel.columns:
        return Check("control_availability", Status.WARN, "no promotion_flag", "")
    promoted = panel["promotion_flag"].astype(bool)
    control_rows = int((~promoted).sum())
    minimum = config.controls.min_control_rows
    status = (
        Status.PASS
        if control_rows >= minimum * 10
        else Status.WARN
        if control_rows >= minimum
        else Status.FAIL
    )
    return Check(
        "control_availability",
        status,
        f"{control_rows:,} unpromoted rows against a floor of {minimum}",
        "Without controls there is nothing to compare against, and any 'uplift' "
        "is the treated period compared to itself.",
        float(control_rows),
    )


def _always_treated_series(panel: pd.DataFrame) -> Check:
    if "promotion_flag" not in panel.columns:
        return Check("always_treated_series", Status.WARN, "no promotion_flag", "")
    rates = panel.groupby(["product_id", "store_id"], observed=True)["promotion_flag"].mean()
    always = int((rates > 0.95).sum())
    share = always / max(len(rates), 1)
    status = Status.PASS if share < 0.02 else Status.WARN if share < 0.10 else Status.FAIL
    return Check(
        "always_treated_series",
        status,
        f"{always:,} listings promoted on over 95% of days ({share:.1%})",
        "A perpetually promoted listing has no within-series control and its "
        "propensity approaches 1, so it contributes an unbounded weight. These "
        "are the rows overlap trimming removes, changing the estimand to "
        "'listings that were sometimes not promoted'.",
        share,
    )


__all__ = ["Check", "QualityReport", "Status", "check_panel"]
