"""Data-quality checks on the forecasting grain (brief section 3).

Distinct from ``data/validation/checks.py``, which validates the *generated
dataset* against its own contract. This module asks a narrower question: **is
this panel fit to forecast from?** Several things that are perfectly acceptable
in a warehouse table are disqualifying for a forecaster.

The clearest example is a missing date. A gap in a series does not violate any
schema and no contract check would flag it, but every lag and rolling feature
silently shifts across it - ``lag_7`` reaches eight days back instead of seven,
and the model learns a relationship that does not exist. That is why each check
below carries a ``why`` string stating the *forecasting* consequence rather than
the generic data-hygiene one.

Every check returns PASS / WARN / FAIL. The distinction matters: WARN means the
panel is usable but the result needs a caveat, FAIL means forecasting from it
would produce numbers nobody should act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger

logger = get_logger(__name__)


class Status(StrEnum):
    PASS = "PASS"  # nosec B105 - a check verdict, not a credential
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    """One check, its verdict, and why a forecaster cares."""

    name: str
    status: Status
    value: str
    why: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL


@dataclass
class QualityReport:
    """The full set of checks for one panel."""

    checks: list[Check] = field(default_factory=list)
    rows: int = 0

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    @property
    def ok(self) -> bool:
        """Whether the panel is fit to forecast from at all."""
        return not self.failed

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"check": c.name, "status": str(c.status), "value": c.value, "why": c.why}
                for c in self.checks
            ]
        )

    def render(self) -> str:
        """Human-readable report."""
        lines = [
            "# Forecasting data-quality report",
            "",
            f"Rows: {self.rows:,}",
            f"Checks: {len(self.checks)}  |  "
            f"failed {len(self.failed)}  |  warned {len(self.warned)}",
            "",
        ]
        width = max((len(c.name) for c in self.checks), default=10)
        for check in self.checks:
            lines.append(f"{check.status!s:<5} {check.name:<{width}}  {check.value}")

        flagged = self.failed + self.warned
        if flagged:
            lines += ["", "## Why these matter", ""]
            for check in flagged:
                lines.append(f"- **{check.name}** ({check.status}): {check.why}")

        if self.ok and not self.warned:
            lines += ["", "All checks passed. The panel is fit to forecast from."]
        return "\n".join(lines)


def _check(
    name: str, ok: bool, value: str, why: str, *, warn_only: bool = False, **detail: Any
) -> Check:
    status = Status.PASS if ok else (Status.WARN if warn_only else Status.FAIL)
    return Check(name=name, status=status, value=value, why=why, detail=detail)


def check_panel(
    panel: pd.DataFrame,
    *,
    date_column: str = "date",
    target_column: str = "units",
    keys: tuple[str, ...] = ("product_id", "store_id"),
    max_zero_share: float = 0.60,
    max_missing_date_share: float = 0.02,
) -> QualityReport:
    """Run every forecasting data-quality check over a panel."""
    report = QualityReport(rows=len(panel))

    if panel.empty:
        report.checks.append(
            _check("rows", False, "0", "An empty panel cannot be forecast from.")
        )
        return report

    working = panel.copy()
    working[date_column] = pd.to_datetime(working[date_column])
    dates = working[date_column]

    # -- shape ---------------------------------------------------------------
    report.checks.append(
        _check(
            "rows", True, f"{len(working):,}",
            "Row count is the denominator for every share below.",
        )
    )
    products = working["product_id"].nunique() if "product_id" in working else 0
    stores = working["store_id"].nunique() if "store_id" in working else 0
    series = len(working.groupby(list(keys), observed=True)) if all(
        k in working for k in keys
    ) else 0
    report.checks.append(
        _check("products", products > 0, f"{products:,}", "No products, nothing to forecast.")
    )
    report.checks.append(
        _check("stores", stores > 0, f"{stores:,}", "No stores, nothing to forecast.")
    )
    report.checks.append(
        _check(
            "series", series > 0, f"{series:,}",
            "A global model pools across series; the count sets how much it can pool.",
        )
    )
    span_days = (dates.max() - dates.min()).days + 1
    report.checks.append(
        _check(
            "date_range", True,
            f"{dates.min().date()} to {dates.max().date()} ({span_days:,} days)",
            "The span bounds every horizon: a 90-day forecast needs 90 days of "
            "known-in-advance data beyond the origin.",
        )
    )

    # -- grain integrity -----------------------------------------------------
    duplicates = int(working.duplicated(subset=[*keys, date_column]).sum())
    report.checks.append(
        _check(
            "duplicate_grain", duplicates == 0, f"{duplicates:,}",
            "A duplicated (product, store, date) row silently doubles that day's "
            "weight in every lag and rolling window, and the self-join then "
            "produces two training rows for one observation.",
            duplicates=duplicates,
        )
    )

    missing_dates, worst_series = _missing_dates(working, date_column=date_column, keys=keys)
    expected = series * ((dates.max() - dates.min()).days + 1)
    missing_share = missing_dates / expected if expected else 0.0
    report.checks.append(
        _check(
            "missing_dates", missing_share <= max_missing_date_share,
            f"{missing_dates:,} ({missing_share:.2%})",
            "A gap in a series shifts every lag across it - lag_7 reaches eight "
            "days back instead of seven - and no schema check would catch it. "
            "This is the check most specific to forecasting.",
            warn_only=missing_share <= max_missing_date_share * 5,
            missing=missing_dates, worst_series=worst_series,
        )
    )

    # -- target --------------------------------------------------------------
    target = pd.to_numeric(working[target_column], errors="coerce")
    negative = int((target < 0).sum())
    report.checks.append(
        _check(
            "negative_units", negative == 0, f"{negative:,}",
            "Negative demand is not a quantity. Under a Poisson objective it is "
            "also undefined, so the fit would fail or silently drop those rows.",
            negative=negative,
        )
    )

    zero_share = float((target == 0).mean())
    report.checks.append(
        _check(
            "zero_sales_share", zero_share <= max_zero_share, f"{zero_share:.1%}",
            "Zero-inflation decides which metrics mean anything. Above this "
            "threshold MAPE is unusable and WMAPE must carry the headline.",
            warn_only=True, zero_share=zero_share,
        )
    )

    missing_target = int(target.isna().sum())
    report.checks.append(
        _check(
            "missing_target", missing_target == 0, f"{missing_target:,}",
            "A row with no target cannot train and cannot score; leaving them in "
            "makes every metric's denominator ambiguous.",
            warn_only=True, missing=missing_target,
        )
    )

    # -- price ---------------------------------------------------------------
    report.checks.extend(_price_checks(working))

    # -- promotion -----------------------------------------------------------
    report.checks.extend(_promotion_checks(working))

    # -- stockouts -----------------------------------------------------------
    if "stockout_flag" in working.columns:
        stockout_share = float(working["stockout_flag"].astype(bool).mean())
        report.checks.append(
            _check(
                "stockout_share", stockout_share < 0.25, f"{stockout_share:.1%}",
                "Stockout rows have a censored target - they record availability, "
                "not demand - and are excluded from training. A high share means "
                "a large slice of history is unusable and the excluded-tail bias "
                "grows.",
                warn_only=stockout_share < 0.40, stockout_share=stockout_share,
            )
        )

    logger.info(
        "forecast.quality_checked",
        rows=report.rows,
        checks=len(report.checks),
        failed=len(report.failed),
        warned=len(report.warned),
    )
    return report


def _missing_dates(
    panel: pd.DataFrame, *, date_column: str, keys: tuple[str, ...]
) -> tuple[int, str | None]:
    """Count date gaps within each series.

    Vectorised over the whole panel rather than looping per series: at 6,000
    series a Python loop is the difference between a second and a minute, and
    this runs on every training call.
    """
    if not all(k in panel.columns for k in keys):
        return 0, None

    counts = panel.groupby(list(keys), observed=True)[date_column].agg(["min", "max", "count"])
    expected = (counts["max"] - counts["min"]).dt.days + 1
    gaps = (expected - counts["count"]).clip(lower=0)

    total = int(gaps.sum())
    worst = None
    if total and not gaps.empty:
        index = gaps.idxmax()
        worst = "|".join(str(part) for part in (index if isinstance(index, tuple) else (index,)))
    return total, worst


def _price_checks(panel: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []

    if "selling_price" in panel.columns:
        price = pd.to_numeric(panel["selling_price"], errors="coerce")
        non_positive = int((price <= 0).sum())
        checks.append(
            _check(
                "price_positive", non_positive == 0, f"{non_positive:,}",
                "A non-positive price breaks the price features and makes revenue "
                "derivation meaningless.",
                non_positive=non_positive,
            )
        )

        if "regular_price" in panel.columns:
            regular = pd.to_numeric(panel["regular_price"], errors="coerce")
            above = int((price > regular * 1.001).sum())
            checks.append(
                _check(
                    "selling_not_above_regular", above == 0, f"{above:,}",
                    "Selling above the regular price inverts the discount features, "
                    "so the model reads a promotion as a price rise.",
                    warn_only=True, above=above,
                )
            )

        # A jump of more than 10x between consecutive days is far outside any
        # real pricing action and points at a units or currency error.
        if "product_id" in panel.columns and "store_id" in panel.columns:
            ordered = panel.sort_values(["product_id", "store_id", "date"])
            previous = ordered.groupby(["product_id", "store_id"], observed=True)[
                "selling_price"
            ].shift(1)
            ratio = pd.to_numeric(ordered["selling_price"], errors="coerce") / previous
            extreme = int(((ratio > 10) | (ratio < 0.1)).sum())
            checks.append(
                _check(
                    "price_jumps", extreme == 0, f"{extreme:,}",
                    "A tenfold overnight price move is not a pricing decision. It "
                    "would dominate the price features and distort elasticity "
                    "downstream.",
                    warn_only=True, extreme=extreme,
                )
            )

    return checks


def _promotion_checks(panel: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []

    if "promotion_flag" not in panel.columns:
        return checks

    flag = panel["promotion_flag"].astype(bool)
    discount_column = next(
        (c for c in ("promotion_discount", "discount_depth", "discount_percentage")
         if c in panel.columns),
        None,
    )

    if discount_column:
        discount = pd.to_numeric(panel[discount_column], errors="coerce").fillna(0.0)
        flagged_no_discount = int((flag & (discount <= 0)).sum())
        checks.append(
            _check(
                "promotion_has_discount", flagged_no_discount == 0, f"{flagged_no_discount:,}",
                "A promotion flag with no discount teaches the model that the flag "
                "alone raises demand, which makes the flag a proxy for whatever "
                "else happened that day.",
                warn_only=True, rows=flagged_no_discount,
            )
        )

        discount_no_flag = int((~flag & (discount > 0.01)).sum())
        checks.append(
            _check(
                "discount_has_flag", discount_no_flag == 0, f"{discount_no_flag:,}",
                "A discount with no flag is an unlabelled promotion. It lands in "
                "the non-promotional baseline and inflates it.",
                warn_only=True, rows=discount_no_flag,
            )
        )

    promo_share = float(flag.mean())
    checks.append(
        _check(
            "promotion_share", 0.0 < promo_share < 0.60, f"{promo_share:.1%}",
            "Too few promotions and the model cannot learn their effect; too many "
            "and the non-promotional baseline has too little support.",
            warn_only=True, promo_share=promo_share,
        )
    )

    return checks


def missing_value_summary(panel: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    """Null rate per column, worst first.

    Reported separately from the checks because a high null rate is rarely
    disqualifying on its own - many features are legitimately undefined early in
    a series - but a column that is *entirely* null means a join failed.
    """
    if panel.empty:
        return pd.DataFrame()

    rates = panel.isna().mean().sort_values(ascending=False)
    frame = rates.head(top_n).reset_index()
    frame.columns = ["column", "null_rate"]
    frame["all_null"] = np.isclose(frame["null_rate"], 1.0)
    return frame
