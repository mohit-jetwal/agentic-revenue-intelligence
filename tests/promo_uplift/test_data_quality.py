"""Causal data-quality checks.

Every check is tested twice: a clean panel must pass it, and a deliberately
corrupted one must fire it. A check that has never failed is indistinguishable
from one that does nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.promo_uplift.quality import Status, check_panel

pytestmark = [pytest.mark.data, pytest.mark.models]


def clean_panel(days: int = 200, listings: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for i in range(listings):
        promoted_window = range(100, 115) if i % 3 == 0 else range(0)
        for j, day in enumerate(dates):
            promoted = j in promoted_window
            rows.append(
                {
                    "date": day,
                    "product_id": f"P{i:02d}",
                    "store_id": f"S{i % 4}",
                    "units": 20,
                    "regular_price": 10.0,
                    "selling_price": 8.0 if promoted else 10.0,
                    "discount_percentage": 20.0 if promoted else 0.0,
                    "promotion_id": f"PR{i:02d}" if promoted else None,
                    "promotion_flag": promoted,
                    "stockout_flag": False,
                }
            )
    return pd.DataFrame(rows)


def status_of(panel: pd.DataFrame, name: str) -> Status:
    report = check_panel(panel)
    return next(c.status for c in report.checks if c.name == name)


class TestCleanPanel:
    def test_a_clean_panel_passes_everything(self) -> None:
        report = check_panel(clean_panel())
        assert report.passed
        assert not report.failures

    def test_every_check_states_why_it_matters(self) -> None:
        """A verdict without a reason is not actionable."""
        for check in check_panel(clean_panel()).checks:
            assert check.why, f"{check.name} has no rationale"

    def test_the_report_renders(self) -> None:
        rendered = check_panel(clean_panel()).render()
        assert "| Check | Status |" in rendered
        assert "passed" in rendered


class TestChecksFire:
    def test_duplicate_grain(self) -> None:
        panel = clean_panel()
        corrupted = pd.concat([panel, panel.iloc[[5]]], ignore_index=True)
        assert status_of(corrupted, "duplicate_grain") is Status.FAIL

    def test_missing_dates(self) -> None:
        """A gap in the middle of a listing, not a shortened range.

        The check compares the calendar span against the row count, so removing
        a listing's tail shortens the span and is invisible - which is correct:
        a listing with less history is a different problem, caught by
        `pre_period_history`.
        """
        panel = clean_panel()
        first_listing = panel["product_id"] == "P00"
        gap = panel[first_listing].index[50:150]
        corrupted = panel.drop(gap).reset_index(drop=True)
        assert status_of(corrupted, "missing_dates") is not Status.PASS

    def test_negative_units(self) -> None:
        panel = clean_panel()
        panel.loc[5, "units"] = -3
        assert status_of(panel, "negative_units") is Status.FAIL

    def test_missing_treatment_label(self) -> None:
        """A promoted day with no id lands in the control arm, carrying its
        lift with it - which understates uplift."""
        panel = clean_panel()
        panel.loc[panel.index[:500], "promotion_flag"] = True
        assert status_of(panel, "missing_treatment_label") is not Status.PASS

    def test_bias_direction_is_stated_for_missing_labels(self) -> None:
        panel = clean_panel()
        panel.loc[panel.index[:500], "promotion_flag"] = True
        check = next(
            c for c in check_panel(panel).checks if c.name == "missing_treatment_label"
        )
        assert "UNDERSTATED" in check.why

    def test_promotion_without_discount(self) -> None:
        panel = clean_panel()
        promoted = panel["promotion_flag"]
        panel.loc[promoted, "discount_percentage"] = 0.0
        assert status_of(panel, "promotion_without_discount") is not Status.PASS

    def test_discount_without_promotion(self) -> None:
        panel = clean_panel()
        unpromoted = ~panel["promotion_flag"].astype(bool)
        panel.loc[panel[unpromoted].index[:2000], "discount_percentage"] = 15.0
        assert status_of(panel, "discount_without_promotion") is not Status.PASS

    def test_overlapping_promotions(self) -> None:
        """Two promotions on one day makes the treatment indicator ambiguous,
        so the estimand is undefined rather than merely imprecise."""
        panel = clean_panel()
        extra = panel[panel["promotion_flag"]].head(5).copy()
        extra["promotion_id"] = "PR_OTHER"
        corrupted = pd.concat([panel, extra], ignore_index=True)
        assert status_of(corrupted, "overlapping_promotions") is Status.FAIL

    def test_invalid_prices(self) -> None:
        panel = clean_panel()
        panel.loc[3, "selling_price"] = -1.0
        assert status_of(panel, "invalid_prices") is not Status.PASS

    def test_selling_above_regular(self) -> None:
        panel = clean_panel()
        panel.loc[3, "selling_price"] = 99.0
        assert status_of(panel, "invalid_prices") is not Status.PASS

    def test_stockout_share(self) -> None:
        panel = clean_panel()
        panel.loc[panel.index[:1000], "stockout_flag"] = True
        assert status_of(panel, "stockout_share") is not Status.PASS

    def test_differential_censoring(self) -> None:
        """The critical stockout check: promotions cause stockouts, so treated
        rows censor more and excluding them understates uplift."""
        panel = clean_panel()
        promoted = panel["promotion_flag"].astype(bool)
        panel.loc[promoted, "stockout_flag"] = True
        assert status_of(panel, "differential_censoring") is Status.WARN

    def test_differential_censoring_passes_when_balanced(self) -> None:
        panel = clean_panel()
        assert status_of(panel, "differential_censoring") is Status.PASS

    def test_pre_period_history(self) -> None:
        panel = clean_panel(days=20)
        assert status_of(panel, "pre_period_history") is not Status.PASS

    def test_control_availability(self) -> None:
        panel = clean_panel(days=200, listings=1)
        panel["promotion_flag"] = True
        panel["promotion_id"] = "PR00"
        assert status_of(panel, "control_availability") is Status.FAIL

    def test_always_treated_series(self) -> None:
        """A perpetually promoted listing has no within-series control and its
        propensity approaches 1, contributing an unbounded weight."""
        panel = clean_panel()
        always = panel["product_id"] == "P00"
        panel.loc[always, "promotion_flag"] = True
        panel.loc[always, "promotion_id"] = "PR00"
        assert status_of(panel, "always_treated_series") is not Status.PASS

    def test_rows(self) -> None:
        assert status_of(clean_panel(days=5, listings=1), "rows") is Status.FAIL


class TestReportBehaviour:
    def test_failures_block_and_warnings_do_not(self) -> None:
        panel = clean_panel()
        panel.loc[5, "units"] = -3
        report = check_panel(panel)
        assert not report.passed

        warned = clean_panel()
        promoted = warned["promotion_flag"].astype(bool)
        warned.loc[promoted, "stockout_flag"] = True
        assert check_panel(warned).passed

    def test_messages_carry_warnings_and_failures(self) -> None:
        panel = clean_panel()
        panel.loc[5, "units"] = -3
        messages = check_panel(panel).messages()
        assert any("negative_units" in m for m in messages)
