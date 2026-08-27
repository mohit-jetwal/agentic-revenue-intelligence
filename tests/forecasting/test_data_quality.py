"""Forecasting data-quality checks.

Every check here is run twice: once against a clean panel, where it must pass,
and once against a **deliberately corrupted** one, where it must fire. The second
half is the part that matters. A check that has only ever returned PASS is
indistinguishable from a check that returns PASS unconditionally, and the whole
point of this module is to catch problems nobody went looking for.

The corruptions are chosen to be realistic rather than absurd - a dropped row, a
duplicated key, a price in the wrong units - because those are the failures that
actually reach a warehouse table without anyone noticing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.forecasting.quality import Status, check_panel, missing_value_summary

pytestmark = pytest.mark.models


@pytest.fixture
def clean_panel() -> pd.DataFrame:
    """A small, deliberately well-formed panel."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    for product in ("P001", "P002"):
        for store in ("S001", "S002"):
            for index, day in enumerate(dates):
                rows.append(
                    {
                        "date": day,
                        "product_id": product,
                        "store_id": store,
                        "units": 40 + (index % 7) * 3,
                        "selling_price": 5.0,
                        "regular_price": 5.0,
                        "promotion_flag": False,
                        "promotion_discount": 0.0,
                        "stockout_flag": False,
                    }
                )
    return pd.DataFrame(rows)


def _status(report, name: str) -> Status:
    return next(c.status for c in report.checks if c.name == name)


class TestCleanPanel:
    def test_a_well_formed_panel_passes_everything(self, clean_panel) -> None:
        report = check_panel(clean_panel)

        assert report.ok
        assert not report.failed

    def test_shape_is_reported(self, clean_panel) -> None:
        report = check_panel(clean_panel)

        assert _status(report, "products") is Status.PASS
        assert _status(report, "stores") is Status.PASS
        assert _status(report, "series") is Status.PASS

    def test_every_check_carries_a_reason(self, clean_panel) -> None:
        """A verdict with no explanation cannot be acted on.

        Each check states the *forecasting* consequence, not a generic
        data-hygiene one - that is what makes this module worth having
        separately from the dataset contract checks.
        """
        report = check_panel(clean_panel)

        for check in report.checks:
            assert check.why, f"{check.name} has no stated reason"
            assert len(check.why) > 30


class TestChecksActuallyFire:
    """The half that proves the checks are not decorative."""

    def test_duplicate_grain_is_caught(self, clean_panel) -> None:
        """A duplicated (product, store, date) row doubles that day's weight in
        every lag and rolling window, and the self-join then emits two training
        rows for one observation."""
        corrupted = pd.concat([clean_panel, clean_panel.head(3)], ignore_index=True)

        report = check_panel(corrupted)

        assert _status(report, "duplicate_grain") is Status.FAIL
        assert not report.ok

    def test_missing_dates_are_caught(self, clean_panel) -> None:
        """The check most specific to forecasting.

        A gap violates no schema, but every lag silently shifts across it -
        ``lag_7`` reaches eight days back instead of seven.
        """
        # Drop a fortnight from the middle of one series.
        mask = (
            (clean_panel["product_id"] == "P001")
            & (clean_panel["store_id"] == "S001")
            & clean_panel["date"].between("2024-02-01", "2024-02-14")
        )
        corrupted = clean_panel[~mask]

        report = check_panel(corrupted)

        assert _status(report, "missing_dates") in (Status.WARN, Status.FAIL)

    def test_negative_units_are_caught(self, clean_panel) -> None:
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:5], "units"] = -10

        report = check_panel(corrupted)

        assert _status(report, "negative_units") is Status.FAIL
        assert not report.ok

    def test_non_positive_price_is_caught(self, clean_panel) -> None:
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:5], "selling_price"] = 0.0

        report = check_panel(corrupted)

        assert _status(report, "price_positive") is Status.FAIL

    def test_selling_above_regular_is_caught(self, clean_panel) -> None:
        """Inverts the discount features, so the model reads a promotion as a
        price rise."""
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:5], "selling_price"] = 9.0

        report = check_panel(corrupted)

        assert _status(report, "selling_not_above_regular") is Status.WARN

    def test_extreme_price_jumps_are_caught(self, clean_panel) -> None:
        """A tenfold overnight move is a units or currency error, not a pricing
        decision."""
        corrupted = clean_panel.sort_values(["product_id", "store_id", "date"]).copy()
        corrupted.iloc[10, corrupted.columns.get_loc("selling_price")] = 500.0

        report = check_panel(corrupted)

        assert _status(report, "price_jumps") is Status.WARN

    def test_excessive_zero_sales_is_flagged(self, clean_panel) -> None:
        """Zero-inflation decides which metrics mean anything."""
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[: int(len(corrupted) * 0.8)], "units"] = 0

        report = check_panel(corrupted)

        assert _status(report, "zero_sales_share") is Status.WARN

    def test_promotion_without_discount_is_flagged(self, clean_panel) -> None:
        """Teaches the model that the flag alone raises demand, which makes it a
        proxy for whatever else happened that day."""
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:20], "promotion_flag"] = True

        report = check_panel(corrupted)

        assert _status(report, "promotion_has_discount") is Status.WARN

    def test_discount_without_flag_is_flagged(self, clean_panel) -> None:
        """An unlabelled promotion lands in the non-promotional baseline and
        inflates it."""
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:20], "promotion_discount"] = 0.25

        report = check_panel(corrupted)

        assert _status(report, "discount_has_flag") is Status.WARN

    def test_empty_panel_fails_rather_than_passing_vacuously(self) -> None:
        """Zero checks over zero rows is not a pass."""
        report = check_panel(pd.DataFrame())

        assert not report.ok


class TestSeverity:
    def test_fail_and_warn_are_distinguished(self, clean_panel) -> None:
        """The distinction is what makes the report actionable.

        WARN means usable with a caveat; FAIL means forecasting from this panel
        would produce numbers nobody should act on. Collapsing them into one
        level would make the report either alarmist or useless.
        """
        corrupted = pd.concat([clean_panel, clean_panel.head(2)], ignore_index=True)
        corrupted.loc[corrupted.index[:20], "promotion_flag"] = True

        report = check_panel(corrupted)

        assert report.failed, "the duplicate should be a FAIL"
        assert report.warned, "the promotion inconsistency should be a WARN"
        assert not report.ok

    def test_render_explains_only_the_flagged_checks(self, clean_panel) -> None:
        corrupted = clean_panel.copy()
        corrupted.loc[corrupted.index[:5], "units"] = -1

        rendered = check_panel(corrupted).render()

        assert "negative_units" in rendered
        assert "Why these matter" in rendered


class TestMissingValues:
    def test_null_rates_are_reported_worst_first(self, clean_panel) -> None:
        corrupted = clean_panel.copy()
        corrupted["mostly_null"] = None

        summary = missing_value_summary(corrupted)

        assert summary.iloc[0]["column"] == "mostly_null"
        assert bool(summary.iloc[0]["all_null"])

    def test_all_null_column_is_flagged(self, clean_panel) -> None:
        """A column that is entirely null means a join failed - distinct from a
        feature that is legitimately undefined early in a series."""
        corrupted = clean_panel.assign(joined_column=None)

        summary = missing_value_summary(corrupted)

        assert bool(summary[summary["column"] == "joined_column"]["all_null"].iloc[0])
