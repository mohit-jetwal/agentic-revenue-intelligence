"""Treatment, control and washout construction."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.promo_uplift.config import get_promo_uplift_config
from ml.promo_uplift.exceptions import TreatmentDefinitionError
from ml.promo_uplift.treatment import (
    RowRole,
    build_analysis_frame,
    extract_events,
    qualify_events,
    treated_period,
)

pytestmark = pytest.mark.models


def _panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _series(
    start: str,
    days: int,
    *,
    promo_days: tuple[int, int] | None = None,
    promotion_id: str = "P1",
    depth: float = 20.0,
    product: str = "A",
    store: str = "S",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for i, day in enumerate(dates):
        promoted = promo_days is not None and promo_days[0] <= i < promo_days[1]
        rows.append(
            {
                "date": day,
                "product_id": product,
                "store_id": store,
                "units": 10,
                "promotion_id": promotion_id if promoted else None,
                "promotion_flag": promoted,
                "discount_percentage": depth if promoted else 0.0,
                "stockout_flag": False,
            }
        )
    return _panel(rows)


class TestEventExtraction:
    def test_events_are_grouped_by_promotion_id(self) -> None:
        panel = _series("2024-01-01", 30, promo_days=(10, 15))
        events = extract_events(panel)

        assert len(events) == 1
        event = events.iloc[0]
        assert event["promotion_id"] == "P1"
        assert event["start_date"] == pd.Timestamp("2024-01-11")
        assert event["end_date"] == pd.Timestamp("2024-01-15")
        assert event["duration_days"] == 5

    def test_duration_uses_calendar_span_not_row_count(self) -> None:
        """A gap inside an event must not shorten it.

        The promotion ran for its full window whether or not every day made it
        into the panel; counting rows would let a data gap change the treatment
        definition.
        """
        panel = _series("2024-01-01", 30, promo_days=(10, 15))
        panel = panel.drop(panel.index[12]).reset_index(drop=True)

        events = extract_events(panel)
        assert events.iloc[0]["duration_days"] == 5
        assert events.iloc[0]["observed_days"] == 4

    def test_depth_is_converted_to_a_fraction(self) -> None:
        panel = _series("2024-01-01", 30, promo_days=(10, 15), depth=25.0)
        assert extract_events(panel).iloc[0]["discount_depth"] == pytest.approx(0.25)

    def test_panel_without_promotion_id_is_refused(self) -> None:
        panel = _series("2024-01-01", 10).drop(columns=["promotion_id"])
        with pytest.raises(TreatmentDefinitionError, match="promotion_id"):
            extract_events(panel)


class TestQualification:
    def test_shallow_events_are_disqualified(self) -> None:
        config = get_promo_uplift_config()
        panel = _series("2024-01-01", 30, promo_days=(10, 20), depth=25.0)
        shallow = _series(
            "2024-01-01", 30, promo_days=(10, 20), depth=2.0,
            promotion_id="P2", product="B",
        )
        events = extract_events(pd.concat([panel, shallow], ignore_index=True))

        qualified, reasons = qualify_events(events, config=config)
        assert reasons["too_shallow"] == 1
        assert qualified.set_index("promotion_id")["qualifies"].to_dict() == {
            "P1": True,
            "P2": False,
        }

    def test_short_events_are_disqualified(self) -> None:
        config = get_promo_uplift_config()
        good = _series("2024-01-01", 30, promo_days=(10, 20))
        brief = _series(
            "2024-01-01", 30, promo_days=(10, 11), promotion_id="P2", product="B"
        )
        events = extract_events(pd.concat([good, brief], ignore_index=True))

        qualified, reasons = qualify_events(events, config=config)
        assert reasons["too_short"] == 1
        assert not qualified.set_index("promotion_id").loc["P2", "qualifies"]

    def test_no_qualifying_events_raises_rather_than_returning_empty(self) -> None:
        """An empty treatment set is a definitional failure, not a zero effect.

        Returning an empty frame would let the pipeline produce "uplift 0%" for
        a configuration that excluded every promotion in the data.
        """
        config = get_promo_uplift_config()
        panel = _series("2024-01-01", 30, promo_days=(10, 20), depth=1.0)
        events = extract_events(panel)

        with pytest.raises(TreatmentDefinitionError, match="no promotion event"):
            qualify_events(events, config=config)


class TestRoles:
    def test_event_days_are_treated(self) -> None:
        panel = _series("2024-01-01", 60, promo_days=(20, 30))
        frame = build_analysis_frame(panel).frame

        treated = frame[frame["role"] == RowRole.TREATED]
        assert len(treated) == 10
        assert treated["date"].min() == pd.Timestamp("2024-01-21")
        assert treated["date"].max() == pd.Timestamp("2024-01-30")

    def test_washout_follows_the_event(self) -> None:
        config = get_promo_uplift_config()
        panel = _series("2024-01-01", 60, promo_days=(20, 30))
        frame = build_analysis_frame(panel, config=config).frame

        washout = frame[frame["role"] == RowRole.WASHOUT]
        assert len(washout) == config.treatment.washout_days
        assert washout["date"].min() == pd.Timestamp("2024-01-31")

    def test_washout_yields_to_a_following_promotion(self) -> None:
        """A second promotion inside the first one's washout is treatment.

        Calling those days "recovery from the previous promotion" would
        attribute one promotion's lift to another's payback.
        """
        first = _series("2024-01-01", 60, promo_days=(20, 25))
        second = first.copy()
        mask = (second["date"] >= "2024-01-28") & (second["date"] <= "2024-02-02")
        second.loc[mask, ["promotion_id", "promotion_flag", "discount_percentage"]] = [
            "P2",
            True,
            20.0,
        ]

        frame = build_analysis_frame(second).frame
        overlap = frame[frame["date"].between("2024-01-28", "2024-02-02")]
        assert set(overlap["role"]) == {RowRole.TREATED}

    def test_washout_rows_are_neither_arm(self) -> None:
        panel = _series("2024-01-01", 60, promo_days=(20, 30))
        result = build_analysis_frame(panel)

        washout_dates = result.frame.loc[
            result.frame["role"] == RowRole.WASHOUT, "date"
        ]
        assert not result.treated["date"].isin(washout_dates).any()
        assert not result.control["date"].isin(washout_dates).any()

    def test_disqualified_promotions_are_excluded_from_both_arms(self) -> None:
        """A 2% discount is not treatment, and it is not a clean control either."""
        good = _series("2024-01-01", 60, promo_days=(20, 30))
        shallow = _series(
            "2024-01-01", 60, promo_days=(40, 50), depth=1.0,
            promotion_id="P2", product="B",
        )
        result = build_analysis_frame(pd.concat([good, shallow], ignore_index=True))

        excluded = result.frame[result.frame["role"] == RowRole.EXCLUDED]
        assert len(excluded) == 10
        assert set(excluded["promotion_id"]) == {"P2"}


class TestGrain:
    def test_duplicate_grain_is_refused(self) -> None:
        panel = _series("2024-01-01", 30, promo_days=(10, 20))
        duplicated = pd.concat([panel, panel.iloc[[5]]], ignore_index=True)

        with pytest.raises(TreatmentDefinitionError, match="duplicate"):
            build_analysis_frame(duplicated)

    def test_treated_period_spans_every_event(self) -> None:
        a = _series("2024-01-01", 90, promo_days=(20, 30))
        b = _series(
            "2024-01-01", 90, promo_days=(60, 70), promotion_id="P2", product="B"
        )
        result = build_analysis_frame(pd.concat([a, b], ignore_index=True))

        start, end = treated_period(result.events)
        assert start == pd.Timestamp("2024-01-21").date()
        # Day index 69 of a range starting 1 January 2024 (a leap year).
        assert end == pd.Timestamp("2024-03-10").date()


class TestWarnings:
    def test_flag_without_promotion_id_warns_about_bias_direction(self) -> None:
        """An unattributable promoted day lands in the control arm.

        It carries its promotional lift with it, which raises the comparison
        baseline and *understates* uplift. The warning has to name the direction
        - "data quality issue" tells a reader nothing they can act on.
        """
        panel = _series("2024-01-01", 60, promo_days=(20, 30))
        panel.loc[40, "promotion_flag"] = True  # flagged, but no id

        result = build_analysis_frame(panel)
        assert any("downward" in w for w in result.warnings)
