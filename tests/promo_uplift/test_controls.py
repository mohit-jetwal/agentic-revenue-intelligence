"""Control pool construction and the refusals that protect it."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.controls import ControlPool, build_control_pool, days_to_nearest_event
from ml.promo_uplift.exceptions import NoControlGroupError
from ml.promo_uplift.treatment import AnalysisFrame, RowRole, build_analysis_frame

pytestmark = pytest.mark.models


def _panel(days: int = 200, *, promo: tuple[int, int] = (100, 115)) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for product in ("A", "B"):
        for i, day in enumerate(dates):
            promoted = product == "A" and promo[0] <= i < promo[1]
            rows.append(
                {
                    "date": day,
                    "product_id": product,
                    "store_id": "S",
                    "category": "Snacks",
                    "region": "North",
                    "units": 10,
                    "promotion_id": "P1" if promoted else None,
                    "promotion_flag": promoted,
                    "discount_percentage": 20.0 if promoted else 0.0,
                    "stockout_flag": False,
                }
            )
    return pd.DataFrame(rows)


class TestPoolComposition:
    def test_treated_and_control_are_both_present(self, pool: ControlPool) -> None:
        assert pool.treated_rows > 0
        assert pool.control_rows > 0
        assert pool.frame["treatment"].sum() == pool.treated_rows

    def test_within_series_controls_are_close_in_time(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        """Close in time, so the seasonal position is comparable."""
        result = build_control_pool(analysis, config=confounded_config)
        within = result.frame[result.frame["control_origin"] == "within_series"]
        if within.empty:
            pytest.skip("no within-series controls")

        gaps = within["_days_to_event"]
        assert gaps.max() <= confounded_config.controls.same_series_window_days

    def test_washout_rows_never_become_controls(self, pool: ControlPool) -> None:
        """Washout rows are depressed *by* the treatment.

        Using them as controls deflates the baseline and inflates uplift - the
        exact error the pull-forward window exists to prevent.
        """
        assert RowRole.WASHOUT not in set(pool.frame["role"])

    def test_cross_sectional_controls_come_from_never_treated_listings(self) -> None:
        panel = _panel()
        result = build_analysis_frame(panel)
        pool = build_control_pool(result)

        cross = pool.frame[pool.frame["control_origin"] == "cross_sectional"]
        assert not cross.empty
        assert set(cross["product_id"]) == {"B"}

    def test_cross_sectional_controls_are_contemporaneous(self) -> None:
        panel = _panel()
        result = build_analysis_frame(panel)
        pool = build_control_pool(result)

        treated = pool.frame[pool.frame["treatment"]]
        cross = pool.frame[pool.frame["control_origin"] == "cross_sectional"]
        assert cross["date"].min() >= treated["date"].min()
        assert cross["date"].max() <= treated["date"].max()


class TestDistanceToEvent:
    def test_zero_inside_an_event(self) -> None:
        panel = _panel()
        result = build_analysis_frame(panel)
        gaps = days_to_nearest_event(result.frame, result.events)

        treated = result.frame["role"] == RowRole.TREATED
        assert (gaps[treated] == 0).all()

    def test_null_for_never_treated_listings(self) -> None:
        """NaN distinguishes a never-treated listing from one merely far from
        its own promotions - and that distinction is what selects the
        cross-sectional pool."""
        panel = _panel()
        result = build_analysis_frame(panel)
        gaps = days_to_nearest_event(result.frame, result.events)

        never = result.frame["product_id"] == "B"
        assert gaps[never].isna().all()

    def test_counts_days_either_side(self) -> None:
        panel = _panel(promo=(100, 115))
        result = build_analysis_frame(panel)
        gaps = days_to_nearest_event(result.frame, result.events)

        frame = result.frame.assign(gap=gaps)
        listing = frame[frame["product_id"] == "A"].sort_values("date")
        # Day 99 is one day before the event starts; day 115 is one day after
        # it ends.
        assert listing.iloc[99]["gap"] == 1
        assert listing.iloc[115]["gap"] == 1


class TestRefusals:
    def test_too_few_controls_is_refused(self) -> None:
        """Below the floor an estimate is arithmetic, not inference."""
        config = get_promo_uplift_config().model_copy(
            update={
                "controls": get_promo_uplift_config().controls.model_copy(
                    update={
                        "min_control_rows": 10_000,
                        "use_cross_sectional_controls": False,
                    }
                )
            }
        )
        panel = _panel()
        result = build_analysis_frame(panel, config=config)

        with pytest.raises(NoControlGroupError, match="eligible control rows"):
            build_control_pool(result, config=config)

    def test_refusal_reports_what_was_found(self) -> None:
        config = get_promo_uplift_config().model_copy(
            update={
                "controls": get_promo_uplift_config().controls.model_copy(
                    update={
                        "min_control_rows": 10_000,
                        "use_cross_sectional_controls": False,
                    }
                )
            }
        )
        result = build_analysis_frame(_panel(), config=config)

        with pytest.raises(NoControlGroupError) as caught:
            build_control_pool(result, config=config)

        detail = caught.value.detail
        assert detail["required_control_rows"] == 10_000
        assert detail["control_rows"] < 10_000
        assert caught.value.recoverable is True

    def test_refusal_suggests_a_recovery(self) -> None:
        """A recoverable error must say what would have worked."""
        config = get_promo_uplift_config().model_copy(
            update={
                "controls": get_promo_uplift_config().controls.model_copy(
                    update={
                        "min_control_rows": 10_000,
                        "use_cross_sectional_controls": False,
                    }
                )
            }
        )
        result = build_analysis_frame(_panel(), config=config)

        with pytest.raises(NoControlGroupError) as caught:
            build_control_pool(result, config=config)
        assert "same_series_window_days" in str(caught.value)


class TestWarnings:
    def test_missing_cross_sectional_pool_is_reported(
        self, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
    ) -> None:
        """Every listing promoted at some point means no never-treated control.

        The comparison then rests entirely on other days of the same listing,
        and anything shared by all listings at that time is inseparable from the
        promotion.
        """
        pool = build_control_pool(analysis, config=confounded_config)
        if pool.cross_sectional_rows == 0:
            assert any("never-treated" in w for w in pool.warnings)
