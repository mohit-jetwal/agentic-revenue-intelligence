"""Turning an effect into money (brief section 18)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.promo_uplift.business import (
    business_impact,
    event_level_impact,
    promotion_spend,
    unit_economics,
)
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.treatment import AnalysisFrame, RowRole

pytestmark = pytest.mark.models


def _analysis(
    *,
    treated_days: int = 10,
    units: int = 100,
    price: float = 10.0,
    cost: float = 6.0,
    spend: float | None = 500.0,
) -> AnalysisFrame:
    dates = pd.date_range("2024-03-01", periods=treated_days, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "product_id": "A",
            "store_id": "S",
            "units": units,
            "revenue": units * price,
            "cost": units * cost,
            "selling_price": price,
            "promotion_id": "P1",
            "role": pd.Categorical([RowRole.TREATED] * treated_days, categories=list(RowRole)),
            "treatment": True,
        }
    )
    events = pd.DataFrame(
        {
            "promotion_id": ["P1"],
            "product_id": ["A"],
            "store_id": ["S"],
            "start_date": [dates[0]],
            "end_date": [dates[-1]],
            "duration_days": [treated_days],
            "discount_depth": [0.2],
            "qualifies": [True],
        }
    )
    if spend is not None:
        events["promotion_spend"] = [spend]
    return AnalysisFrame(frame=frame, events=events)


def _estimate(ate: float, *, n_treated: int = 10, **kwargs: object) -> EffectEstimate:
    return EffectEstimate(
        method="augmented_ipw",
        ate=ate,
        ate_pct=0.2,
        baseline_units=100.0,
        n_treated=n_treated,
        n_control=400,
        **kwargs,  # type: ignore[arg-type]
    )


class TestUnitEconomics:
    def test_margin_is_taken_at_the_promotional_price(self) -> None:
        """The incremental units were sold at a discount, so they earn the
        discounted margin. Valuing them at full margin is the most common way a
        losing promotion is reported as a winner."""
        price, margin, rate = unit_economics(_analysis(price=8.0, cost=6.0))
        assert price == pytest.approx(8.0)
        assert margin == pytest.approx(2.0)
        assert rate == pytest.approx(0.25)

    def test_falls_back_to_the_configured_rate_without_cost(self) -> None:
        analysis = _analysis()
        analysis.frame = analysis.frame.drop(columns=["cost"])
        _, _, rate = unit_economics(analysis)
        assert rate == pytest.approx(0.30)


class TestSpend:
    def test_spend_is_summed_over_events_not_rows(self) -> None:
        """A per-event total broadcast across a ten-day window would be counted
        ten times, and ROI would come back at a tenth of its true value."""
        total, warnings = promotion_spend(_analysis(treated_days=10, spend=500.0))
        assert total == pytest.approx(500.0)
        assert not warnings

    def test_missing_spend_is_reported_not_assumed_zero(self) -> None:
        total, warnings = promotion_spend(_analysis(spend=None))
        assert total == 0.0
        assert any("ROI cannot be computed" in w for w in warnings)


class TestBusinessImpact:
    def test_incremental_profit_subtracts_spend(self) -> None:
        analysis = _analysis(price=10.0, cost=6.0, spend=100.0)
        impact = business_impact(_estimate(5.0), analysis)

        # 5 units/day x 10 treated days x 4.00 margin = 200, less 100 spend.
        assert impact.incremental_units == pytest.approx(50.0)
        assert impact.incremental_margin == pytest.approx(200.0)
        assert impact.incremental_profit == pytest.approx(100.0)

    def test_roi_is_profit_over_spend(self) -> None:
        impact = business_impact(_estimate(5.0), _analysis(spend=100.0))
        assert impact.roi == pytest.approx(1.0)
        assert impact.breaks_even

    def test_roi_is_none_without_spend_not_infinity(self) -> None:
        """A display-only mechanic has no return on investment. It has an
        incremental profit, which is reported."""
        impact = business_impact(_estimate(5.0), _analysis(spend=None))
        assert impact.roi is None
        assert impact.incremental_profit > 0

    def test_negative_uplift_produces_negative_profit(self) -> None:
        """Not floored. A promotion that reduced volume is a real finding and
        the one Step 8 most needs to act on."""
        impact = business_impact(_estimate(-3.0), _analysis(spend=100.0))
        assert impact.incremental_units < 0
        assert impact.incremental_profit < 0
        assert impact.roi is not None and impact.roi < 0
        assert not impact.profitable
        assert any("reduced volume" in w for w in (impact.warnings or []))

    def test_value_destroying_promotion_is_flagged(self) -> None:
        """Positive uplift, negative ROI - the case the whole capability exists
        to surface."""
        impact = business_impact(_estimate(1.0), _analysis(spend=1000.0))
        assert impact.incremental_units > 0
        assert impact.roi is not None and impact.roi < 1.0
        assert any("below break-even" in w for w in (impact.warnings or []))

    def test_profit_bounds_come_from_the_effect_interval(self) -> None:
        impact = business_impact(
            _estimate(5.0, ci_lower=3.0, ci_upper=7.0), _analysis(spend=100.0)
        )
        assert impact.profit_lower is not None
        assert impact.profit_upper is not None
        assert impact.profit_lower < impact.incremental_profit < impact.profit_upper

    def test_cannibalisation_is_declared_as_a_gap(self) -> None:
        """Profit here is an upper bound on category profit, and every result
        has to say so."""
        impact = business_impact(_estimate(5.0), _analysis())
        assert any("Cannibalisation" in a for a in (impact.assumptions or []))

    def test_margin_assumption_states_the_realised_rate(self) -> None:
        impact = business_impact(_estimate(5.0), _analysis(price=10.0, cost=6.0))
        assert any("promotional margin" in a for a in (impact.assumptions or []))


class TestEventLevelImpact:
    def test_one_row_per_promotion(self) -> None:
        analysis = _analysis(treated_days=10)
        cate = np.full(10, 5.0)
        frame = event_level_impact(cate, analysis)

        assert len(frame) == 1
        assert frame.iloc[0]["promotion_id"] == "P1"
        assert frame.iloc[0]["treated_days"] == 10
        assert frame.iloc[0]["incremental_units"] == pytest.approx(50.0)

    def test_value_destroying_events_are_kept_not_filtered(self) -> None:
        """The optimiser's job is partly to allocate *away* from them, which it
        cannot do if they are missing from the table."""
        analysis = _analysis(treated_days=10, spend=10_000.0)
        frame = event_level_impact(np.full(10, 1.0), analysis)

        assert frame.iloc[0]["value_destroying"]
        assert frame.iloc[0]["incremental_profit"] < 0

    def test_mismatched_cate_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="treated_rows has"):
            event_level_impact(np.zeros(3), _analysis(treated_days=10))
