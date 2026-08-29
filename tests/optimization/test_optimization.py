"""Budget allocation, price optimisation and scenario projection."""

from __future__ import annotations

import pandas as pd
import pytest

from app.schemas.domain import RiskLevel
from ml.price_optimization.optimizer import (
    DEFAULT_GRID,
    evaluate_candidates,
    recommend,
)
from ml.scenario.engine import Inputs, Lever, project
from ml.trade_promo_optimization.optimizer import (
    Candidate,
    allocate,
    candidates_from_events,
)

pytestmark = pytest.mark.models


def cells() -> list[Candidate]:
    """Three cells with clearly different efficiency."""
    return [
        Candidate("A", "P1", region="North", reference_profit=300.0, reference_spend=100.0),
        Candidate("B", "P2", region="North", reference_profit=200.0, reference_spend=100.0),
        Candidate("C", "P3", region="South", reference_profit=50.0, reference_spend=100.0),
    ]


class TestAllocation:
    def test_concavity_spreads_the_budget(self) -> None:
        """The property the whole design exists for.

        A linear model over constant ROI-per-rupee puts everything into the
        single best cell, which is wrong and obviously wrong to any category
        manager. Diminishing returns are what make it spread.
        """
        outcome = allocate(cells(), total_budget=150.0)
        assert len(outcome.lines) >= 2
        assert outcome.lines["allocated_spend"].max() < 150.0

    def test_prefers_the_more_efficient_cell(self) -> None:
        outcome = allocate(cells(), total_budget=150.0)
        spend = dict(
            zip(outcome.lines["candidate_id"], outcome.lines["allocated_spend"], strict=True)
        )
        assert spend["A"] > spend.get("B", 0.0)

    def test_never_exceeds_the_budget(self) -> None:
        for budget in (50.0, 150.0, 400.0):
            outcome = allocate(cells(), total_budget=budget)
            assert outcome.allocated <= budget + 1e-6

    def test_stops_when_marginal_return_dies(self) -> None:
        """A saturating curve means extra budget eventually buys nothing, and
        the optimiser should decline to spend it rather than pad the allocation."""
        outcome = allocate(cells(), total_budget=100_000.0)
        assert outcome.allocated < 100_000.0
        assert any("no positive-return home" in w for w in outcome.warnings)

    def test_value_destroying_cells_are_not_funded(self) -> None:
        candidates = [
            *cells(),
            Candidate("D", "P4", reference_profit=-500.0, reference_spend=100.0),
        ]
        outcome = allocate(candidates, total_budget=300.0)
        assert "D" not in set(outcome.lines["candidate_id"])

    def test_marginal_roi_decreases_with_spend(self) -> None:
        outcome = allocate(cells(), total_budget=250.0)
        line = outcome.lines.iloc[0]
        assert line["marginal_roi"] < line["expected_roi"]

    def test_no_cell_is_funded_beyond_its_observed_range(self) -> None:
        """Extrapolating a saturating curve past any observed spend produces a
        confident recommendation nobody should act on."""
        outcome = allocate(cells(), total_budget=10_000.0, max_cell_multiple=2.0)
        assert (outcome.lines["spend_vs_reference"] <= 2.0 + 1e-6).all()


class TestConstraints:
    def test_regional_minimum_is_honoured(self) -> None:
        outcome = allocate(
            cells(), total_budget=200.0, dimension_limits={("region", "South"): (80.0, None)}
        )
        south = outcome.lines.loc[outcome.lines["region"] == "South", "allocated_spend"].sum()
        assert south >= 79.9

    def test_maximum_is_honoured(self) -> None:
        outcome = allocate(
            cells(), total_budget=200.0, dimension_limits={("region", "North"): (None, 50.0)}
        )
        north = outcome.lines.loc[outcome.lines["region"] == "North", "allocated_spend"].sum()
        assert north <= 50.1

    def test_binding_constraints_are_reported(self) -> None:
        outcome = allocate(
            cells(), total_budget=200.0, dimension_limits={("region", "South"): (80.0, None)}
        )
        assert any("South" in c for c in outcome.binding_constraints)

    def test_infeasible_reports_the_conflict_rather_than_raising(self) -> None:
        """'Your minimum spends already exceed the budget' is a business
        finding the agent should surface, not a crash."""
        outcome = allocate(
            cells(),
            total_budget=50.0,
            dimension_limits={
                ("region", "North"): (100.0, None),
                ("region", "South"): (100.0, None),
            },
        )
        assert outcome.status == "infeasible"
        assert any("exceeds the budget" in c for c in outcome.binding_constraints)

    def test_a_constraint_matching_nothing_is_reported_not_dropped(self) -> None:
        """The bug this test exists for.

        A caller who asks for 900,000 in a region that no candidate carries and
        receives an allocation with none of it has been misled twice: the
        constraint was ignored, and the result looked honoured.
        """
        outcome = allocate(
            cells(), total_budget=200.0, dimension_limits={("region", "Mars"): (100.0, None)}
        )
        assert any("matched no candidate" in w for w in outcome.warnings)
        assert not any("Mars" in c for c in outcome.binding_constraints)


class TestCandidatesFromEvents:
    def _events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "promotion_id": ["E1", "E2", "E3"],
                "product_id": ["P1", "P2", "P3"],
                "incremental_profit": [500.0, 100.0, -50.0],
                "promotion_spend": [100.0, 100.0, 100.0],
                "incremental_units": [50.0, 20.0, -5.0],
                "incremental_revenue": [900.0, 300.0, -80.0],
            }
        )

    def test_ranks_by_efficiency_not_absolute_profit(self) -> None:
        """With a fixed budget the question is what each rupee buys."""
        candidates = candidates_from_events(self._events(), max_candidates=2)
        assert [c.candidate_id for c in candidates] == ["E1", "E2"]

    def test_keeps_value_destroying_events(self) -> None:
        """The optimiser's job is partly to allocate away from them."""
        candidates = candidates_from_events(self._events())
        assert "E3" in {c.candidate_id for c in candidates}

    def test_events_with_no_spend_are_unfundable(self) -> None:
        events = self._events()
        events["promotion_spend"] = 0.0
        assert candidates_from_events(events) == []

    def test_missing_columns_are_refused(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            candidates_from_events(pd.DataFrame({"promotion_id": ["E1"]}))

    def test_empty_input_yields_no_candidates(self) -> None:
        assert candidates_from_events(pd.DataFrame()) == []


class TestPriceOptimization:
    def test_matches_the_closed_form_optimum(self) -> None:
        """For constant elasticity, p* = c·e/(1+e). At e=-2 and c=60 that is 120.

        Checked against theory rather than against a previous run, so a
        regression in the grid or the profit function is caught rather than
        enshrined.
        """
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-2.0
        )
        result = recommend("P1", candidates, current_price=100.0, elasticity=-2.0)
        assert result.recommended_price == pytest.approx(120.0, abs=5.0)

    def test_inelastic_demand_wants_a_rise(self) -> None:
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-0.5
        )
        result = recommend("P2", candidates, current_price=100.0, elasticity=-0.5)
        assert result.change_pct > 0

    def test_returns_a_range_not_just_a_point(self) -> None:
        """An optimum from an uncertain elasticity is false precision."""
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-2.0
        )
        result = recommend(
            "P1", candidates, current_price=100.0, elasticity=-2.0,
            elasticity_interval=(-2.4, -1.6),
        )
        assert result.recommended_range is not None
        assert result.recommended_range[0] < result.recommended_range[1]

    def test_margin_floor_is_respected(self) -> None:
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0,
            elasticity=-2.0, min_margin_pct=0.45,
        )
        result = recommend("P3", candidates, current_price=100.0, elasticity=-2.0)
        margin = (result.recommended_price - 60.0) / result.recommended_price
        assert margin >= 0.45 - 1e-6

    def test_price_change_cap_is_respected(self) -> None:
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0,
            elasticity=-2.0, max_change_pct=0.05,
        )
        result = recommend("P4", candidates, current_price=100.0, elasticity=-2.0)
        assert abs(result.change_pct) <= 0.05 + 1e-9

    def test_impossible_constraints_report_rather_than_raise(self) -> None:
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0,
            elasticity=-2.0, min_margin_pct=0.99,
        )
        result = recommend("P5", candidates, current_price=100.0, elasticity=-2.0)
        assert result.binding_constraints
        assert any("no candidate price" in w for w in result.warnings)

    def test_warns_when_the_optimum_is_at_the_grid_edge(self) -> None:
        """Then the true optimum lies outside the evaluated range and the
        recommendation is a property of the grid, not the demand curve."""
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-0.3
        )
        result = recommend("P6", candidates, current_price=100.0, elasticity=-0.3)
        assert any("edge of the evaluated range" in w for w in result.warnings)

    def test_substitutes_push_the_optimum_higher(self) -> None:
        """Raising price sends volume to a substitute you also own, so the
        portfolio optimum is a larger rise than the standalone one."""
        alone = recommend(
            "P7",
            evaluate_candidates(
                current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-2.0
            ),
            current_price=100.0, elasticity=-2.0,
        )
        portfolio = recommend(
            "P7",
            evaluate_candidates(
                current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-2.0,
                cross_effects={"P8": (0.8, 800.0, 25.0)},
            ),
            current_price=100.0, elasticity=-2.0,
        )
        assert portfolio.recommended_price >= alone.recommended_price

    def test_wide_elasticity_interval_is_flagged(self) -> None:
        candidates = evaluate_candidates(
            current_price=100.0, current_units=1000.0, unit_cost=60.0, elasticity=-2.0
        )
        result = recommend(
            "P9", candidates, current_price=100.0, elasticity=-2.0,
            elasticity_interval=(-3.0, -1.0),
        )
        assert any("interval spans" in w for w in result.warnings)
        assert result.risk is RiskLevel.HIGH

    def test_grid_covers_the_usual_optimum(self) -> None:
        assert max(DEFAULT_GRID) >= 0.20
        assert min(DEFAULT_GRID) <= -0.20


class TestScenario:
    def _inputs(self, **overrides: object) -> Inputs:
        base = {
            "baseline_units": 500.0,
            "unit_price": 100.0,
            "unit_cost": 60.0,
            "elasticity": -2.0,
            "elasticity_interval": (-2.3, -1.7),
        }
        base.update(overrides)
        return Inputs(**base)  # type: ignore[arg-type]

    def test_price_cut_raises_volume(self) -> None:
        result = project([Lever("price", change_pct=-0.10)], self._inputs())
        assert result.units_impact > 0

    def test_price_rise_reduces_volume(self) -> None:
        result = project([Lever("price", change_pct=0.10)], self._inputs())
        assert result.units_impact < 0

    def test_levers_compose(self) -> None:
        """Effects accumulate in log space, so order does not matter."""
        inputs = self._inputs(promo_uplift=0.30)
        forward = project(
            [Lever("price", change_pct=-0.05), Lever("promotion", change_pct=1.0)], inputs
        )
        reverse = project(
            [Lever("promotion", change_pct=1.0), Lever("price", change_pct=-0.05)], inputs
        )
        assert forward.scenario_units == pytest.approx(reverse.scenario_units)

    def test_confidence_is_the_weakest_link_not_an_average(self) -> None:
        """Averaging 0.85 and 0.40 into 0.62 describes a projection nobody made."""
        result = project(
            [Lever("price", change_pct=-0.05), Lever("competitor_price", change_pct=-0.10)],
            self._inputs(competitor_sensitivity=0.5),
        )
        assert result.confidence == pytest.approx(0.40)

    def test_horizon_scales_the_outcome(self) -> None:
        short = project([Lever("price", change_pct=-0.05)], self._inputs(), horizon_days=7)
        long = project([Lever("price", change_pct=-0.05)], self._inputs(), horizon_days=28)
        assert long.profit_impact == pytest.approx(short.profit_impact * 4)

    def test_range_comes_from_the_input_intervals(self) -> None:
        result = project([Lever("price", change_pct=-0.10)], self._inputs())
        assert result.profit_range is not None
        assert result.profit_range[0] < result.profit_range[1]

    def test_no_interval_is_stated_rather_than_implied(self) -> None:
        result = project(
            [Lever("price", change_pct=-0.10)],
            self._inputs(elasticity_interval=None),
        )
        assert result.profit_range is None
        assert any("no stated uncertainty" in w for w in result.warnings)

    def test_unmodelled_lever_is_reported_not_ignored(self) -> None:
        result = project([Lever("inventory", change_pct=0.1)], self._inputs())
        assert any("not modelled" in w for w in result.warnings)

    def test_missing_elasticity_is_reported(self) -> None:
        result = project(
            [Lever("price", change_pct=-0.05)], self._inputs(elasticity=None)
        )
        assert any("no elasticity is available" in w for w in result.warnings)

    def test_wide_range_is_high_risk(self) -> None:
        result = project(
            [Lever("price", change_pct=-0.02)],
            self._inputs(elasticity_interval=(-4.0, -0.2)),
        )
        assert result.risk is RiskLevel.HIGH

    def test_contributing_models_are_recorded(self) -> None:
        result = project(
            [Lever("price", change_pct=-0.05), Lever("promotion", change_pct=1.0)],
            self._inputs(promo_uplift=0.3),
        )
        assert set(result.contributing_models) == {"price_elasticity", "promo_uplift"}
