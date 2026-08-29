"""The golden set, the scorer and the regression check.

The scorer is the instrument every later claim about the agent rests on, so it
gets tested the way an instrument does: fed known inputs, checked for the answer
that must come out. A scorer nobody verified turns "the agent scores 0.83" into
a number with no evidence behind it.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.schemas.agent_state import Recommendation, new_agent_state
from app.schemas.domain import IntentType
from app.schemas.tool_contract import ToolErrorCode, ToolResult
from evaluation.baseline_planner import classify, select_tools
from evaluation.golden_set import (
    GoldenQuestion,
    build_golden_set,
    coverage_summary,
    load_golden_set,
)
from evaluation.runner import (
    EvaluationRun,
    baseline_path,
    compare_to_baseline,
    load_baseline,
    write_baseline,
)
from evaluation.scoring import score_question, score_run

pytestmark = pytest.mark.agents

ALL_TOOLS = {
    "estimate_promo_uplift",
    "estimate_price_elasticity",
    "forecast_demand",
    "allocate_promotion_budget",
    "optimize_price",
    "simulate_scenario",
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def scenario(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "scenario_id": "B9",
        "label": "successful_promo",
        "product_ids": ["P00091"],
        "store_ids": [],
        "region": None,
        "start_date": "2024-04-08",
        "end_date": "2024-04-28",
    }
    base.update(overrides)
    return base


def uplift(units: float = 4200.0, roi: float | None = 1.8) -> ToolResult:
    return ToolResult.success(
        tool_name="estimate_promo_uplift",
        result={"incremental_units": units, "incremental_profit": -90_000.0, "roi": roi},
    )


def state_with(*results: ToolResult, recommendation: Recommendation | None = None):
    state = new_agent_state("q")
    state["tool_results"] = list(results)
    state["final_recommendation"] = recommendation
    return state


def recommendation(
    *, action: str = "Proceed.", confidence: float = 0.8, risks: list[str] | None = None
) -> Recommendation:
    return Recommendation(
        executive_summary="s",
        recommended_action=action,
        confidence=confidence,
        risks=risks or [],
    )


def question(**overrides: object) -> GoldenQuestion:
    base: dict[str, object] = {
        "question_id": "Q-B9",
        "scenario_id": "B9",
        "label": "successful_promo",
        "question": "Did the promotion work?",
        "expected_intent": IntentType.PROMOTION_DECISION,
        "required_tools": frozenset({"estimate_promo_uplift"}),
        "expected_direction": "positive",
    }
    base.update(overrides)
    return GoldenQuestion(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The golden set is derived, not invented
# --------------------------------------------------------------------------


class TestGoldenSet:
    def test_questions_carry_the_scenario_they_came_from(self) -> None:
        """A failing question has to be traceable to the truth it was built
        from, or the failure cannot be diagnosed."""
        built = build_golden_set([scenario()])

        assert built[0].scenario_id == "B9"
        assert "P00091" in built[0].question
        assert built[0].start_date == date(2024, 4, 8)

    def test_unmapped_labels_are_skipped_not_guessed(self) -> None:
        """An unmapped label means nobody decided what a right answer looks
        like. Grading against an undecided expectation is worse than not
        grading."""
        assert build_golden_set([scenario(label="something_new")]) == []

    def test_scenarios_without_a_tool_are_marked_unanswerable(self) -> None:
        built = build_golden_set([scenario(label="stockout", scenario_id="D9")])

        assert not built[0].answerable
        assert built[0].coverage_gap
        assert "availability" in built[0].coverage_gap

    def test_entities_are_carried_for_scoping(self) -> None:
        built = build_golden_set(
            [scenario(label="regional_shock", region="North", product_ids=[])]
        )

        assert built[0].entities()["regions"] == ["North"]

    def test_the_real_set_loads_and_covers_both_kinds(self) -> None:
        """Guards the actual file: if Step 2's scenario record changes shape,
        this fails here rather than silently grading nothing."""
        loaded = load_golden_set()
        summary = coverage_summary(loaded)

        assert summary["answerable"] > 0
        assert summary["abstention_expected"] > 0
        assert summary["questions"] == len(loaded)

    def test_coverage_gaps_are_counted_by_label(self) -> None:
        summary = coverage_summary(load_golden_set())

        # Stated as a fact about the current toolset, so adding a diagnostic
        # tool later makes this fail and forces the golden set to be revisited.
        assert set(summary["coverage_gaps"]) == {
            "stockout",
            "competitor_price_cut",
            "regional_shock",
        }


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


class TestToolSelection:
    def test_calling_the_required_tool_scores_full(self) -> None:
        score = score_question(question(), state_with(uplift()))

        assert score.tool_selection == 1.0
        assert score.missing_tools == ()

    def test_missing_the_required_tool_scores_zero(self) -> None:
        score = score_question(question(), state_with())

        assert score.tool_selection == 0.0
        assert score.missing_tools == ("estimate_promo_uplift",)

    def test_fanning_out_costs_points(self) -> None:
        """Calling every tool available looks thorough and is the cheapest way
        to appear rigorous without being it."""
        forecast = ToolResult.success(
            tool_name="forecast_demand", result={"total_predicted_units": 10.0}
        )
        score = score_question(
            question(discouraged_tools=frozenset({"forecast_demand"})),
            state_with(uplift(), forecast),
        )

        assert score.tool_selection < 1.0
        assert score.extra_tools == ("forecast_demand",)


class TestEvidence:
    def test_a_starved_tool_is_not_a_planning_error(self) -> None:
        """The distinction that keeps a data gap from being read as a reasoning
        gap - they have different fixes and the report must not conflate them."""
        failed = ToolResult.failure(
            tool_name="estimate_promo_uplift",
            code=ToolErrorCode.INSUFFICIENT_DATA,
            message="no analysed promotion matches this request",
        )
        score = score_question(question(), state_with(failed))

        # The plan was right; the artefact did not cover it.
        assert score.tool_selection == 1.0
        assert score.evidence == 0.0
        assert score.starved_tools == ("estimate_promo_uplift",)
        assert "coverage gap" in score.notes[0]


class TestDirection:
    def test_volume_is_read_before_profit(self) -> None:
        """Step 7 established that promotional spend in this dataset runs ~20x
        the achievable margin, so incremental profit is negative even for
        promotions injected as successful. The scenario record fixes the volume
        sign, so that is what gets graded."""
        # Profit is negative here and units are positive.
        score = score_question(question(), state_with(uplift(units=4200.0)))

        assert score.direction == 1.0

    def test_the_wrong_sign_scores_zero(self) -> None:
        score = score_question(question(), state_with(uplift(units=-500.0)))

        assert score.direction == 0.0

    def test_no_evidence_scores_zero(self) -> None:
        assert score_question(question(), state_with()).direction == 0.0

    def test_a_scenario_fixing_no_direction_credits_any_result(self) -> None:
        score = score_question(
            question(expected_direction=None), state_with(uplift())
        )

        assert score.direction == 1.0


class TestTheTrap:
    """`bad_promo`: uplift is genuinely positive while the promotion destroys
    value, so every easy reading gives the wrong recommendation."""

    def _question(self) -> GoldenQuestion:
        return question(
            label="bad_promo", expected_direction="positive_uplift_negative_value"
        )

    def test_advising_against_scores_full(self) -> None:
        score = score_question(
            self._question(),
            state_with(
                uplift(roi=0.4),
                recommendation=recommendation(action="Do not repeat this promotion."),
            ),
        )

        assert score.direction == 1.0

    def test_proceeding_on_positive_uplift_earns_half(self) -> None:
        """The evidence was gathered correctly and the conclusion drawn from it
        is wrong - which is a different failure from never looking."""
        score = score_question(
            self._question(),
            state_with(
                uplift(roi=0.4),
                recommendation=recommendation(action="Proceed and repeat it."),
            ),
        )

        assert score.direction == 0.5

    def test_without_the_roi_evidence_it_scores_zero(self) -> None:
        score = score_question(
            self._question(),
            state_with(uplift(roi=None), recommendation=recommendation()),
        )

        assert score.direction == 0.0


class TestAbstention:
    """On a question the toolset cannot answer, confidence is the failure."""

    def _question(self) -> GoldenQuestion:
        return question(label="stockout", answerable=False, expected_direction=None)

    def test_declining_scores_full(self) -> None:
        score = score_question(self._question(), state_with(uplift()))

        assert score.abstention == 1.0
        assert score.applicable == ("abstention",)

    def test_a_confident_answer_scores_zero(self) -> None:
        score = score_question(
            self._question(),
            state_with(uplift(), recommendation=recommendation(confidence=0.8)),
        )

        assert score.abstention == 0.0
        assert "cannot establish" in score.notes[0]

    def test_hedging_earns_most_of_the_credit(self) -> None:
        score = score_question(
            self._question(),
            state_with(
                uplift(),
                recommendation=recommendation(
                    confidence=0.3, risks=["no availability data was available"]
                ),
            ),
        )

        assert score.abstention == 0.75

    def test_direction_is_not_applicable(self) -> None:
        """Averaging a structural zero into an unanswerable question would
        report a coverage gap as a failure of reasoning."""
        score = score_question(self._question(), state_with())

        assert "direction" not in score.applicable
        assert score.overall == score.abstention


# --------------------------------------------------------------------------
# Aggregation and the committed baseline
# --------------------------------------------------------------------------


class TestRunScore:
    def _run(self) -> object:
        answerable = (question(), state_with(uplift()))
        unanswerable = (
            question(question_id="Q-D9", label="stockout", answerable=False),
            state_with(uplift(), recommendation=recommendation(confidence=0.9)),
        )
        return score_run([answerable, unanswerable], provider="test")

    def test_the_two_headline_numbers_stay_separate(self) -> None:
        """Collapsing capability and abstention into one figure would let a
        system that answers everything trade a coverage failure against a
        reasoning success."""
        run = self._run()

        assert run.answerable_mean == 1.0  # type: ignore[attr-defined]
        assert run.abstention_mean == 0.0  # type: ignore[attr-defined]

    def test_artefact_gaps_are_reported_beside_the_score(self) -> None:
        failed = ToolResult.failure(
            tool_name="estimate_promo_uplift",
            code=ToolErrorCode.INSUFFICIENT_DATA,
            message="no data",
        )
        run = score_run([(question(), state_with(failed))], provider="test")

        assert run.artefact_gaps() == {"estimate_promo_uplift": 1}


class TestBaseline:
    def _run(self, mean_source: float) -> EvaluationRun:
        results = [(question(), state_with(uplift(units=mean_source)))]
        return EvaluationRun(
            score=score_run(results, provider="test"),
            coverage={},
            failures={},
            provider="test",
            duration_seconds=0.1,
        )

    def test_a_written_baseline_round_trips(self, tmp_path) -> None:
        path = baseline_path("test", tmp_path)
        write_baseline(self._run(100.0), path)

        assert load_baseline("test", path)["provider"] == "test"  # type: ignore[index]

    def test_no_baseline_reports_no_regression(self) -> None:
        assert compare_to_baseline(self._run(100.0), None) == []

    def test_a_drop_is_caught(self) -> None:
        baseline = self._run(100.0).as_dict()
        regressions = compare_to_baseline(self._run(-100.0), baseline)

        assert regressions
        assert any(r.dimension == "answerable_mean" for r in regressions)
        assert regressions[0].delta < 0

    def test_an_unchanged_run_is_not_a_regression(self) -> None:
        baseline = self._run(100.0).as_dict()

        assert compare_to_baseline(self._run(100.0), baseline) == []

    def test_comparing_across_providers_is_refused(self) -> None:
        """A Claude run graded against the keyword floor would report noise as
        improvement, so the mismatch is an error rather than a comparison."""
        baseline = self._run(100.0).as_dict()
        baseline["provider"] = "claude-something"

        with pytest.raises(ValueError, match="recorded for provider"):
            compare_to_baseline(self._run(100.0), baseline)

    def test_the_committed_baseline_matches_the_current_scorer(self) -> None:
        """The point of committing it: a scoring change that silently moves
        every number fails here instead of in an interview."""
        committed = load_baseline("stub+keyword")
        if committed is None:
            pytest.skip("no baseline recorded yet")

        assert committed["provider"] == "stub+keyword"
        assert set(committed["dimensions"]) == {
            "tool_selection",
            "evidence",
            "direction",
            "abstention",
        }
        # Recorded rather than asserted loosely: the keyword policy cannot
        # abstain, and if that ever becomes non-zero the baseline was rewritten
        # by something that was not the policy.
        assert committed["abstention_mean"] == 0.0


# --------------------------------------------------------------------------
# The keyword floor
# --------------------------------------------------------------------------


class TestKeywordBaseline:
    def test_it_routes_obvious_questions(self) -> None:
        assert select_tools("Did the promotion work?", ALL_TOOLS) == [
            "estimate_promo_uplift"
        ]

    def test_it_cannot_tell_that_it_cannot_answer(self) -> None:
        """The floor's defining weakness: a stockout question routes to a tool
        as happily as anything else, which is why every abstention question is
        a loss for it."""
        selected = select_tools(
            "Sales of P00245 fell sharply. What caused the decline?", ALL_TOOLS
        )

        assert selected  # it answers rather than declining
        assert "estimate_promo_uplift" not in selected

    def test_it_fans_out_when_several_keywords_match(self) -> None:
        selected = select_tools(
            "Should we run the promotion or cut the price?", ALL_TOOLS
        )

        assert len(selected) > 1

    def test_classification_is_crude_by_design(self) -> None:
        assert classify("What should we forecast?") is IntentType.FORECAST
        assert classify("Why did sales fall?") is IntentType.ROOT_CAUSE
