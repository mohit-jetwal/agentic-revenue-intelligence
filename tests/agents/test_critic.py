"""The Critic, output validation, and bounded re-planning.

The property under test throughout is the one the plan named: **no number in a
final recommendation is absent from ``tool_results``** - and where that cannot be
enforced, it is at least declared.

Everything runs against the stub provider, so a scripted verdict makes "what did
the graph do when the Critic said no" an assertion rather than an observation.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.agents.critic import CriticAgent, CriticAssessment
from app.agents.recommendation import (
    DraftRecommendation,
    RecommendationAgent,
    extract_evidence,
)
from app.agents.supervisor import (
    IntentClassification,
    PlannedStep,
    ProposedPlan,
    SupervisorAgent,
)
from app.config.settings import AgentSettings
from app.guardrails.budget import BudgetTracker
from app.guardrails.output_validation import collect_source_values, validate_output
from app.llm.stub import StubProvider
from app.schemas.agent_state import new_agent_state
from app.schemas.domain import BusinessObjective, IntentType
from app.schemas.tool_contract import ToolResult
from app.tools.base import AnalyticalTool, ToolOutput
from app.tools.registry import ToolRegistry
from app.workflows.graph import (
    InvestigationDeps,
    build_graph,
    run_investigation,
    summarise,
)

pytestmark = pytest.mark.agents


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

#: A figure with an awkward rounding, so "1.43M" in prose is a real match rather
#: than a coincidence of round numbers.
PROFIT = 1_427_355.0


class UpliftInput(BaseModel):
    value: str = "x"


class UpliftOutput(BaseModel):
    incremental_profit: float = PROFIT
    roi: float = 0.6761


class UpliftTool(AnalyticalTool[UpliftInput, UpliftOutput]):
    name = "estimate_promo_uplift"
    description = "Estimate incremental profit from a promotion."
    input_schema = UpliftInput
    output_schema = UpliftOutput
    permission = "run_model"

    def _execute(self, payload: UpliftInput) -> ToolOutput[UpliftOutput]:
        return ToolOutput(payload=UpliftOutput())


def uplift_result(**overrides: object) -> ToolResult:
    payload: dict[str, object] = {"incremental_profit": PROFIT, "roi": 0.6761}
    payload.update(overrides)
    return ToolResult.success(tool_name="estimate_promo_uplift", result=payload)


# --------------------------------------------------------------------------
# Output validation - the architectural hallucination control
# --------------------------------------------------------------------------


class TestOutputValidation:
    def test_sourced_figures_pass_through_their_rounding(self) -> None:
        """A well-written sentence rounds. 1,427,355 becomes "1.43M" and 0.6761
        becomes "68%", and flagging either would make the check unusable."""
        report = validate_output(
            "Incremental profit was 1.43M at an ROI of 68%.", [uplift_result()]
        )

        assert report.clean
        assert report.checked == 2

    def test_invented_figure_is_caught(self) -> None:
        report = validate_output(
            "We project 9,999,999 of upside next quarter.", [uplift_result()]
        )

        assert not report.clean
        assert report.unsourced[0].value == pytest.approx(9_999_999.0)
        # The sentence travels with the number so a reviewer sees the claim.
        assert "upside next quarter" in report.unsourced[0].context

    def test_suffixed_figures_are_never_structural(self) -> None:
        """Regression: the bare numeral in "9.99M" is 9.99, which falls under
        the structural floor. Without the suffix the commonest way of writing a
        large invented figure would be skipped rather than checked."""
        report = validate_output("We project 9.99M of upside.", [uplift_result()])

        assert not report.clean
        assert report.unsourced[0].value == pytest.approx(9_990_000.0)

    def test_percentages_are_never_structural(self) -> None:
        """Same hole, other form: 40 is under the floor, but "40%" is a
        measurement, not bookkeeping."""
        report = validate_output("Margin improved by 40%.", [uplift_result()])

        assert not report.clean

    def test_structural_numbers_are_skipped(self) -> None:
        """Years and small counts are never tool outputs. Checking them buries
        the real finding in noise."""
        report = validate_output(
            "Across 2024 and 2025 the top 3 products led.", [uplift_result()]
        )

        assert report.clean
        assert report.checked == 0
        assert report.skipped >= 3

    def test_sources_are_collected_from_the_whole_payload(self) -> None:
        """Walked rather than enumerated: a recommendation may legitimately
        quote an interval bound, and a fixed field list would not have it."""
        result = uplift_result(confidence_interval=[1_200_000.0, 1_650_000.0])
        values = collect_source_values([result])

        assert 1_200_000.0 in values
        assert 1_650_000.0 in values

    def test_a_quoted_interval_bound_validates(self) -> None:
        """The "95%" names the interval's level, not a finding - exempting
        percentages from the structural floor must not start flagging it."""
        report = validate_output(
            "Profit was 1.43M (95% confidence interval 1.20M to 1.65M).",
            [uplift_result(confidence_interval=[1_200_000.0, 1_650_000.0])],
        )

        assert report.clean

    def test_a_bare_ninety_five_percent_is_still_checked(self) -> None:
        """The confidence-level exemption is scoped by the surrounding words.
        Without them, 95% is an ordinary claim and must be sourced."""
        report = validate_output("Distribution reached 95%.", [uplift_result()])

        assert not report.clean


# --------------------------------------------------------------------------
# The Critic
# --------------------------------------------------------------------------


class TestCritic:
    def test_empty_evidence_needs_no_model_call(self) -> None:
        """Nothing was gathered, so nothing is supported. Asking an LLM to
        confirm that spends a round trip on a decided question."""
        stub = StubProvider()
        verdict = CriticAgent(provider=stub).review("q", [])

        assert not verdict.valid
        assert verdict.confidence == 0.0
        assert stub.calls == []

    def test_failed_validation_overrides_a_confident_model(self) -> None:
        """The mechanical check is decidable and the model's reading is not. A
        result whose own validation_status is `failed` cannot be rescued by a
        confident sentence about it."""
        stub = StubProvider()
        stub.script_structured(CriticAssessment(sufficient=True, confidence=0.95))

        verdict = CriticAgent(provider=stub).review(
            "did it work?", [uplift_result(validation_status="failed")]
        )

        assert not verdict.valid
        assert verdict.confidence <= 0.5
        assert any(issue.startswith("BLOCKING") for issue in verdict.issues)

    def test_warnings_are_raised_without_blocking(self) -> None:
        """A caveat is a matter of degree, so it is surfaced and left to the
        model - unlike a failure, which is not."""
        stub = StubProvider()
        stub.script_structured(CriticAssessment(sufficient=True, confidence=0.8))

        verdict = CriticAgent(provider=stub).review(
            "did it work?", [uplift_result(validation_status="warnings")]
        )

        assert verdict.valid
        assert any("caveats" in issue for issue in verdict.issues)

    def test_mechanical_and_model_issues_are_both_reported(self) -> None:
        stub = StubProvider()
        stub.script_structured(
            CriticAssessment(
                sufficient=False,
                confidence=0.3,
                issues=["no counterfactual was established"],
                required_followup=["run a placebo test"],
            )
        )

        verdict = CriticAgent(provider=stub).review(
            "did it work?", [uplift_result(validation_status="warnings")]
        )

        assert not verdict.valid
        assert len(verdict.issues) == 2
        assert verdict.required_followup == ["run a placebo test"]


# --------------------------------------------------------------------------
# Recommendation synthesis
# --------------------------------------------------------------------------


class TestRecommendation:
    def _agent(self, stub: StubProvider, threshold: float = 1e6) -> RecommendationAgent:
        return RecommendationAgent(provider=stub, approval_threshold=threshold)

    def _draft(self, **overrides: object) -> DraftRecommendation:
        fields: dict[str, object] = {
            "executive_summary": "The promotion produced 1.43M of incremental profit.",
            "recommended_action": "Continue funding it.",
            "confidence": 0.85,
            "estimated_profit_impact": PROFIT,
        }
        fields.update(overrides)
        return DraftRecommendation(**fields)  # type: ignore[arg-type]

    def test_evidence_is_linked_to_its_trace(self) -> None:
        """Built from the results, not from the model's prose. A claim the model
        *said* it sourced and a claim that *is* sourced are different things."""
        result = uplift_result()
        evidence = extract_evidence([result])

        assert evidence[0].source_trace_id == result.trace_id
        assert str(PROFIT) in evidence[0].claim

    def test_repeated_results_do_not_pad_the_evidence(self) -> None:
        """A re-planned investigation reruns the same tool and gets the same
        answer. Listing it three times is the appearance of corroboration
        without any."""
        evidence = extract_evidence([uplift_result() for _ in range(3)])

        assert len(evidence) == 1

    def test_unsourced_figure_caps_confidence_and_is_declared(self) -> None:
        stub = StubProvider()
        stub.script_structured(
            self._draft(
                executive_summary="Profit was 1.43M and next year brings 9,999,999 more."
            )
        )

        recommendation = self._agent(stub).synthesise("q", [uplift_result()])

        assert recommendation.confidence <= 0.6
        assert any("UNSOURCED FIGURES" in risk for risk in recommendation.risks)

    def test_unresolved_critic_objection_caps_confidence(self) -> None:
        """Reached only when the re-plan budget ran out with the objection
        standing. The model's own confidence would present a rejected
        recommendation as settled."""
        stub = StubProvider()
        stub.script_structured(self._draft())

        recommendation = self._agent(stub).synthesise(
            "q",
            [uplift_result()],
            critic_issues=["no counterfactual"],
            critic_satisfied=False,
        )

        assert recommendation.confidence <= 0.5
        assert "no counterfactual" in recommendation.risks

    def test_truncation_is_stated_not_implied(self) -> None:
        stub = StubProvider()
        stub.script_structured(self._draft())

        recommendation = self._agent(stub).synthesise(
            "q", [uplift_result()], truncated=True
        )

        assert recommendation.confidence <= 0.5
        assert any("INCOMPLETE INVESTIGATION" in r for r in recommendation.risks)

    def test_impact_above_threshold_requires_approval(self) -> None:
        stub = StubProvider()
        stub.script_structured(self._draft())

        assert self._agent(stub, threshold=1e6).synthesise(
            "q", [uplift_result()]
        ).requires_human_approval

    def test_impact_below_threshold_does_not(self) -> None:
        stub = StubProvider()
        stub.script_structured(self._draft())

        assert not self._agent(stub, threshold=1e9).synthesise(
            "q", [uplift_result()]
        ).requires_human_approval

    def test_a_loss_of_the_same_size_also_requires_approval(self) -> None:
        """The threshold is on magnitude. A recommendation to give up 1.4M is
        exactly as consequential as one to chase it."""
        stub = StubProvider()
        stub.script_structured(self._draft(estimated_profit_impact=-PROFIT))

        assert self._agent(stub, threshold=1e6).synthesise(
            "q", [uplift_result()]
        ).requires_human_approval

    def test_tool_assumptions_are_carried_verbatim(self) -> None:
        """Dropping them converts a qualified finding into a fact."""
        result = ToolResult.success(
            tool_name="estimate_promo_uplift",
            result={"incremental_profit": PROFIT},
            assumptions=["no unobserved confounding"],
        )
        stub = StubProvider()
        stub.script_structured(self._draft())

        recommendation = self._agent(stub).synthesise("q", [result])

        assert "no unobserved confounding" in recommendation.assumptions


# --------------------------------------------------------------------------
# The graph: re-planning is bounded
# --------------------------------------------------------------------------


def _deps(
    *, critic_sufficient: bool, max_replans: int = 2
) -> tuple[InvestigationDeps, StubProvider]:
    registry = ToolRegistry()
    registry.register(UpliftTool())

    stub = StubProvider()
    stub.script_structured(
        IntentClassification(
            intent=IntentType.PROMOTION_DECISION,
            objective=BusinessObjective.MAXIMISE_PROFIT,
        )
    )
    stub.script_structured(
        ProposedPlan(
            steps=[
                PlannedStep(tool_name="estimate_promo_uplift", rationale="measure it")
            ]
        )
    )
    stub.script_structured(
        CriticAssessment(
            sufficient=critic_sufficient,
            confidence=0.4,
            issues=[] if critic_sufficient else ["the profit figure is unbounded"],
            required_followup=[] if critic_sufficient else ["bound the profit figure"],
        )
    )
    stub.script_structured(
        DraftRecommendation(
            executive_summary=f"The promotion produced {PROFIT:,.0f} of profit.",
            recommended_action="Continue funding it.",
            confidence=0.85,
            estimated_profit_impact=PROFIT,
        )
    )

    deps = InvestigationDeps(
        supervisor=SupervisorAgent(provider=stub, tools=registry.specs()),
        registry=registry,
        budget=BudgetTracker.from_settings(AgentSettings()),
        critic=CriticAgent(provider=stub),
        recommender=RecommendationAgent(provider=stub, approval_threshold=1e6),
        max_replans=max_replans,
    )
    return deps, stub


class TestBoundedReplanning:
    def test_a_never_satisfied_critic_terminates(self) -> None:
        """The realistic way an agent loops forever: a Critic that always wants
        one more thing. The cap, not the Critic, ends the investigation."""
        deps, _ = _deps(critic_sufficient=False, max_replans=2)

        summary = summarise(run_investigation("Did the promotion work?", deps))

        assert summary["replans"] == 2
        # One execution per plan: the original and two re-plans.
        assert summary["tool_calls"] == 3

    def test_a_satisfied_critic_does_not_replan(self) -> None:
        """Minimum sufficient work. Re-planning because the graph has an edge
        for it would burn budget on a question already answered."""
        deps, _ = _deps(critic_sufficient=True)

        summary = summarise(run_investigation("Did the promotion work?", deps))

        assert summary["replans"] == 0
        assert summary["tool_calls"] == 1
        assert summary["critic"]["valid"]

    def test_zero_replans_is_honoured(self) -> None:
        deps, _ = _deps(critic_sufficient=False, max_replans=0)

        summary = summarise(run_investigation("q", deps))

        assert summary["replans"] == 0
        assert summary["tool_calls"] == 1

    def test_the_critic_objection_reaches_the_recommendation(self) -> None:
        """An objection that was never resolved must be visible in the output,
        not silently dropped when the cap ends the loop."""
        deps, _ = _deps(critic_sufficient=False, max_replans=1)

        summary = summarise(run_investigation("Did the promotion work?", deps))
        recommendation = summary["recommendation"]

        assert any(
            "unbounded" in risk for risk in recommendation["risks"]
        ), recommendation["risks"]
        assert recommendation["confidence"] <= 0.5

    def test_the_recommendation_quotes_only_sourced_numbers(self) -> None:
        """The plan's headline property, end to end."""
        deps, _ = _deps(critic_sufficient=True)

        summary = summarise(run_investigation("Did the promotion work?", deps))

        assert not any(
            "UNSOURCED FIGURES" in risk for risk in summary["recommendation"]["risks"]
        )


class TestHumanApproval:
    """The interrupt is what makes approval a gate rather than a label."""

    def _config(self) -> dict[str, object]:
        return {"configurable": {"thread_id": "approval-test"}}

    def test_the_graph_stops_before_recommending(self) -> None:
        """With a checkpointer the graph halts before `recommend`, so a person
        reviews the evidence rather than rubber-stamping a written conclusion."""
        deps, _ = _deps(critic_sufficient=True)
        graph = build_graph(deps, checkpointer=MemorySaver())
        config = self._config()

        state = graph.invoke(
            new_agent_state("Did the promotion work?", max_replans=deps.max_replans),
            config,
        )

        # Evidence gathered, conclusion withheld.
        assert state["tool_results"]
        assert state["final_recommendation"] is None
        assert graph.get_state(config).next == ("recommend",)

    def test_resuming_produces_the_recommendation(self) -> None:
        """Approval is a resume, not a re-run: the evidence already gathered is
        restored from the checkpoint rather than paid for again."""
        deps, _ = _deps(critic_sufficient=True)
        graph = build_graph(deps, checkpointer=MemorySaver())
        config = self._config()

        graph.invoke(
            new_agent_state("Did the promotion work?", max_replans=deps.max_replans),
            config,
        )
        resumed = graph.invoke(None, config)

        assert resumed["final_recommendation"] is not None
        assert resumed["final_recommendation"].requires_human_approval
        # One tool call in total - the interrupt did not cost a second run.
        assert resumed["tool_call_count"] == 1

    def test_without_a_checkpointer_execution_runs_through(self) -> None:
        """The flag is still set, but nothing blocks on it. Stating the
        difference here keeps "flagged" from being mistaken for "gated"."""
        deps, _ = _deps(critic_sufficient=True)

        state = run_investigation("Did the promotion work?", deps)

        assert state["final_recommendation"] is not None
        assert state["final_recommendation"].requires_human_approval
