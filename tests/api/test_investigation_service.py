"""The service seam and the trace it reconstructs.

Tested apart from HTTP because the guarantees here are not about status codes:
that a failed investigation still leaves a record, that the trace shows every
stage, and that no private reasoning reaches it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.agents.critic import CriticAssessment
from app.agents.recommendation import DraftRecommendation
from app.agents.supervisor import IntentClassification, PlannedStep, ProposedPlan
from app.config.settings import AgentSettings
from app.llm.stub import StubProvider
from app.schemas.agent_state import CriticVerdict, new_agent_state
from app.schemas.api import InvestigationStatus
from app.schemas.domain import BusinessObjective, IntentType
from app.schemas.tool_contract import ToolResult
from app.services.investigation_service import InvestigationService, build_trace
from app.store.investigations import InvestigationStore
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.agents


class In(BaseModel):
    product_id: str | None = None


class Out(BaseModel):
    incremental_units: float = 4200.0


class UpliftTool(AnalyticalTool[In, Out]):
    name = "estimate_promo_uplift"
    description = "d"
    input_schema = In
    output_schema = Out
    permission = "run_model"

    def _execute(self, payload: In) -> ToolOutput[Out]:
        return ToolOutput(payload=Out())


class ExplodingTool(AnalyticalTool[In, Out]):
    name = "estimate_promo_uplift"
    description = "d"
    input_schema = In
    output_schema = Out
    permission = "run_model"

    def _execute(self, payload: In) -> ToolOutput[Out]:
        raise ToolExecutionError("no data for that window", recoverable=False)


def scripted_stub() -> StubProvider:
    stub = StubProvider()
    stub.script_structured(
        IntentClassification(
            intent=IntentType.PROMOTION_DECISION,
            objective=BusinessObjective.MAXIMISE_PROFIT,
            entities={"products": ["P00091"]},
        )
    )
    stub.script_structured(
        ProposedPlan(
            steps=[PlannedStep(tool_name="estimate_promo_uplift", rationale="measure")]
        )
    )
    stub.script_structured(CriticAssessment(sufficient=True, confidence=0.8))
    stub.script_structured(
        DraftRecommendation(
            executive_summary="Volume rose measurably.",
            recommended_action="Continue funding it.",
            confidence=0.8,
            estimated_profit_impact=50_000.0,
        )
    )
    return stub


def service(tmp_path, tool: object | None = None) -> InvestigationService:
    registry = ToolRegistry()
    registry.register(tool or UpliftTool())
    return InvestigationService(
        provider=scripted_stub(),
        registry=registry,
        store=InvestigationStore(f"sqlite:///{tmp_path / 'svc.sqlite'}"),
        settings=AgentSettings(human_approval_threshold=1_000_000.0),
    )


class TestService:
    def test_a_completed_investigation_is_persisted(self, tmp_path) -> None:
        svc = service(tmp_path)
        outcome = svc.run("Did the promotion work?")

        stored = svc.get(outcome.investigation_id)
        assert stored is not None
        assert stored.status is InvestigationStatus.COMPLETED
        assert stored.recommendation is not None

    def test_the_scope_reaches_the_agent(self, tmp_path) -> None:
        """A caller who already knows which products they mean should not have
        their filter depend on the model reading it out of prose."""
        svc = service(tmp_path)
        svc.run("Was it worth it?", scope={"products": ["P00091"], "region": "North"})

        asked = svc.provider.calls[0]["last_user"]  # type: ignore[attr-defined]
        assert "P00091" in asked
        assert "North" in asked

    def test_an_empty_scope_leaves_the_question_alone(self, tmp_path) -> None:
        svc = service(tmp_path)
        svc.run("Plain question?", scope={"region": None, "products": []})

        assert "SCOPE" not in svc.provider.calls[0]["last_user"]  # type: ignore[attr-defined]

    def test_a_tool_failure_still_produces_a_recorded_investigation(
        self, tmp_path
    ) -> None:
        """The tool contract absorbs its own errors, so this completes with the
        failure captured as evidence rather than crashing the run."""
        svc = service(tmp_path, ExplodingTool())
        outcome = svc.run("Did the promotion work?")

        stored = svc.get(outcome.investigation_id)
        assert stored is not None
        trace = svc.get_trace(outcome.investigation_id)
        assert trace is not None
        assert any(e.event_type == "tool_failed" for e in trace.events)

    def test_an_investigation_id_is_shared_with_the_graph(self, tmp_path) -> None:
        """The stored row and the graph's own state must agree, or the two
        records cannot be joined afterwards."""
        svc = service(tmp_path)
        outcome = svc.run("Did the promotion work?")
        trace = svc.get_trace(outcome.investigation_id)

        assert trace is not None
        assert trace.trace_id == outcome.trace_id

    def test_gathering_no_evidence_is_failed_not_completed(self, tmp_path) -> None:
        """`completed` claims the question was answered. An unscripted provider
        plans nothing, so nothing runs - and reporting that as complete is how
        an empty result gets mistaken for a finding of no effect."""
        registry = ToolRegistry()
        registry.register(UpliftTool())
        svc = InvestigationService(
            provider=StubProvider(),  # nothing scripted: the plan comes back empty
            registry=registry,
            store=InvestigationStore(f"sqlite:///{tmp_path / 'empty.sqlite'}"),
            settings=AgentSettings(),
        )
        outcome = svc.run("Did the promotion work?")

        assert outcome.status is InvestigationStatus.FAILED

    def test_an_empty_summary_reads_as_no_conclusion(self, tmp_path) -> None:
        """A blank answer in a chat window reads as a system fault. The actual
        outcome is that nothing was established, and it should say so."""
        registry = ToolRegistry()
        registry.register(UpliftTool())
        svc = InvestigationService(
            provider=StubProvider(),
            registry=registry,
            store=InvestigationStore(f"sqlite:///{tmp_path / 'blank.sqlite'}"),
            settings=AgentSettings(),
        )
        outcome = svc.run("Did the promotion work?")

        assert outcome.answer.strip()
        assert "does not support a conclusion" in outcome.answer

    def test_recent_lists_newest_first(self, tmp_path) -> None:
        svc = service(tmp_path)
        first = svc.run("First question?")
        second = svc.run("Second question?")

        listed = [i.investigation_id for i in svc.recent()]
        assert listed.index(second.investigation_id) < listed.index(
            first.investigation_id
        )


class TestTraceConstruction:
    def _state(self, **overrides):
        state = new_agent_state("Did it work?")
        state["intent"] = IntentType.PROMOTION_DECISION
        state["entities"] = {"products": ["P00091"]}
        state["tool_results"] = [
            ToolResult.success(
                tool_name="estimate_promo_uplift", result={"incremental_units": 4200.0}
            )
        ]
        state.update(overrides)
        return state

    def test_a_replan_is_visible(self) -> None:
        """An investigation that had to try twice looks different from one that
        got it right first time, and the trace should say so."""
        events = build_trace(self._state(replan_count=2))

        assert any(e.event_type == "replanned" for e in events)

    def test_no_replan_produces_no_replan_event(self) -> None:
        events = build_trace(self._state(replan_count=0))

        assert not any(e.event_type == "replanned" for e in events)

    def test_the_critic_verdict_carries_its_issues(self) -> None:
        events = build_trace(
            self._state(
                critic_verdict=CriticVerdict(
                    valid=False, confidence=0.3, issues=["no counterfactual"]
                )
            )
        )
        verdict = next(e for e in events if e.event_type == "critic_verdict")

        assert "no counterfactual" in verdict.summary
        assert verdict.payload["valid"] is False

    def test_tool_results_travel_with_the_event(self) -> None:
        """So the UI can show what a claim rests on without a second round
        trip."""
        events = build_trace(self._state())
        call = next(e for e in events if e.event_type == "tool_called")

        assert call.payload["result"]["incremental_units"] == 4200.0

    def test_events_are_sequenced_from_one(self) -> None:
        events = build_trace(self._state())

        assert [e.sequence for e in events] == list(range(1, len(events) + 1))

    def test_an_empty_state_produces_an_empty_trace(self) -> None:
        assert build_trace(new_agent_state("q")) == []
