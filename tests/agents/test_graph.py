"""The Supervisor and the LangGraph investigation loop.

Everything runs against the stub provider: offline, free, deterministic. A
scripted plan makes "what did the graph do with this plan" an assertion rather
than an observation.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.supervisor import (
    MAX_PLAN_STEPS,
    IntentClassification,
    PlannedStep,
    ProposedPlan,
    SupervisorAgent,
)
from app.config.settings import AgentSettings
from app.guardrails.budget import BudgetTracker
from app.llm.stub import StubProvider
from app.schemas.agent_state import StepStatus, new_agent_state
from app.schemas.domain import BusinessObjective, IntentType
from app.schemas.tool_contract import ToolErrorCode, ToolResult, ToolSpec
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput
from app.tools.registry import ToolRegistry
from app.workflows.graph import InvestigationDeps, run_investigation, summarise

pytestmark = pytest.mark.agents


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class EchoInput(BaseModel):
    value: str = "x"


class EchoOutput(BaseModel):
    echoed: str


class EchoTool(AnalyticalTool[EchoInput, EchoOutput]):
    """A tool that succeeds, for testing the loop rather than a model."""

    name = "echo"
    description = "Echo the input back."
    input_schema = EchoInput
    output_schema = EchoOutput
    permission = "read_analytics"

    def _execute(self, payload: EchoInput) -> ToolOutput[EchoOutput]:
        return ToolOutput(payload=EchoOutput(echoed=payload.value))


class FailingTool(AnalyticalTool[EchoInput, EchoOutput]):
    name = "always_fails"
    description = "Always fails."
    input_schema = EchoInput
    output_schema = EchoOutput
    permission = "read_analytics"

    def _execute(self, payload: EchoInput) -> ToolOutput[EchoOutput]:
        raise ToolExecutionError("nothing to work with", recoverable=True)


def registry_with(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def scripted(
    steps: list[PlannedStep],
    *,
    intent: IntentType = IntentType.FORECAST,
) -> StubProvider:
    stub = StubProvider()
    stub.script_structured(
        IntentClassification(
            intent=intent, objective=BusinessObjective.MAXIMISE_REVENUE
        )
    )
    stub.script_structured(ProposedPlan(steps=steps))
    return stub


def deps_for(
    stub: StubProvider, registry: ToolRegistry, *, budget: BudgetTracker | None = None
) -> InvestigationDeps:
    return InvestigationDeps(
        supervisor=SupervisorAgent(provider=stub, tools=registry.specs()),
        registry=registry,
        budget=budget or BudgetTracker.from_settings(AgentSettings()),
    )


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------


class TestPlanValidation:
    def _supervisor(self, stub: StubProvider) -> SupervisorAgent:
        return SupervisorAgent(
            provider=stub,
            tools=[
                ToolSpec(name="echo", description="d", input_schema={"type": "object"})
            ],
        )

    def test_unknown_tools_are_dropped(self) -> None:
        """A model selecting from a registry occasionally invents a plausible
        name. Executing it would fail deep in the loop instead of here."""
        stub = scripted(
            [
                PlannedStep(tool_name="does_not_exist", rationale="invented"),
                PlannedStep(tool_name="echo", rationale="real"),
            ]
        )
        supervisor = self._supervisor(stub)
        plan = supervisor.plan("q", supervisor.classify("q"))

        assert [s.tool_name for s in plan.steps] == ["echo"]

    def test_plan_is_capped(self) -> None:
        """A long plan is nearly always a fan-out across every tool shown, not
        an answer to the question."""
        stub = scripted(
            [PlannedStep(tool_name="echo", rationale=f"step {i}") for i in range(12)]
        )
        supervisor = self._supervisor(stub)
        plan = supervisor.plan("q", supervisor.classify("q"))

        assert len(plan.steps) == MAX_PLAN_STEPS

    def test_empty_plan_is_allowed(self) -> None:
        """"This cannot be established" is a valid outcome, and a better one
        than running tools that will not establish it."""
        stub = scripted([])
        supervisor = self._supervisor(stub)
        plan = supervisor.plan("q", supervisor.classify("q"))

        assert plan.steps == []

    def test_revision_increments_on_replan(self) -> None:
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        supervisor = self._supervisor(stub)
        first = supervisor.plan("q", supervisor.classify("q"))
        second = supervisor.plan(
            "q", supervisor.classify("q"), previous=first, replan_reason="thin evidence"
        )

        assert first.revision == 0
        assert second.revision == 1
        assert second.replan_reason == "thin evidence"

    def test_replan_reason_reaches_the_model_verbatim(self) -> None:
        """A re-plan that does not know precisely what was insufficient repeats
        the same steps."""
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        supervisor = self._supervisor(stub)
        first = supervisor.plan("q", supervisor.classify("q"))
        supervisor.plan(
            "q",
            supervisor.classify("q"),
            previous=first,
            replan_reason="no confidence interval on the uplift",
        )

        planning_calls = [c for c in stub.calls if c["model"] == "ProposedPlan"]
        assert "no confidence interval on the uplift" in planning_calls[-1]["last_user"]

    def test_tool_descriptions_are_shown_to_the_planner(self) -> None:
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        supervisor = self._supervisor(stub)
        supervisor.plan("q", supervisor.classify("q"))

        message = [c for c in stub.calls if c["model"] == "ProposedPlan"][-1]["last_user"]
        assert "AVAILABLE TOOLS" in message
        assert "echo" in message


class TestObservation:
    def _supervisor(self) -> SupervisorAgent:
        return SupervisorAgent(provider=StubProvider(), tools=[])

    def test_error_is_summarised_as_a_failure(self) -> None:
        from app.schemas.agent_state import PlanStep

        result = ToolResult.failure(
            tool_name="echo",
            message="nothing to work with",
            code=ToolErrorCode.INSUFFICIENT_DATA,
            recoverable=True,
        )
        summary = self._supervisor().observe(
            PlanStep(tool_name="echo", rationale="r"), result
        )
        assert "failed" in summary

    def test_warnings_are_counted(self) -> None:
        from app.schemas.agent_state import PlanStep

        result = ToolResult.success(
            tool_name="echo", result={"echoed": "x"}, warnings=["thin evidence"]
        )
        summary = self._supervisor().observe(
            PlanStep(tool_name="echo", rationale="r"), result
        )
        assert "1 warning" in summary


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


class TestInvestigation:
    def test_runs_a_plan_to_completion(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r", parameters={"value": "hi"})])

        state = run_investigation("q", deps_for(stub, registry))
        summary = summarise(state)

        assert summary["complete"]
        assert summary["confidence"] == 1.0
        assert [r["tool"] for r in summary["results"]] == ["echo"]

    def test_executes_every_planned_step(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted(
            [
                PlannedStep(tool_name="echo", rationale="one"),
                PlannedStep(tool_name="echo", rationale="two"),
                PlannedStep(tool_name="echo", rationale="three"),
            ]
        )
        state = run_investigation("q", deps_for(stub, registry))

        assert state["tool_call_count"] == 3
        assert len(state["tool_results"]) == 3

    def test_minimum_sufficiency_is_preserved(self) -> None:
        """A one-step plan must produce one tool call, not a sweep of the
        registry because other tools happen to exist."""
        registry = registry_with(EchoTool(), FailingTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])

        state = run_investigation("q", deps_for(stub, registry))
        assert state["tool_call_count"] == 1

    def test_empty_plan_finishes_without_looping(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted([])

        summary = summarise(run_investigation("q", deps_for(stub, registry)))
        assert summary["plan"] == []
        assert summary["complete"]
        assert summary["confidence"] == 0.0

    def test_intent_is_recorded(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted(
            [PlannedStep(tool_name="echo", rationale="r")], intent=IntentType.ROOT_CAUSE
        )
        state = run_investigation("why did sales drop", deps_for(stub, registry))
        assert state["intent"] is IntentType.ROOT_CAUSE

    def test_observations_are_recorded_per_step(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])

        state = run_investigation("q", deps_for(stub, registry))
        assert len(state["observations"]) == 1
        assert state["observations"][0].tool_name == "echo"

    def test_trace_id_is_stable_across_the_run(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])

        state = run_investigation("q", deps_for(stub, registry))
        assert state["trace_id"]
        assert summarise(state)["trace_id"] == state["trace_id"]


class TestFailureHandling:
    def test_a_failing_tool_does_not_crash_the_run(self) -> None:
        registry = registry_with(FailingTool())
        stub = scripted([PlannedStep(tool_name="always_fails", rationale="r")])

        summary = summarise(run_investigation("q", deps_for(stub, registry)))
        assert summary["results"][0]["status"] == "error"
        assert summary["confidence"] < 1.0

    def test_step_status_reflects_the_outcome(self) -> None:
        registry = registry_with(FailingTool())
        stub = scripted([PlannedStep(tool_name="always_fails", rationale="r")])

        state = run_investigation("q", deps_for(stub, registry))
        assert state["plan"].steps[0].status is StepStatus.FAILED

    def test_confidence_is_coverage_not_judgement(self) -> None:
        """One of two steps usable gives 0.5. This measures what ran, not
        whether the answer is any good - the Critic supplies that."""
        registry = registry_with(EchoTool(), FailingTool())
        stub = scripted(
            [
                PlannedStep(tool_name="echo", rationale="works"),
                PlannedStep(tool_name="always_fails", rationale="does not"),
            ]
        )
        state = run_investigation("q", deps_for(stub, registry))
        assert state["confidence"] == pytest.approx(0.5)


class TestBudget:
    def test_exhausted_tool_budget_truncates_and_says_so(self) -> None:
        """A truthful partial answer beats an infinite loop, and beats a
        confident answer built on an investigation that never finished."""
        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        budget = BudgetTracker(
            max_iterations=10,
            max_tool_calls=0,
            max_execution_seconds=60,
            max_token_budget=100_000,
        )

        summary = summarise(run_investigation("q", deps_for(stub, registry, budget=budget)))

        assert not summary["complete"]
        assert any("budget exhausted" in e for e in summary["errors"])
        assert summary["tool_calls"] == 0

    def test_budget_stops_partway_through_a_plan(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted(
            [PlannedStep(tool_name="echo", rationale=f"s{i}") for i in range(4)]
        )
        budget = BudgetTracker(
            max_iterations=10,
            max_tool_calls=2,
            max_execution_seconds=60,
            max_token_budget=100_000,
        )

        state = run_investigation("q", deps_for(stub, registry, budget=budget))
        assert state["tool_call_count"] == 2
        assert state["errors"]

    def test_iteration_budget_blocks_planning(self) -> None:
        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        budget = BudgetTracker(
            max_iterations=0,
            max_tool_calls=10,
            max_execution_seconds=60,
            max_token_budget=100_000,
        )

        summary = summarise(run_investigation("q", deps_for(stub, registry, budget=budget)))
        assert any("budget exhausted" in e for e in summary["errors"])


class TestStateShape:
    def test_new_state_initialises_every_counter(self) -> None:
        """Centralised so a node can never observe a missing counter and
        default it to something inconsistent."""
        state = new_agent_state("q")
        for key in ("iteration", "tool_call_count", "tokens_used", "replan_count"):
            assert state[key] == 0
        for key in ("tool_results", "observations", "errors", "completed_steps"):
            assert state[key] == []

    def test_entities_survive_into_planning(self) -> None:
        registry = registry_with(EchoTool())
        stub = StubProvider()
        stub.script_structured(
            IntentClassification(
                intent=IntentType.FORECAST,
                objective=BusinessObjective.MAXIMISE_REVENUE,
                entities={"product": ["P00003"]},
            )
        )
        stub.script_structured(ProposedPlan(steps=[PlannedStep(tool_name="echo", rationale="r")]))

        state = run_investigation("q", deps_for(stub, registry))
        assert state["entities"] == {"product": ["P00003"]}

    def test_summarise_is_json_safe(self) -> None:
        import json

        registry = registry_with(EchoTool())
        stub = scripted([PlannedStep(tool_name="echo", rationale="r")])
        summary = summarise(run_investigation("q", deps_for(stub, registry)))

        assert json.dumps(summary)
