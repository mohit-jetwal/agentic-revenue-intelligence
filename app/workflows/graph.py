"""The LangGraph investigation loop.

Assembles the nodes into the plan / act / observe cycle::

    classify_intent -> plan -> execute_step -> observe -> evaluate
                        ^                                    |
                        +--------- more steps ---------------+
                                                             |
                                                          finish

Step 12 adds the Critic, re-planning and the human-approval interrupt on the
edge out of ``evaluate``. The shape is built to accept them: ``evaluate`` is
already a conditional router rather than a straight edge, so adding a
``replan`` destination is an edge change rather than a restructure.

**Three properties this layer owns.**

*Bounded loops.* Every path that can repeat passes through the
:class:`~app.guardrails.budget.BudgetTracker`. An agent that can re-plan is an
agent that can loop forever, and a Critic that keeps returning "insufficient
evidence" is the realistic way it happens.

*Tool execution stays here.* The provider returns requested calls and never
runs them, so permission checks and budget accounting happen in one auditable
place.

*A truncated investigation says so.* Running out of budget produces a partial
answer explicitly flagged as incomplete, never a confident one built on an
investigation that did not finish.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.supervisor import SupervisorAgent
from app.guardrails.budget import BudgetExceededError, BudgetTracker
from app.observability.context import trace_context
from app.observability.logging import get_logger
from app.schemas.agent_state import (
    AgentState,
    Observation,
    StepStatus,
    new_agent_state,
)
from app.schemas.tool_contract import ToolResult
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


@dataclass
class InvestigationDeps:
    """Everything the graph nodes need, injected rather than imported.

    Passed as a closure over the node functions so the graph can be built
    against a stub provider and a fake registry in tests without touching
    module-level state.
    """

    supervisor: SupervisorAgent
    registry: ToolRegistry
    budget: BudgetTracker


def build_graph(deps: InvestigationDeps) -> Any:
    """Compile the investigation graph."""
    graph = StateGraph(AgentState)

    # `type: ignore[call-overload]` on each: LangGraph's `add_node` overloads
    # are written against an unparameterised node protocol and do not accept a
    # precisely-typed `Callable[[AgentState], AgentState]`. Loosening the node
    # signatures to `Any` would satisfy the stub by giving up the type checking
    # that matters - inside the nodes, where the state keys are.
    graph.add_node("classify_intent", _classify_node(deps))  # type: ignore[call-overload]
    graph.add_node("plan", _plan_node(deps))  # type: ignore[call-overload]
    graph.add_node("execute_step", _execute_node(deps))  # type: ignore[call-overload]
    graph.add_node("observe", _observe_node(deps))  # type: ignore[call-overload]
    graph.add_node("finish", _finish_node(deps))  # type: ignore[call-overload]

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan")
    graph.add_conditional_edges(
        "plan", _after_plan, {"execute_step": "execute_step", "finish": "finish"}
    )
    graph.add_edge("execute_step", "observe")
    graph.add_conditional_edges(
        "observe", _evaluate, {"execute_step": "execute_step", "finish": "finish"}
    )
    graph.add_edge("finish", END)

    return graph.compile()


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def _classify_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def classify_intent(state: AgentState) -> AgentState:
        classification = deps.supervisor.classify(state["user_question"])
        deps.budget.record_tokens(deps.supervisor.provider.count_tokens(state["user_question"]))
        return {
            "intent": classification.intent,
            "objective": classification.objective,
            "entities": classification.entities,
        }

    return classify_intent


def _plan_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def plan(state: AgentState) -> AgentState:
        from app.agents.supervisor import IntentClassification

        try:
            deps.budget.record_iteration()
        except BudgetExceededError as exc:
            # Not an error to hide. The investigation stops and says why.
            return {"errors": [f"budget exhausted before planning: {exc}"]}

        classification = IntentClassification(
            intent=state["intent"],  # type: ignore[arg-type]
            objective=state["objective"],  # type: ignore[arg-type]
            entities=state.get("entities", {}),
        )
        new_plan = deps.supervisor.plan(
            state["user_question"],
            classification,
            previous=state.get("plan"),
            replan_reason=state.get("replan_reason"),
            results=state.get("tool_results", []),
        )
        return {
            "plan": new_plan,
            "iteration": state.get("iteration", 0) + 1,
            "replan_count": new_plan.revision,
        }

    return plan


def _execute_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def execute_step(state: AgentState) -> AgentState:
        """Run the next pending step.

        One step per visit rather than the whole plan, so the budget is checked
        between calls and a later step can be skipped once the budget is gone.
        """
        plan = state.get("plan")
        if plan is None:
            return {"errors": ["execute_step reached with no plan"]}

        pending = plan.pending_steps()
        if not pending:
            return {}

        step = pending[0]

        if deps.budget.would_exceed(tool_calls=1):
            step.status = StepStatus.SKIPPED
            return {
                "errors": [
                    f"budget exhausted before {step.tool_name}; the investigation "
                    f"is incomplete and its conclusion must say so"
                ]
            }

        if not deps.registry.has(step.tool_name):
            # The Supervisor drops unknown tools when validating a plan, so
            # reaching here means the registry changed mid-investigation. Fail
            # the step rather than the run.
            step.status = StepStatus.FAILED
            return {"errors": [f"tool {step.tool_name!r} is not registered"]}

        tool = deps.registry.get(step.tool_name)
        step.status = StepStatus.RUNNING
        result = tool.run(step.parameters)
        deps.budget.record_tool_call()
        step.status = (
            StepStatus.COMPLETED if result.status != "error" else StepStatus.FAILED
        )

        logger.info(
            "graph.step_executed",
            tool=step.tool_name,
            status=result.status,
            step_id=step.step_id,
        )
        return {
            "tool_results": [result],
            "completed_steps": [step.step_id],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        }

    return execute_step


def _observe_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def observe(state: AgentState) -> AgentState:
        results = state.get("tool_results", [])
        plan = state.get("plan")
        if not results or plan is None:
            return {}

        latest = results[-1]
        step = next(
            (s for s in plan.steps if s.tool_name == latest.tool_name),
            None,
        )
        if step is None:
            return {}

        observation = Observation(
            step_id=step.step_id,
            tool_name=step.tool_name,
            summary=deps.supervisor.observe(step, latest),
            informative=latest.status != "error",
        )
        return {"observations": [observation]}

    return observe


def _finish_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def finish(state: AgentState) -> AgentState:
        """Close the investigation and record how confident it may be.

        Confidence here is a *coverage* measure - what share of planned steps
        produced a usable result - not a judgement about the answer. The Critic
        in Step 12 supplies the judgement; conflating the two would let a
        fully-executed plan of weak evidence report high confidence.
        """
        results = state.get("tool_results", [])
        plan = state.get("plan")
        planned = len(plan.steps) if plan else 0
        usable = sum(1 for r in results if r.status != "error")

        confidence = usable / planned if planned else 0.0
        if state.get("errors"):
            confidence = min(confidence, 0.5)

        logger.info(
            "graph.finished",
            steps_planned=planned,
            results=len(results),
            usable=usable,
            confidence=round(confidence, 3),
        )
        _ = deps
        return {"confidence": confidence}

    return finish


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def _after_plan(state: AgentState) -> str:
    """An empty plan is a valid outcome, not a failure.

    The Supervisor is instructed to return no steps when a question cannot be
    answered with the available tools. Routing that to ``finish`` lets the
    investigation say so, rather than looping trying to find something to run.
    """
    plan = state.get("plan")
    if plan is None or not plan.steps:
        return "finish"
    return "execute_step"


def _evaluate(state: AgentState) -> str:
    """Continue through the plan, or stop.

    Step 12 adds a ``replan`` destination here when the Critic judges the
    evidence insufficient. Today the only reasons to stop are: the plan is done,
    or the budget is gone.
    """
    plan = state.get("plan")
    if plan is None or not plan.pending_steps():
        return "finish"
    if state.get("errors"):
        # A budget or execution failure already recorded. Stop rather than
        # burning the remainder on steps that will hit the same wall.
        return "finish"
    return "execute_step"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_investigation(
    question: str,
    deps: InvestigationDeps,
    *,
    user_id: str | None = None,
    investigation_id: str | None = None,
) -> AgentState:
    """Run one investigation to completion.

    Wrapped in a trace context so every log line from every node - including
    inside tool execution - correlates to one investigation id.
    """
    state = new_agent_state(question, user_id=user_id, investigation_id=investigation_id)
    graph = build_graph(deps)

    with trace_context(
        trace_id=state["trace_id"],
        investigation_id=state["investigation_id"],
        user_id=user_id,
    ):
        logger.info("graph.started", question_length=len(question))
        final: AgentState = graph.invoke(state)

    return final


def summarise(state: AgentState) -> dict[str, Any]:
    """A compact view of what an investigation did, for the API and the UI."""
    plan = state.get("plan")
    results: list[ToolResult] = state.get("tool_results", [])

    return {
        "investigation_id": state.get("investigation_id"),
        "trace_id": state.get("trace_id"),
        "question": state.get("user_question"),
        "intent": str(state.get("intent")) if state.get("intent") else None,
        "objective": str(state.get("objective")) if state.get("objective") else None,
        "plan": [
            {
                "tool": step.tool_name,
                "rationale": step.rationale,
                "status": str(step.status),
            }
            for step in (plan.steps if plan else [])
        ],
        "revision": plan.revision if plan else 0,
        "results": [
            {
                "tool": result.tool_name,
                "status": str(result.status),
                "warnings": len(result.warnings),
            }
            for result in results
        ],
        "observations": [o.summary for o in state.get("observations", [])],
        "errors": state.get("errors", []),
        "confidence": state.get("confidence", 0.0),
        "tool_calls": state.get("tool_call_count", 0),
        "complete": not state.get("errors"),
    }


__all__ = [
    "InvestigationDeps",
    "build_graph",
    "run_investigation",
    "summarise",
]
