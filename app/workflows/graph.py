"""The LangGraph investigation loop.

Assembles the nodes into the plan / act / observe / critique cycle::

    classify_intent -> plan -> execute_step -> observe
                        ^                        |
                        |                   more steps
                        |                        |
                        |                     critic
                        |                    /      \\
                        +--- insufficient ---        -> recommend -> finish
                          (bounded by max_replans)        ^
                                                          |
                                                 human approval interrupt
                                                 (above the impact threshold)

**Four properties this layer owns.**

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

*Re-planning is bounded twice.* By ``max_replans`` and by the budget. A Critic
that is never satisfied is the realistic way an agent loops forever, and the
loop back to ``plan`` is the only cycle in the graph.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.critic import CriticAgent
from app.agents.recommendation import RecommendationAgent
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
    #: Optional. Without them the graph runs plan/act/observe and stops, which
    #: is a legitimate configuration for a tool-only investigation.
    critic: CriticAgent | None = None
    recommender: RecommendationAgent | None = None
    max_replans: int = 2


def build_graph(deps: InvestigationDeps, *, checkpointer: Any | None = None) -> Any:
    """Compile the investigation graph.

    ``checkpointer`` is what makes the human-approval interrupt real: the graph
    has to persist its state to stop, wait for a person, and resume. Without one
    the approval flag is still set on the recommendation but execution runs
    straight through - the difference between "flagged for approval" and
    "blocked pending approval".
    """
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
    graph.add_node("critic", _critic_node(deps))  # type: ignore[call-overload]
    graph.add_node("recommend", _recommend_node(deps))  # type: ignore[call-overload]
    graph.add_node("finish", _finish_node(deps))  # type: ignore[call-overload]

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan")
    graph.add_conditional_edges(
        "plan", _after_plan, {"execute_step": "execute_step", "critic": "critic"}
    )
    graph.add_edge("execute_step", "observe")
    graph.add_conditional_edges(
        "observe", _evaluate, {"execute_step": "execute_step", "critic": "critic"}
    )
    # The only cycle in the graph, bounded twice: by max_replans here and by the
    # budget check inside `plan`.
    graph.add_conditional_edges(
        "critic", _after_critic, {"plan": "plan", "recommend": "recommend"}
    )
    graph.add_edge("recommend", "finish")
    graph.add_edge("finish", END)

    if checkpointer is not None:
        # Interrupt *before* recommend, not after: the point is to review the
        # evidence before a recommendation is written, not to rubber-stamp one
        # already produced.
        return graph.compile(checkpointer=checkpointer, interrupt_before=["recommend"])
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


def _critic_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def critic(state: AgentState) -> AgentState:
        """Judge whether the evidence supports a conclusion.

        With no Critic configured the investigation proceeds - a tool-only run
        is a legitimate configuration, and silently inventing a passing verdict
        would be worse than having none.
        """
        if deps.critic is None:
            return {}

        verdict = deps.critic.review(
            state["user_question"], state.get("tool_results", [])
        )
        return {
            "critic_verdict": verdict,
            "replan_reason": (
                "; ".join(verdict.required_followup) or "; ".join(verdict.issues)
                if not verdict.valid
                else None
            ),
        }

    return critic


def _recommend_node(deps: InvestigationDeps) -> Callable[[AgentState], AgentState]:
    def recommend(state: AgentState) -> AgentState:
        if deps.recommender is None:
            return {}

        verdict = state.get("critic_verdict")
        recommendation = deps.recommender.synthesise(
            state["user_question"],
            state.get("tool_results", []),
            critic_issues=list(verdict.issues) if verdict else None,
            # False only when the re-plan budget ran out with the objection
            # still standing; the recommendation caps its confidence to match.
            critic_satisfied=verdict.valid if verdict else True,
            truncated=bool(state.get("errors")),
        )
        return {"final_recommendation": recommendation}

    return recommend


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
    answered with the available tools. Routing to ``critic`` rather than
    straight to ``finish`` means the investigation still produces a verdict and
    a recommendation saying so, instead of returning nothing.
    """
    plan = state.get("plan")
    if plan is None or not plan.steps:
        return "critic"
    return "execute_step"


def _evaluate(state: AgentState) -> str:
    """Continue through the plan, or hand it to the Critic."""
    plan = state.get("plan")
    if plan is None or not plan.pending_steps():
        return "critic"
    if state.get("errors"):
        # A budget or execution failure already recorded. Stop executing rather
        # than burning the remainder on steps that will hit the same wall - but
        # still critique what was gathered.
        return "critic"
    return "execute_step"


def _after_critic(state: AgentState) -> str:
    """Re-plan on an insufficient verdict, bounded twice.

    ``max_replans`` caps how many times the Critic may send an investigation
    back, and the budget check inside ``plan`` catches the rest. A Critic that
    is never satisfied is the realistic way an agent loops forever, and this is
    the only cycle in the graph.

    Hitting the cap is **not** silent. The verdict's issues travel into the
    recommendation, so a conclusion reached after exhausting the re-plan budget
    says what remained unresolved.
    """
    verdict = state.get("critic_verdict")
    if verdict is None or verdict.valid:
        return "recommend"

    if state.get("replan_count", 0) >= _max_replans(state):
        logger.info(
            "graph.replan_limit_reached",
            replans=state.get("replan_count", 0),
            unresolved=len(verdict.required_followup),
        )
        return "recommend"

    if state.get("errors"):
        # The budget is already gone. Re-planning would produce a plan that
        # cannot execute.
        return "recommend"

    logger.info("graph.replanning", reason=state.get("replan_reason"))
    return "plan"


def _max_replans(state: AgentState) -> int:
    """The cap, carried in state so the router stays a pure function."""
    return int(state.get("max_replans", 2))


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
    state = new_agent_state(
        question,
        user_id=user_id,
        investigation_id=investigation_id,
        max_replans=deps.max_replans,
    )
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
        "replans": state.get("replan_count", 0),
        "complete": not state.get("errors"),
        "critic": _critic_summary(state),
        "recommendation": _recommendation_summary(state),
    }


def _critic_summary(state: AgentState) -> dict[str, Any] | None:
    verdict = state.get("critic_verdict")
    if verdict is None:
        return None
    return {
        "valid": verdict.valid,
        "confidence": verdict.confidence,
        "issues": list(verdict.issues),
        "required_followup": list(verdict.required_followup),
    }


def _recommendation_summary(state: AgentState) -> dict[str, Any] | None:
    recommendation = state.get("final_recommendation")
    if recommendation is None:
        return None
    return {
        "executive_summary": recommendation.executive_summary,
        "root_cause": recommendation.root_cause,
        "recommended_action": recommendation.recommended_action,
        "confidence": recommendation.confidence,
        # The flag a caller must check before acting. See the approval threshold
        # in AgentSettings.
        "requires_human_approval": recommendation.requires_human_approval,
        "evidence": [
            {"claim": e.claim, "source": e.source_tool, "trace_id": e.source_trace_id}
            for e in recommendation.evidence
        ],
        "assumptions": list(recommendation.assumptions),
        "risks": list(recommendation.risks),
    }


__all__ = [
    "InvestigationDeps",
    "build_graph",
    "run_investigation",
    "summarise",
]
