"""Running an investigation and recording what it did.

The seam between the agent graph and everything that consumes it. The API, the
CLI and the UI all call this; none of them knows about `AgentState`, and the
graph knows nothing about HTTP or SQL.

**The trace is reconstructed from the finished state, not emitted during the
run.** Instrumenting every node with a callback would couple the graph to a
sink, and the sink would then be the thing that must not fail - a trace writer
throwing mid-investigation would lose the investigation. Reading the state
afterwards produces the same events with none of that coupling. What it gives up
is per-node wall-clock timing, which nothing currently asks for.

**A failed investigation is still recorded.** The row is written before the
agent runs and updated after, so a crash leaves a `failed` row with the error on
it rather than no evidence that anything was attempted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.recommendation import RecommendationAgent
from app.agents.supervisor import SupervisorAgent
from app.config.settings import AgentSettings
from app.guardrails.budget import BudgetTracker
from app.llm.base import LLMProvider
from app.observability.logging import get_logger
from app.schemas.agent_state import AgentState, Recommendation
from app.schemas.api import (
    InvestigationResponse,
    InvestigationStatus,
    TraceEvent,
    TraceResponse,
)
from app.store.investigations import InvestigationStore
from app.tools.registry import ToolRegistry
from app.workflows.graph import InvestigationDeps, run_investigation

logger = get_logger(__name__)


@dataclass
class InvestigationOutcome:
    """What a caller needs, without reaching into agent state."""

    investigation_id: str
    trace_id: str
    status: InvestigationStatus
    question: str
    recommendation: Recommendation | None
    answer: str
    intent: str | None
    objective: str | None
    events: list[TraceEvent]
    error: str | None = None


@dataclass
class InvestigationService:
    """Runs investigations and persists them."""

    provider: LLMProvider
    registry: ToolRegistry
    store: InvestigationStore
    settings: AgentSettings

    def run(
        self,
        question: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> InvestigationOutcome:
        """Run one investigation to completion and record it."""
        investigation_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        prompt = _with_scope(question, scope)

        self.store.create(
            investigation_id=investigation_id,
            trace_id=trace_id,
            question=question,
            user_id=user_id,
            session_id=session_id,
        )

        try:
            state = run_investigation(
                prompt,
                self._deps(),
                user_id=user_id,
                investigation_id=investigation_id,
                trace_id=trace_id,
            )
        except Exception as exc:
            logger.exception(
                "investigation.failed", investigation_id=investigation_id
            )
            self.store.complete(
                investigation_id,
                status=InvestigationStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            return InvestigationOutcome(
                investigation_id=investigation_id,
                trace_id=trace_id,
                status=InvestigationStatus.FAILED,
                question=question,
                recommendation=None,
                answer=(
                    "The investigation could not be completed. No conclusion is "
                    "available, and none should be inferred from the absence of one."
                ),
                intent=None,
                objective=None,
                events=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        events = build_trace(state)
        recommendation = state.get("final_recommendation")
        status = _status_for(state)

        self.store.append_events(investigation_id, events)
        self.store.complete(
            investigation_id,
            status=status,
            intent=str(state.get("intent")) if state.get("intent") else None,
            objective=str(state.get("objective")) if state.get("objective") else None,
            recommendation=recommendation.model_dump(mode="json")
            if recommendation
            else None,
            tool_calls=int(state.get("tool_call_count", 0)),
            replans=int(state.get("replan_count", 0)),
        )

        return InvestigationOutcome(
            investigation_id=investigation_id,
            trace_id=trace_id,
            status=status,
            question=question,
            recommendation=recommendation,
            answer=_answer_text(state),
            intent=str(state.get("intent")) if state.get("intent") else None,
            objective=str(state.get("objective")) if state.get("objective") else None,
            events=events,
        )

    def get(self, investigation_id: str) -> InvestigationResponse | None:
        return self.store.get(investigation_id)

    def get_trace(self, investigation_id: str) -> TraceResponse | None:
        return self.store.get_trace(investigation_id)

    def recent(self, limit: int = 20) -> list[InvestigationResponse]:
        return self.store.recent(limit)

    def _deps(self) -> InvestigationDeps:
        return InvestigationDeps(
            supervisor=SupervisorAgent(
                provider=self.provider, tools=self.registry.specs()
            ),
            registry=self.registry,
            budget=BudgetTracker.from_settings(self.settings),
            critic=CriticAgent(provider=self.provider),
            recommender=RecommendationAgent(
                provider=self.provider,
                approval_threshold=self.settings.human_approval_threshold,
            ),
            max_replans=self.settings.max_replans,
        )


def _status_for(state: AgentState) -> InvestigationStatus:
    """Failed, waiting for a person, or genuinely done.

    **An investigation that gathered no usable evidence is `failed`**, even
    though the graph ran to the end without raising. `completed` is a claim that
    the question was answered, and a recommendation built on nothing does not
    answer it - reporting that as complete is how an empty result gets mistaken
    for a finding of no effect.

    `awaiting_approval` is likewise a distinct status rather than a flag on a
    completed one: an investigation whose recommendation crosses the approval
    threshold has not finished from the business's point of view, however
    finished the graph is.
    """
    usable = [
        result
        for result in state.get("tool_results", [])
        if result.status != "error" and result.result
    ]
    if not usable:
        return InvestigationStatus.FAILED

    recommendation = state.get("final_recommendation")
    if recommendation is not None and recommendation.requires_human_approval:
        return InvestigationStatus.AWAITING_APPROVAL
    return InvestigationStatus.COMPLETED


def _with_scope(question: str, scope: dict[str, Any] | None) -> str:
    """Append explicit scope filters to the question.

    `POST /investigate` lets a caller name products, stores and a window rather
    than hoping the model extracts them from prose. Passed as text because that
    is what the Supervisor reads - the alternative, threading a scope object
    through every node, would touch the whole graph to serve one endpoint.
    """
    if not scope:
        return question

    parts = [f"{key}: {value}" for key, value in scope.items() if value]
    if not parts:
        return question
    return f"{question}\n\nSCOPE: " + "; ".join(parts)


def _answer_text(state: AgentState) -> str:
    """The prose a chat caller sees.

    Falls back to a statement of what was not established rather than to an
    empty string. A blank answer in a chat window reads as a system fault; a
    sentence saying no conclusion was reached is the actual outcome.
    """
    recommendation = state.get("final_recommendation")
    # An empty summary counts as no recommendation. Returning the empty string
    # would render as a blank chat message, which reads as a system fault rather
    # than as the "nothing was established" that it is.
    if recommendation is not None and recommendation.executive_summary.strip():
        parts = [recommendation.executive_summary]
        if recommendation.root_cause:
            parts.append(recommendation.root_cause)
        parts.append(recommendation.recommended_action)
        return "\n\n".join(p for p in parts if p.strip())

    verdict = state.get("critic_verdict")
    if verdict is not None and not verdict.valid:
        missing = "; ".join(verdict.required_followup or verdict.issues)
        return (
            "The evidence gathered does not support a conclusion. "
            f"Outstanding: {missing}"
            if missing
            else "The evidence gathered does not support a conclusion."
        )
    return "No conclusion was reached."


# --------------------------------------------------------------------------
# Trace construction
# --------------------------------------------------------------------------


def build_trace(state: AgentState) -> list[TraceEvent]:
    """Rebuild the user-facing trace from a finished investigation.

    Ordered as the investigation ran: understand, plan, act, observe, judge,
    conclude. Every event carries a summary a person can read; none carries
    private chain-of-thought.
    """
    events: list[TraceEvent] = []
    now = datetime.now(UTC)

    def add(
        event_type: str,
        actor: str,
        summary: str,
        *,
        tool_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        events.append(
            TraceEvent(
                sequence=len(events) + 1,
                timestamp=now,
                event_type=event_type,
                actor=actor,
                summary=summary,
                tool_name=tool_name,
                payload=payload or {},
            )
        )

    intent = state.get("intent")
    if intent:
        entities = state.get("entities") or {}
        scoped = ", ".join(f"{k}={v}" for k, v in entities.items() if v)
        add(
            "intent_classified",
            "supervisor",
            f"Read as a {intent} question" + (f", scoped to {scoped}" if scoped else ""),
            payload={"intent": str(intent), "entities": entities},
        )

    plan = state.get("plan")
    if plan:
        add(
            "plan_created",
            "supervisor",
            f"Planned {len(plan.steps)} step(s): "
            + ", ".join(s.tool_name for s in plan.steps),
            payload={
                "revision": plan.revision,
                "steps": [
                    {"tool": s.tool_name, "rationale": s.rationale} for s in plan.steps
                ],
            },
        )

    observations = {o.step_id: o for o in state.get("observations", [])}
    for result in state.get("tool_results", []):
        add(
            "tool_called" if result.status != "error" else "tool_failed",
            "executor",
            _tool_summary(result),
            tool_name=result.tool_name,
            payload={
                "status": str(result.status),
                "warnings": list(result.warnings),
                "assumptions": list(result.assumptions),
                # The numbers themselves, so the UI can show what a claim rests
                # on without a second round trip.
                "result": result.result or {},
            },
        )

    for observation in observations.values():
        add(
            "observation",
            "supervisor",
            observation.summary,
            tool_name=observation.tool_name,
            payload={"informative": observation.informative},
        )

    replans = int(state.get("replan_count", 0))
    if replans:
        add(
            "replanned",
            "supervisor",
            f"Re-planned {replans} time(s) after the Critic found the evidence "
            f"insufficient",
            payload={"replan_count": replans},
        )

    verdict = state.get("critic_verdict")
    if verdict is not None:
        add(
            "critic_verdict",
            "critic",
            ("Evidence supports a conclusion" if verdict.valid else "Evidence is insufficient")
            + (f": {'; '.join(verdict.issues)}" if verdict.issues else ""),
            payload={
                "valid": verdict.valid,
                "confidence": verdict.confidence,
                "issues": list(verdict.issues),
                "required_followup": list(verdict.required_followup),
            },
        )

    recommendation = state.get("final_recommendation")
    if recommendation is not None:
        add(
            "recommendation",
            "recommender",
            recommendation.recommended_action,
            payload={
                "confidence": recommendation.confidence,
                "requires_human_approval": recommendation.requires_human_approval,
                "risks": list(recommendation.risks),
                "evidence": [
                    {"claim": e.claim, "source_tool": e.source_tool}
                    for e in recommendation.evidence
                ],
            },
        )

    for error in state.get("errors", []):
        add("error", "system", error)

    return events


def _tool_summary(result: Any) -> str:
    if result.status == "error":
        detail = result.error.message if result.error else "unknown error"
        return f"{result.tool_name} could not run: {detail}"
    warnings = f" ({len(result.warnings)} warning(s))" if result.warnings else ""
    return f"{result.tool_name} returned a result{warnings}"


__all__ = [
    "InvestigationOutcome",
    "InvestigationService",
    "build_trace",
]
