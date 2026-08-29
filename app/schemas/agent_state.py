"""LangGraph agent state and planning primitives.

Implements section 17 of the brief. ``AgentState`` is a ``TypedDict`` because
that is what LangGraph expects for a graph state schema; the richer structures
it holds (plan steps, hypotheses, evidence) are Pydantic models so they stay
validated.

Design note on reducers: ``tool_results``, ``observations``, ``errors`` and
``completed_steps`` are append-only across nodes, so they are annotated with
``operator.add``. ``plan`` is deliberately *not* append-only - re-planning
replaces it, and keeping the superseded plan around would let a stale step be
executed twice.
"""

from __future__ import annotations

import operator
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.domain import BusinessObjective, IntentType, RiskLevel
from app.schemas.tool_contract import ToolResult


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    #: Dropped during re-planning because it became unnecessary.
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """One unit of analytical work in an investigation plan."""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    #: Name of the tool to invoke, e.g. "price_elasticity".
    tool_name: str
    #: Why the Supervisor added this step. Shown in the UI trace; also what the
    #: Critic checks the evidence against.
    rationale: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: step_ids that must complete before this one runs.
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING


class InvestigationPlan(BaseModel):
    """A dynamically generated plan. Replaced wholesale on re-plan."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    #: 0 for the initial plan, incremented on each re-plan.
    revision: int = 0
    objective: BusinessObjective
    steps: list[PlanStep] = Field(default_factory=list)
    #: Populated on revisions >= 1: what the previous plan failed to establish.
    replan_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status is StepStatus.PENDING]


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class Hypothesis(BaseModel):
    """A candidate explanation produced by the Root Cause agent.

    Explicitly modelled rather than left as free text so the Critic can check
    that each surviving hypothesis is actually backed by a tool result, and so
    the UI can show which explanations were considered and rejected.
    """

    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    #: trace_ids of the ToolResults that bear on this hypothesis.
    supporting_evidence: list[str] = Field(default_factory=list)
    refuting_evidence: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    """The Supervisor's reading of a tool result.

    Kept separate from the ``ToolResult`` itself: the result is ground truth and
    immutable, the observation is interpretation and may be revised. Conflating
    them is how numbers get quietly rewritten.
    """

    step_id: str
    tool_name: str
    #: One-line interpretation. Never restates a number the tool did not return.
    summary: str
    #: Did this move the investigation forward?
    informative: bool = True


class EvidenceItem(BaseModel):
    """A validated fact admitted into the final recommendation."""

    claim: str
    #: trace_id of the ToolResult backing this claim. Required - a claim without
    #: a source is exactly what the Critic exists to reject.
    source_trace_id: str
    source_tool: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class CriticVerdict(BaseModel):
    """Output of the Critic agent (section 19 of the brief)."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    required_followup: list[str] = Field(default_factory=list)


class ScenarioComparison(BaseModel):
    """One option in a recommendation's trade-off table."""

    name: str
    description: str
    expected_revenue_impact: float | None = None
    expected_profit_impact: float | None = None
    expected_margin_impact: float | None = None
    risk: RiskLevel = RiskLevel.MEDIUM
    #: trace_ids of the tool results this scenario was computed from.
    source_trace_ids: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Final business output (section 45 of the brief)."""

    executive_summary: str
    root_cause: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    scenarios: list[ScenarioComparison] = Field(default_factory=list)
    recommended_action: str
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    #: True when the impact crosses the human-approval threshold.
    requires_human_approval: bool = False


class AgentState(TypedDict, total=False):
    """LangGraph state for a single investigation.

    ``total=False`` because nodes populate this incrementally; a node reads the
    keys it needs and returns only the keys it changed.
    """

    # --- request ---
    investigation_id: str
    trace_id: str
    user_question: str
    user_id: str | None

    # --- understanding ---
    intent: IntentType | None
    objective: BusinessObjective | None
    #: Products, stores, regions and categories named in the question. Extracted
    #: at classification so plans can scope their tool calls rather than running
    #: platform-wide, and carried in state because re-planning needs them too.
    entities: dict[str, list[str]]

    # --- planning (replaced on re-plan, not appended) ---
    plan: InvestigationPlan | None
    #: What the Critic said was missing, passed verbatim into the next plan. A
    #: re-plan that does not know precisely what was insufficient repeats the
    #: same steps.
    replan_reason: str | None

    # --- execution (append-only across nodes) ---
    completed_steps: Annotated[list[str], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    observations: Annotated[list[Observation], operator.add]
    errors: Annotated[list[str], operator.add]

    # --- reasoning ---
    hypotheses: list[Hypothesis]
    evidence: list[EvidenceItem]
    critic_verdict: CriticVerdict | None
    confidence: float

    # --- output ---
    final_recommendation: Recommendation | None

    # --- budget / control (section 18) ---
    iteration: int
    tool_call_count: int
    tokens_used: int
    started_at: datetime
    replan_count: int


def new_agent_state(
    user_question: str,
    *,
    user_id: str | None = None,
    investigation_id: str | None = None,
    trace_id: str | None = None,
) -> AgentState:
    """Build a fresh state with all control counters initialised.

    Centralised so a new node can never observe a missing counter and default
    it to something inconsistent.
    """
    return AgentState(
        investigation_id=investigation_id or str(uuid.uuid4()),
        trace_id=trace_id or str(uuid.uuid4()),
        user_question=user_question,
        user_id=user_id,
        intent=None,
        objective=None,
        entities={},
        plan=None,
        replan_reason=None,
        completed_steps=[],
        tool_results=[],
        observations=[],
        errors=[],
        hypotheses=[],
        evidence=[],
        critic_verdict=None,
        confidence=0.0,
        final_recommendation=None,
        iteration=0,
        tool_call_count=0,
        tokens_used=0,
        started_at=datetime.now(UTC),
        replan_count=0,
    )
