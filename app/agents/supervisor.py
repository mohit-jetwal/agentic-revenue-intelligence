"""The Supervisor: intent, planning, and tool selection.

Decides *what to investigate and in what order*. It performs no calculation —
every number in an investigation comes from a tool result.

**The property this agent is judged on is minimum sufficiency.** A forecast
question must reach the forecasting tool and stop. Fanning out to elasticity,
uplift and optimisation because the registry happens to contain them produces a
report nobody asked for, burns budget, and looks impressive in a demo while
being wrong. The prompt says so, the plan is validated against the registry, and
:func:`plan_investigation` truncates a plan that exceeds the configured step cap.

**Planning is structured output, not parsed text.** The plan comes back as a
validated ``InvestigationPlan`` via forced tool-calling. A malformed plan then
fails at the provider boundary rather than propagating into the graph as a
string nobody checked.

**Unknown tools are dropped, not attempted.** A model asked to select from a
registry will occasionally invent a plausible name. Executing it would produce a
confusing runtime error deep in the loop; dropping it with a warning keeps the
investigation running on the steps that are real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.llm.base import LLMProvider, Message
from app.observability.logging import get_logger
from app.schemas.agent_state import InvestigationPlan, PlanStep
from app.schemas.domain import BusinessObjective, IntentType
from app.schemas.tool_contract import ToolResult, ToolSpec
from prompts.registry import load_prompt

logger = get_logger(__name__)

PROMPT_NAME = "supervisor"
PROMPT_VERSION = "v1"

#: Hard cap on steps in one plan. Not a performance guard - a plan with fifteen
#: steps is almost always a model fanning out across every tool it was shown
#: rather than answering the question.
MAX_PLAN_STEPS = 6


class IntentClassification(BaseModel):
    """What the user is asking for."""

    intent: IntentType
    objective: BusinessObjective
    #: Products, stores, regions or categories named in the question. Extracted
    #: so a plan can scope its tool calls rather than running platform-wide.
    entities: dict[str, list[str]] = Field(default_factory=dict)
    reasoning: str = ""


class PlannedStep(BaseModel):
    """One step, as the model proposes it."""

    tool_name: str
    rationale: str = Field(description="Why this step is needed to answer the question.")
    parameters: dict[str, object] = Field(default_factory=dict)


class ProposedPlan(BaseModel):
    """The model's plan, before validation against the registry."""

    steps: list[PlannedStep] = Field(default_factory=list)
    reasoning: str = ""


@dataclass
class SupervisorAgent:
    """Plans investigations and interprets what came back."""

    provider: LLMProvider
    tools: list[ToolSpec] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    max_steps: int = MAX_PLAN_STEPS

    @property
    def system_prompt(self) -> str:
        return load_prompt(PROMPT_NAME, self.prompt_version)

    @property
    def tool_names(self) -> set[str]:
        return {spec.name for spec in self.tools}

    # -- intent -------------------------------------------------------------

    def classify(self, question: str) -> IntentClassification:
        """Work out what is being asked before deciding how to answer it.

        Separate from planning because the two fail differently: a
        misclassified intent produces a coherent plan for the wrong question,
        which is far harder to spot than a bad plan for the right one.
        """
        result, _ = self.provider.complete_structured(
            [Message(role="user", content=question)],
            IntentClassification,
            system=self._intent_prompt(),
        )
        logger.info(
            "supervisor.intent", intent=str(result.intent), objective=str(result.objective)
        )
        return result

    # -- planning -----------------------------------------------------------

    def plan(
        self,
        question: str,
        classification: IntentClassification,
        *,
        previous: InvestigationPlan | None = None,
        replan_reason: str | None = None,
        results: list[ToolResult] | None = None,
    ) -> InvestigationPlan:
        """Produce a plan, or a revision of one.

        ``replan_reason`` is what the Critic said was missing. It is passed
        through verbatim rather than summarised, because a re-plan that does not
        know precisely what was insufficient will usually repeat the same steps.
        """
        proposal, _ = self.provider.complete_structured(
            [Message(role="user", content=self._planning_message(
                question, classification, previous, replan_reason, results
            ))],
            ProposedPlan,
            system=self.system_prompt,
            # The one call where a stronger model clearly pays for itself: a bad
            # plan wastes every tool call that follows it, and tool calls are
            # far more expensive than the token difference.
            use_planner=True,
        )
        return self._validate(proposal, classification, previous, replan_reason)

    def _validate(
        self,
        proposal: ProposedPlan,
        classification: IntentClassification,
        previous: InvestigationPlan | None,
        replan_reason: str | None,
    ) -> InvestigationPlan:
        """Turn a proposal into a plan the graph can execute.

        Two filters. Unknown tools are dropped - a model selecting from a
        registry occasionally invents a plausible name, and executing it would
        fail deep in the loop rather than here. And the plan is truncated to
        ``max_steps``, because a long plan is nearly always a fan-out across
        every tool the model was shown.
        """
        steps: list[PlanStep] = []
        unknown: list[str] = []

        for proposed in proposal.steps:
            if proposed.tool_name not in self.tool_names:
                unknown.append(proposed.tool_name)
                continue
            steps.append(
                PlanStep(
                    tool_name=proposed.tool_name,
                    rationale=proposed.rationale,
                    parameters=dict(proposed.parameters),
                )
            )

        truncated = len(steps) > self.max_steps
        steps = steps[: self.max_steps]

        if unknown:
            logger.warning("supervisor.unknown_tools", tools=unknown)
        if truncated:
            logger.warning("supervisor.plan_truncated", cap=self.max_steps)

        plan = InvestigationPlan(
            revision=(previous.revision + 1) if previous else 0,
            objective=classification.objective,
            steps=steps,
            replan_reason=replan_reason,
        )
        logger.info(
            "supervisor.planned",
            revision=plan.revision,
            steps=[s.tool_name for s in steps],
            dropped=len(unknown),
        )
        return plan

    # -- observation --------------------------------------------------------

    def observe(self, step: PlanStep, result: ToolResult) -> str:
        """A one-line reading of what a tool returned.

        Deliberately not an LLM call. The observation is bookkeeping - did this
        move the investigation forward - and spending a round trip on it would
        double the token cost of every step for a sentence the Critic re-derives
        from the result anyway.
        """
        if result.status == "error":
            detail = result.error.message if result.error else "unknown error"
            return f"{step.tool_name} failed: {detail}"

        parts = [f"{step.tool_name} returned a result"]
        if result.warnings:
            parts.append(f"with {len(result.warnings)} warning(s)")
        if result.status == "partial":
            parts.append("(partial)")
        return " ".join(parts)

    # -- prompt assembly ----------------------------------------------------

    def _intent_prompt(self) -> str:
        return (
            "Classify a commercial analytics question for a CPG/Retail business.\n\n"
            "Return the intent, the business objective it serves, and any "
            "products, stores, regions or categories named in the question. "
            "Extract entities exactly as written - they are used to scope tool "
            "calls, and a paraphrased identifier will not match anything.\n\n"
            "Do not answer the question. Do not estimate any number."
        )

    def _planning_message(
        self,
        question: str,
        classification: IntentClassification,
        previous: InvestigationPlan | None,
        replan_reason: str | None,
        results: list[ToolResult] | None,
    ) -> str:
        lines = [
            f"QUESTION: {question}",
            f"INTENT: {classification.intent}",
            f"OBJECTIVE: {classification.objective}",
        ]
        if classification.entities:
            lines.append(f"ENTITIES: {classification.entities}")

        lines.append("")
        lines.append("AVAILABLE TOOLS:")
        for spec in self.tools:
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  parameters: {sorted(spec.input_schema.get('properties', {}))}")

        if previous is not None and replan_reason:
            lines.extend(
                [
                    "",
                    "THIS IS A RE-PLAN.",
                    f"Previous steps: {[s.tool_name for s in previous.steps]}",
                    f"What was insufficient: {replan_reason}",
                    "",
                    "Plan only what is needed to close that specific gap. Do not "
                    "repeat a step that already returned a usable result.",
                ]
            )

        if results:
            lines.extend(["", "RESULTS SO FAR:"])
            for result in results[-6:]:
                lines.append(f"- {result.tool_name}: {result.status}")

        lines.extend(
            [
                "",
                f"Produce the MINIMUM sufficient plan, at most {self.max_steps} steps.",
            ]
        )
        return "\n".join(lines)


def build_supervisor(
    provider: LLMProvider, tools: list[ToolSpec], **kwargs: object
) -> SupervisorAgent:
    """Construct a Supervisor over the given tool registry."""
    return SupervisorAgent(provider=provider, tools=tools, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "MAX_PLAN_STEPS",
    "PROMPT_NAME",
    "PROMPT_VERSION",
    "IntentClassification",
    "PlannedStep",
    "ProposedPlan",
    "SupervisorAgent",
    "build_supervisor",
]
