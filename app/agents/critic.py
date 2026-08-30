"""The Critic: does the evidence support the conclusion?

The agent whose job is to say no.

**Why it is separate from the Supervisor.** The Supervisor built the plan and
has every incentive to believe it worked — an agent asked to plan an
investigation *and* judge whether its own investigation succeeded will usually
find that it did. Splitting the roles is not ceremony; it is the only structural
defence against a confident conclusion drawn from thin evidence.

**What it checks, in order of what actually goes wrong.**

*Unsupported claims.* Every claim in a recommendation must trace to a tool
result. This is the failure that matters most, because a fluent sentence with no
source behind it is indistinguishable from one with a source until someone looks.

*Contradiction.* Two tools can disagree — a forecast that says demand is rising
and an uplift estimate that says the promotion did nothing. Presenting either
alone would be a choice nobody declared.

*Sufficiency.* Whether the evidence gathered actually answers the question
asked, rather than a neighbouring one that was easier to answer.

*Ignored warnings.* A result carrying `validation_status: failed` or a
differential-censoring warning is weaker evidence. A conclusion that quotes its
number without its caveat has laundered a qualified finding into a fact.

**Its verdict routes the graph.** `valid=False` with `required_followup` sends
the investigation back to planning with a specific gap named, bounded by
`max_replans`. That bound matters: a Critic that is never satisfied is the
realistic way an agent loops forever.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.base import LLMProvider, Message
from app.observability.logging import get_logger
from app.schemas.agent_state import CriticVerdict, EvidenceItem
from app.schemas.tool_contract import ToolResult
from prompts.registry import load_prompt

logger = get_logger(__name__)

PROMPT_NAME = "critic"
PROMPT_VERSION = "v1"


class CriticAssessment(BaseModel):
    """The Critic's structured judgement."""

    sufficient: bool = Field(
        description="Does the evidence answer the question that was asked?"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    #: Specific problems: unsupported claims, contradictions, ignored caveats.
    issues: list[str] = Field(default_factory=list)
    #: What would close the gap. Read verbatim by the Supervisor on a re-plan,
    #: so vagueness here produces a repeat of the same steps.
    required_followup: list[str] = Field(default_factory=list)
    #: A one-sentence business note, not a chain-of-thought dump - see
    #: `app.agents.supervisor.IntentClassification.rationale` for why a bare
    #: field named "reasoning" is actively dangerous here: it reads to the
    #: model as a request to reproduce its internal reasoning in the response,
    #: which the Claude 5 family can refuse outright.
    rationale: str = Field(
        default="", description="One-sentence note on the verdict, for the trace."
    )


@dataclass
class CriticAgent:
    """Validates evidence before a recommendation is allowed out."""

    provider: LLMProvider
    prompt_version: str = PROMPT_VERSION
    #: Deterministic checks run before the model is asked anything. They cost
    #: nothing and catch the failures that are decidable without judgement.
    mechanical_checks: bool = True

    @property
    def system_prompt(self) -> str:
        return load_prompt(PROMPT_NAME, self.prompt_version)

    def review(
        self,
        question: str,
        results: list[ToolResult],
        *,
        evidence: list[EvidenceItem] | None = None,
        draft: str | None = None,
    ) -> CriticVerdict:
        """Judge whether the investigation can support a conclusion."""
        mechanical = self._mechanical_issues(results) if self.mechanical_checks else []

        if not results:
            # No model call needed. Nothing was gathered, so nothing is
            # supported, and asking an LLM to confirm that wastes a round trip.
            return CriticVerdict(
                valid=False,
                confidence=0.0,
                issues=["no tool results were gathered"],
                required_followup=["run at least one analytical tool"],
            )

        message = self._review_message(question, results, evidence, draft)
        assessment, _ = self.provider.complete_structured(
            [Message(role="user", content=message)],
            CriticAssessment,
            system=self.system_prompt,
            # The Critic decides whether the investigation continues, so its
            # judgement compounds the same way the plan's does. Classification
            # and drafting stay on the worker model: both are constrained
            # enough that the stronger model has little room to be better.
            use_planner=True,
        )

        issues = mechanical + list(assessment.issues)
        # A mechanical failure overrides the model's judgement. Those checks are
        # decidable - a result whose own validation_status is `failed` is not
        # something a confident reading can rescue.
        blocking = any(issue.startswith("BLOCKING") for issue in mechanical)
        valid = assessment.sufficient and not blocking

        verdict = CriticVerdict(
            valid=valid,
            confidence=min(assessment.confidence, 0.5) if blocking else assessment.confidence,
            issues=issues,
            required_followup=list(assessment.required_followup),
        )
        logger.info(
            "critic.reviewed",
            valid=verdict.valid,
            confidence=round(verdict.confidence, 3),
            issues=len(issues),
            blocking=blocking,
        )
        return verdict

    def _mechanical_issues(self, results: list[ToolResult]) -> list[str]:
        """Checks that need no judgement, so no tokens.

        Prefixed BLOCKING when the finding is decidable rather than a matter of
        degree - those override whatever the model concludes.
        """
        issues: list[str] = []

        for result in results:
            if result.status == "error":
                detail = result.error.message if result.error else "unknown"
                issues.append(f"{result.tool_name} failed: {detail}")
                continue

            payload = result.result or {}
            status = payload.get("validation_status")
            if status == "failed":
                issues.append(
                    f"BLOCKING: {result.tool_name} reports validation_status "
                    f"'failed' - its number must not be presented as causal"
                )
            elif status == "warnings":
                issues.append(
                    f"{result.tool_name} passed validation with warnings; its "
                    f"caveats must appear alongside its number"
                )

            if result.status == "partial":
                issues.append(
                    f"{result.tool_name} returned a partial result "
                    f"({len(result.warnings)} warning(s))"
                )

        return issues

    def _review_message(
        self,
        question: str,
        results: list[ToolResult],
        evidence: list[EvidenceItem] | None,
        draft: str | None,
    ) -> str:
        lines = [f"QUESTION: {question}", "", "EVIDENCE GATHERED:"]

        for result in results:
            lines.append(f"\n--- {result.tool_name} ({result.status}) ---")
            # The full envelope, not just the numbers. The Critic cannot judge
            # whether a caveat was ignored without seeing the caveat.
            lines.append(f"result: {result.result}")
            if result.assumptions:
                lines.append(f"assumptions: {result.assumptions}")
            if result.warnings:
                lines.append(f"warnings: {result.warnings}")

        if evidence:
            lines.extend(["", "CLAIMS MADE:"])
            lines.extend(f"- {item.claim} (source: {item.source_tool})" for item in evidence)

        if draft:
            lines.extend(["", "DRAFT RECOMMENDATION:", draft])

        lines.extend(
            [
                "",
                "Judge whether this evidence answers the question ASKED. If not, "
                "say precisely what is missing - your followup is read verbatim "
                "when the investigation re-plans, and a vague one produces a "
                "repeat of the same steps.",
            ]
        )
        return "\n".join(lines)


__all__ = ["PROMPT_NAME", "PROMPT_VERSION", "CriticAgent", "CriticAssessment"]
