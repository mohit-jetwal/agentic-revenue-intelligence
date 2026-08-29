"""The Recommendation agent: turning evidence into a business answer.

The only agent that writes prose a person will read, which makes it the one
place invented numbers could reach a decision.

Three defences, applied in order:

**Evidence is extracted before it is written up.** Each claim is built from a
tool result and carries that result's ``trace_id``, so the link between a
sentence and its source exists as data rather than as a hope about what the
model was looking at.

**Every numeral is checked against the tool results.** See
:mod:`app.guardrails.output_validation`. This catches the failure the prompt can
only ask about.

**Impact above a threshold requires human approval.** The graph interrupts
before the recommendation leaves the system. A large recommendation is not more
likely to be wrong, but it is more expensive to be wrong about.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.guardrails.output_validation import validate_output
from app.llm.base import LLMProvider, Message
from app.observability.logging import get_logger
from app.schemas.agent_state import EvidenceItem, Recommendation, ScenarioComparison
from app.schemas.domain import RiskLevel
from app.schemas.tool_contract import ToolResult
from prompts.registry import load_prompt

logger = get_logger(__name__)

PROMPT_NAME = "recommendation"
PROMPT_VERSION = "v1"


class DraftRecommendation(BaseModel):
    """What the model writes, before validation."""

    executive_summary: str
    root_cause: str | None = None
    recommended_action: str
    #: Claims the model believes it made, each naming the tool behind it.
    claims: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    #: Absolute profit impact, used to decide whether a human must approve.
    #: Zero when the recommendation carries no financial estimate.
    estimated_profit_impact: float = 0.0


@dataclass
class RecommendationAgent:
    """Synthesises evidence into a business recommendation."""

    provider: LLMProvider
    approval_threshold: float = 1_000_000.0
    prompt_version: str = PROMPT_VERSION

    @property
    def system_prompt(self) -> str:
        return load_prompt(PROMPT_NAME, self.prompt_version)

    def synthesise(
        self,
        question: str,
        results: list[ToolResult],
        *,
        critic_issues: list[str] | None = None,
        critic_satisfied: bool = True,
        truncated: bool = False,
    ) -> Recommendation:
        """Produce the final recommendation, validated."""
        evidence = extract_evidence(results)

        message = self._message(question, results, critic_issues, truncated)
        draft, _ = self.provider.complete_structured(
            [Message(role="user", content=message)],
            DraftRecommendation,
            system=self.system_prompt,
        )

        prose = " ".join(
            filter(None, [draft.executive_summary, draft.root_cause, draft.recommended_action])
        )
        validation = validate_output(prose, results)

        warnings = list(validation.warnings())
        confidence = draft.confidence
        if not validation.clean:
            # An unsourced figure is not automatically wrong - it may be
            # arithmetic over sourced values - but it is not established either,
            # and the confidence should not claim otherwise.
            confidence = min(confidence, 0.6)
        if truncated:
            warnings.insert(
                0,
                "INCOMPLETE INVESTIGATION: the execution budget was exhausted "
                "before every planned step ran. This conclusion rests on partial "
                "evidence.",
            )
            confidence = min(confidence, 0.5)
        if critic_issues:
            warnings.extend(critic_issues)
        if not critic_satisfied:
            # The graph only reaches here with an unsatisfied Critic when the
            # re-plan budget ran out, which means the objection was never
            # answered. Carrying the model's own confidence forward would let a
            # recommendation the Critic rejected arrive looking settled.
            confidence = min(confidence, 0.5)

        requires_approval = abs(draft.estimated_profit_impact) >= self.approval_threshold

        recommendation = Recommendation(
            executive_summary=draft.executive_summary,
            root_cause=draft.root_cause,
            evidence=evidence,
            scenarios=_scenarios_from(results),
            recommended_action=draft.recommended_action,
            assumptions=list(draft.assumptions) + _tool_assumptions(results),
            risks=list(draft.risks) + warnings,
            confidence=confidence,
            requires_human_approval=requires_approval,
        )
        logger.info(
            "recommendation.synthesised",
            evidence=len(evidence),
            unsourced=len(validation.unsourced),
            confidence=round(confidence, 3),
            requires_approval=requires_approval,
        )
        return recommendation

    def _message(
        self,
        question: str,
        results: list[ToolResult],
        critic_issues: list[str] | None,
        truncated: bool,
    ) -> str:
        lines = [f"QUESTION: {question}", "", "TOOL RESULTS:"]
        for result in results:
            lines.append(f"\n--- {result.tool_name} ({result.status}) ---")
            lines.append(f"{result.result}")
            if result.assumptions:
                lines.append(f"assumptions: {result.assumptions}")
            if result.warnings:
                lines.append(f"warnings: {result.warnings}")

        if critic_issues:
            lines.extend(["", "THE CRITIC RAISED:"])
            lines.extend(f"- {issue}" for issue in critic_issues)
            lines.append(
                "Address these in the recommendation rather than writing around "
                "them."
            )

        if truncated:
            lines.extend(
                [
                    "",
                    "THE INVESTIGATION WAS TRUNCATED by the execution budget. Say "
                    "so plainly and do not present a confident conclusion.",
                ]
            )

        lines.extend(
            [
                "",
                "Every number you write must appear in a tool result above. "
                "Report the estimated profit impact as a number so the approval "
                "threshold can be applied.",
            ]
        )
        return "\n".join(lines)


def extract_evidence(results: list[ToolResult]) -> list[EvidenceItem]:
    """One evidence item per usable tool result, linked to its trace.

    Built from the results rather than from the model's prose. A claim the model
    *said* it sourced and a claim that *is* sourced are different things, and
    only the second survives review.
    """
    evidence: list[EvidenceItem] = []
    seen: set[str] = set()

    for result in results:
        if result.status == "error" or not result.result:
            continue
        claim = _headline(result)
        # Deduplicated by claim. A re-planned investigation runs the same tool
        # again and gets the same answer; listing it three times pads the
        # evidence without adding any, which is exactly the appearance of
        # corroboration without the substance.
        if claim in seen:
            continue
        seen.add(claim)
        evidence.append(
            EvidenceItem(
                claim=claim,
                source_trace_id=result.trace_id or "",
                source_tool=result.tool_name,
                confidence=result.confidence,
            )
        )
    return evidence


def _headline(result: ToolResult) -> str:
    """The one figure a result is about, chosen by tool.

    Hand-mapped rather than guessed from the payload: which field matters
    depends on the question the tool answers, and a generic "first numeric
    field" rule would quote a sample size as the finding.
    """
    payload = result.result or {}
    headlines: dict[str, str] = {
        "forecast_demand": "total_predicted_units",
        "estimate_promo_uplift": "incremental_profit",
        "estimate_price_elasticity": "elasticity",
        "allocate_promotion_budget": "expected_incremental_profit",
        "optimize_price": "recommended_price",
        "simulate_scenario": "profit_impact",
    }
    key = headlines.get(result.tool_name)
    if key and key in payload:
        return f"{result.tool_name}: {key} = {payload[key]}"
    return f"{result.tool_name} returned a result"


def _scenarios_from(results: list[ToolResult]) -> list[ScenarioComparison]:
    """Trade-off options, where a tool produced comparable alternatives."""
    scenarios: list[ScenarioComparison] = []
    for result in results:
        payload = result.result or {}
        if result.tool_name == "optimize_price" and payload.get("recommended_price"):
            scenarios.append(
                ScenarioComparison(
                    name=f"price {payload['recommended_price']}",
                    description=f"change of {payload.get('change_pct', 0):+.1%}",
                    expected_profit_impact=payload.get("expected_profit_impact"),
                    expected_revenue_impact=payload.get("expected_revenue_impact"),
                    risk=_risk(payload.get("risk")),
                    source_trace_ids=[result.trace_id or ""],
                )
            )
        elif result.tool_name == "simulate_scenario":
            scenarios.append(
                ScenarioComparison(
                    name=str(payload.get("scenario_name", "scenario")),
                    description=f"over {payload.get('horizon_days', 0)} days",
                    expected_profit_impact=payload.get("profit_impact"),
                    expected_revenue_impact=payload.get("revenue_impact"),
                    risk=_risk(payload.get("risk")),
                    source_trace_ids=[result.trace_id or ""],
                )
            )
    return scenarios


def _risk(value: object) -> RiskLevel:
    try:
        return RiskLevel(str(value))
    except ValueError:
        return RiskLevel.MEDIUM


def _tool_assumptions(results: list[ToolResult]) -> list[str]:
    """Every assumption the tools attached, deduplicated.

    Carried through verbatim. These are the conditions under which the numbers
    mean what they say, and a recommendation that drops them has converted a
    qualified finding into a fact.
    """
    seen: dict[str, None] = {}
    for result in results:
        for assumption in result.assumptions:
            seen.setdefault(assumption, None)
    return list(seen)


__all__ = [
    "PROMPT_NAME",
    "PROMPT_VERSION",
    "DraftRecommendation",
    "RecommendationAgent",
    "extract_evidence",
]
