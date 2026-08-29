"""A deliberately weak planner, used as the floor.

**The problem this solves.** A golden-set run against the stub provider grades
whatever the stub was scripted to return. Script the correct plan for every
question and the score is 100%, which measures nothing about the system and
everything about the person who wrote the script.

So the stub is scripted from a *policy* instead: keyword matching over the
question text, with no reasoning of any kind. That turns the stub run into a
real measurement of a real planner - one that is genuinely bad in the ways
keyword matching is always bad, and whose score is therefore a floor.

**What the floor is for.** A language model in the Supervisor seat costs money
and latency. If it cannot beat regular expressions on this set, it is not
earning either. Recording both numbers side by side is the only way that
question gets asked at all; a benchmark with one system in it always looks fine.

**Where it is expected to fail**, and these are the interesting rows in the
report rather than defects to fix:

*It cannot abstain.* Keywords match a stockout question to the uplift tool as
happily as to anything else. Every abstention question is a loss, which is the
point - knowing coverage requires judgement, and this policy has none.

*It cannot read a trap.* The `bad_promo` questions say "sales rose", and a
keyword planner has no way to know that positive uplift and a worthwhile
promotion are different claims.

*It fans out on ambiguity.* Several keywords match, so several tools are called,
which is what the fan-out penalty exists to price.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.critic import CriticAssessment
from app.agents.recommendation import DraftRecommendation
from app.agents.supervisor import IntentClassification, PlannedStep, ProposedPlan
from app.llm.stub import StubProvider
from app.schemas.domain import BusinessObjective, IntentType
from evaluation.golden_set import GoldenQuestion

#: Keyword -> tool. Ordered by how specific the trigger is; every match fires,
#: which is how the fan-out this policy is criticised for actually happens.
_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("promotion", "promoted", "discount"), "estimate_promo_uplift"),
    (("price", "raised", "elasticity"), "estimate_price_elasticity"),
    (("forecast", "plan for", "demand should"), "forecast_demand"),
    (("budget", "allocate", "spend across"), "allocate_promotion_budget"),
    (("what if", "scenario", "simulate"), "simulate_scenario"),
)

#: Keyword -> intent. Same crudeness, deliberately.
_INTENTS: tuple[tuple[tuple[str, ...], IntentType], ...] = (
    (("forecast", "plan for"), IntentType.FORECAST),
    (("caused", "why", "what happened"), IntentType.ROOT_CAUSE),
    (("promotion", "promoted"), IntentType.PROMOTION_DECISION),
    (("price", "raised"), IntentType.PRICE_DECISION),
)


def select_tools(question: str, available: set[str]) -> list[str]:
    """Every tool whose keywords appear. No ranking, no judgement."""
    text = question.lower()
    selected = [
        tool
        for triggers, tool in _RULES
        if tool in available and any(t in text for t in triggers)
    ]
    if not selected and "forecast_demand" in available:
        # A planner with nothing to go on has to do something. Guessing the most
        # common tool is what a fallback rule looks like, and it is wrong often.
        selected = ["forecast_demand"]
    return selected


def classify(question: str) -> IntentType:
    text = question.lower()
    for triggers, intent in _INTENTS:
        if any(t in text for t in triggers):
            return intent
    return IntentType.PERFORMANCE_EXPLANATION


@dataclass
class KeywordBaseline:
    """Scripts a stub provider to follow the keyword policy."""

    available_tools: set[str]

    def script(self, provider: StubProvider, questions: list[GoldenQuestion]) -> None:
        """Register a plan per question, keyed on a distinctive substring.

        Keyed on the question id rather than its prose: two golden questions
        built from the same scenario shape differ only by product id, and
        matching on wording would make one question's plan answer another's.
        """
        for question in questions:
            key = question.question_id
            tagged = _tag(question)

            provider.script_structured(
                IntentClassification(
                    intent=classify(question.question),
                    objective=_objective(question.question),
                    entities=question.entities(),
                ),
                when_contains=key,
            )
            provider.script_structured(
                ProposedPlan(
                    steps=[
                        PlannedStep(
                            tool_name=tool,
                            rationale="keyword match on the question text",
                            parameters=_parameters(question),
                        )
                        for tool in select_tools(question.question, self.available_tools)
                    ]
                ),
                when_contains=key,
            )
            # The policy has no way to judge sufficiency, so it always says yes.
            # An always-satisfied Critic is the other half of a bad agent, and
            # the report should show what that costs.
            provider.script_structured(
                CriticAssessment(sufficient=True, confidence=0.7),
                when_contains=key,
            )
            provider.script_structured(
                DraftRecommendation(
                    executive_summary=tagged,
                    recommended_action="Proceed on the evidence gathered.",
                    confidence=0.75,
                    estimated_profit_impact=0.0,
                ),
                when_contains=key,
            )


def _tag(question: GoldenQuestion) -> str:
    """A summary that states no number, because this policy has none to state."""
    return (
        f"[{question.question_id}] Keyword baseline: tools were selected by "
        f"matching words in the question. No reasoning was applied."
    )


def _objective(question: str) -> BusinessObjective:
    text = question.lower()
    if "profit" in text or "worth" in text:
        return BusinessObjective.MAXIMISE_PROFIT
    if "forecast" in text or "plan for" in text:
        return BusinessObjective.MAXIMISE_VOLUME
    return BusinessObjective.EXPLAIN_PERFORMANCE


def _parameters(question: GoldenQuestion) -> dict[str, object]:
    """Scope the call to what the scenario actually touched.

    Handed over rather than parsed. Entity extraction is a separate capability,
    and letting a weak planner also fail at it would confound two measurements.
    """
    params: dict[str, object] = {}
    if question.product_ids:
        params["product_id"] = question.product_ids[0]
    if question.region:
        params["region"] = question.region
    if question.start_date:
        params["start_date"] = question.start_date.isoformat()
    if question.end_date:
        params["end_date"] = question.end_date.isoformat()
    return params


__all__ = ["KeywordBaseline", "classify", "select_tools"]
