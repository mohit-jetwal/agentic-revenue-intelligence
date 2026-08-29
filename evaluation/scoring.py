"""Scoring one investigation against a question whose answer is known.

**Four dimensions, deliberately not averaged into one number.** They fail for
different reasons and a single score hides which: an agent that selects perfect
tools and then writes an unsupported conclusion scores the same as one that
picks badly and reports honestly, and those are not the same system.

*Tool selection* — were the required tools called, and was the workflow minimum
sufficient? Fan-out is penalised, because calling every tool available looks
thorough and is the cheapest way to appear rigorous without being it.

*Evidence* — did the calls actually produce usable results? A plan that names
the right tool and then fails to execute it has not answered anything.

*Direction* — does the conclusion point the way the injected truth points? Only
direction, never magnitude. The scenario record fixes the *sign* of each effect;
its size depends on estimator choices that are Step 7 and 8's business, and
grading magnitude here would re-litigate those with a cruder instrument.

*Abstention* — on a question the toolset cannot answer, did it decline? Scored
inversely: confidence is the failure. This is the dimension most benchmarks
leave out, and the one that decides whether the other three can be trusted.

**Partial credit is intentional.** A binary pass/fail on a 20-question set moves
in jumps too coarse to tell a prompt improvement from noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.agent_state import AgentState
from evaluation.golden_set import GoldenQuestion

#: Above this, a recommendation on an unanswerable question counts as
#: confabulation rather than a hedged observation. Set where a business reader
#: would start treating a statement as actionable.
CONFIDENT = 0.6

#: Weight applied to each discouraged tool that was called. One extra call is a
#: judgement call; three is a fan-out.
FANOUT_PENALTY = 0.25


@dataclass
class QuestionScore:
    """How one investigation did, dimension by dimension."""

    question_id: str
    scenario_id: str
    label: str
    answerable: bool

    tool_selection: float = 0.0
    evidence: float = 0.0
    direction: float = 0.0
    abstention: float = 0.0

    #: Which dimensions apply. An unanswerable question has no direction to get
    #: right, and averaging a structural zero into its score would report a
    #: coverage gap as a failure of reasoning.
    applicable: tuple[str, ...] = ()

    tools_called: tuple[str, ...] = ()
    missing_tools: tuple[str, ...] = ()
    extra_tools: tuple[str, ...] = ()
    #: Required tools that were called and refused for lack of data. Tracked
    #: apart from `missing_tools` because the two are different failures with
    #: different fixes: one is the agent planning badly, the other is a trained
    #: artefact not covering the product or window asked about. Averaging them
    #: into one number makes a data gap look like a reasoning gap.
    starved_tools: tuple[str, ...] = ()
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Mean of the applicable dimensions only."""
        if not self.applicable:
            return 0.0
        return sum(getattr(self, name) for name in self.applicable) / len(self.applicable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "scenario_id": self.scenario_id,
            "label": self.label,
            "answerable": self.answerable,
            "overall": round(self.overall, 4),
            "dimensions": {
                name: round(getattr(self, name), 4) for name in self.applicable
            },
            "tools_called": list(self.tools_called),
            "missing_tools": list(self.missing_tools),
            "extra_tools": list(self.extra_tools),
            "starved_tools": list(self.starved_tools),
            "confidence": round(self.confidence, 4),
            "notes": list(self.notes),
        }


def score_question(question: GoldenQuestion, state: AgentState) -> QuestionScore:
    """Grade one finished investigation."""
    called = tuple(dict.fromkeys(r.tool_name for r in state.get("tool_results", [])))
    recommendation = state.get("final_recommendation")
    confidence = float(recommendation.confidence) if recommendation else 0.0

    score = QuestionScore(
        question_id=question.question_id,
        scenario_id=question.scenario_id,
        label=question.label,
        answerable=question.answerable,
        tools_called=called,
        confidence=confidence,
    )

    if question.answerable:
        _score_answerable(question, state, called, score)
    else:
        _score_abstention(question, state, called, score, recommendation)

    return score


def _score_answerable(
    question: GoldenQuestion,
    state: AgentState,
    called: tuple[str, ...],
    score: QuestionScore,
) -> None:
    score.applicable = ("tool_selection", "evidence", "direction")

    # -- tool selection --
    required = question.required_tools
    missing = required - set(called)
    extra = set(called) & question.discouraged_tools
    score.missing_tools = tuple(sorted(missing))
    score.extra_tools = tuple(sorted(extra))

    recall = (len(required) - len(missing)) / len(required) if required else 1.0
    score.tool_selection = max(0.0, recall - FANOUT_PENALTY * len(extra))
    if missing:
        score.notes.append(f"did not call {', '.join(sorted(missing))}")
    if extra:
        score.notes.append(f"fanned out to {', '.join(sorted(extra))}")

    # -- evidence --
    results = state.get("tool_results", [])
    usable = [
        r for r in results if r.status != "error" and r.result and r.tool_name in required
    ]
    starved = {
        r.tool_name
        for r in results
        if r.status == "error" and r.tool_name in required
    }
    score.starved_tools = tuple(sorted(starved))
    score.evidence = min(1.0, len(usable) / len(required)) if required else 0.0
    if starved:
        score.notes.append(
            f"{', '.join(sorted(starved))} was called and had no data for this "
            f"product/window - an artefact coverage gap, not a planning error"
        )
    elif required and not usable:
        score.notes.append("no required tool produced a usable result")

    # -- direction --
    score.direction = _score_direction(question, state)


def _score_direction(question: GoldenQuestion, state: AgentState) -> float:
    """Whether the conclusion points the way the truth points.

    Read from the tool results rather than the prose. Whether the *estimate* has
    the right sign is a fact; whether a sentence about it reads as positive is a
    parse, and grading the parse would measure the scorer's regex.
    """
    if question.expected_direction is None:
        # No direction was fixed by the scenario. Credit a usable result rather
        # than inventing an expectation the ground truth does not contain.
        return 1.0 if state.get("tool_results") else 0.0

    values = _signed_findings(state)
    if not values:
        return 0.0

    if question.expected_direction == "positive":
        return 1.0 if any(v > 0 for v in values) else 0.0
    if question.expected_direction == "negative":
        return 1.0 if any(v < 0 for v in values) else 0.0

    if question.expected_direction == "positive_uplift_negative_value":
        # The trap case, and the only dimension scored from the prose.
        #
        # Whether the ROI figure is poor is the tool's finding, not the agent's -
        # scoring it here would grade Step 7 again. What is genuinely the agent's
        # is the recommendation: uplift is positive, so every easy reading says
        # repeat it, and the right answer is do not. That decision exists nowhere
        # except the conclusion, so that is where it has to be read from.
        uplift_positive = any(v > 0 for v in values)
        roi = _roi(state)
        evidence_present = roi is not None and roi < 1.0
        if not (uplift_positive and evidence_present):
            return 0.0
        return 1.0 if _advises_against(state) else 0.5

    return 0.0


#: Phrases that decline to repeat something. Matched on the recommended action,
#: which is short and declarative - the same match over a whole summary would
#: fire on any sentence that discussed a risk.
_AGAINST = (
    "do not",
    "don't",
    "should not",
    "avoid",
    "discontinue",
    "stop ",
    "not worth",
    "not repeat",
    "against repeating",
    "reduce the discount",
    "shallower discount",
)


def _advises_against(state: AgentState) -> bool:
    """Whether the recommendation actually declines to repeat the promotion.

    A keyword match, with the weakness that implies: a conclusion that argues
    against repetition in words this list does not contain scores as though it
    argued for it. Tolerated because the alternative - a model grading a model -
    makes the evaluation depend on the thing being evaluated, and a scorer whose
    own errors are correlated with the system's is worse than a blunt one.
    """
    recommendation = state.get("final_recommendation")
    if recommendation is None:
        return False
    action = recommendation.recommended_action.lower()
    return any(phrase in action for phrase in _AGAINST)


def _signed_findings(state: AgentState) -> list[float]:
    """The headline signed quantity from each result, where one exists.

    **Volume before profit, deliberately.** The scenario record fixes the sign of
    the *demand* effect - `expected_log_lift` is a volume lift - and says nothing
    about profit. Step 7 measured why that distinction matters here: promotional
    spend in the generated data runs ~20x the achievable margin at product-store
    grain, so incremental profit is negative for almost every event including the
    ones injected as successful. Scoring direction on profit would grade that
    known data artefact and report it as the agent reasoning backwards.
    """
    keys = (
        "incremental_units",
        "elasticity",
        "total_predicted_units",
        "profit_impact",
        "expected_incremental_profit",
        "incremental_profit",
    )
    found: list[float] = []
    for result in state.get("tool_results", []):
        payload = result.result or {}
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                found.append(float(value))
                break
    return found


def _roi(state: AgentState) -> float | None:
    for result in state.get("tool_results", []):
        value = (result.result or {}).get("roi")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _score_abstention(
    question: GoldenQuestion,
    state: AgentState,
    called: tuple[str, ...],
    score: QuestionScore,
    recommendation: Any,
) -> None:
    """Grade a question the registered tools cannot answer.

    Full marks for declining. Partial marks for hedging - a low-confidence
    answer that names its own limits is worse than an abstention but far better
    than a confident one. Zero for a confident conclusion, which is the failure
    this dimension exists to catch.
    """
    score.applicable = ("abstention",)

    if recommendation is None:
        score.abstention = 1.0
        score.notes.append("declined to conclude")
        return

    confidence = float(recommendation.confidence)
    hedged = bool(recommendation.risks) or recommendation.requires_human_approval

    if confidence >= CONFIDENT:
        score.abstention = 0.0
        score.notes.append(
            f"answered at confidence {confidence:.2f} on a question the "
            f"registered tools cannot establish"
        )
    elif hedged:
        score.abstention = 0.75
        score.notes.append(f"hedged at confidence {confidence:.2f} with stated risks")
    else:
        score.abstention = 0.4
        score.notes.append(f"low confidence ({confidence:.2f}) but no stated limits")


@dataclass
class RunScore:
    """The whole set, aggregated."""

    scores: list[QuestionScore] = field(default_factory=list)
    provider: str = "unknown"

    @property
    def answerable(self) -> list[QuestionScore]:
        return [s for s in self.scores if s.answerable]

    @property
    def unanswerable(self) -> list[QuestionScore]:
        return [s for s in self.scores if not s.answerable]

    def dimension_mean(self, name: str) -> float:
        applicable = [s for s in self.scores if name in s.applicable]
        if not applicable:
            return 0.0
        return sum(getattr(s, name) for s in applicable) / len(applicable)

    @property
    def answerable_mean(self) -> float:
        scored = self.answerable
        return sum(s.overall for s in scored) / len(scored) if scored else 0.0

    @property
    def abstention_mean(self) -> float:
        scored = self.unanswerable
        return sum(s.abstention for s in scored) / len(scored) if scored else 0.0

    def as_dict(self) -> dict[str, Any]:
        """The committed shape. Two headline numbers, never one.

        Collapsing capability and abstention into a single figure would let a
        system that answers everything confidently trade a coverage failure
        against a reasoning success, which is exactly the trade nobody wants it
        making.
        """
        return {
            "provider": self.provider,
            "questions": len(self.scores),
            "answerable_mean": round(self.answerable_mean, 4),
            "abstention_mean": round(self.abstention_mean, 4),
            "artefact_gaps": self.artefact_gaps(),
            "dimensions": {
                "tool_selection": round(self.dimension_mean("tool_selection"), 4),
                "evidence": round(self.dimension_mean("evidence"), 4),
                "direction": round(self.dimension_mean("direction"), 4),
                "abstention": round(self.dimension_mean("abstention"), 4),
            },
            "by_label": self._by_label(),
            "questions_detail": [s.as_dict() for s in self.scores],
        }

    def artefact_gaps(self) -> dict[str, int]:
        """Required tools that ran and found no data, counted by tool.

        Reported next to the score because it changes what the score means. An
        `evidence` figure held down by missing trained series is a coverage
        problem with a coverage fix - retrain over the products the questions
        ask about - and reading it as poor reasoning would send the effort
        somewhere it cannot help.
        """
        gaps: dict[str, int] = {}
        for score in self.scores:
            for tool in score.starved_tools:
                gaps[tool] = gaps.get(tool, 0) + 1
        return dict(sorted(gaps.items()))

    def _by_label(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for score in self.scores:
            grouped.setdefault(score.label, []).append(score.overall)
        return {
            label: round(sum(values) / len(values), 4)
            for label, values in sorted(grouped.items())
        }


def score_run(
    results: list[tuple[GoldenQuestion, AgentState]], *, provider: str = "unknown"
) -> RunScore:
    """Score every question in a run."""
    return RunScore(
        scores=[score_question(q, s) for q, s in results], provider=provider
    )


__all__ = [
    "CONFIDENT",
    "FANOUT_PENALTY",
    "QuestionScore",
    "RunScore",
    "score_question",
    "score_run",
]
