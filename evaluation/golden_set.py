"""The golden set: questions whose correct answer is already known.

**Why these questions and not others.** Step 2 injected scenarios A-J into the
synthetic data and recorded exactly which products, stores and windows each one
touched. That record is the only place in this system where the right answer
exists independently of the thing being graded, so the golden set is *derived*
from it rather than written by hand. A question invented separately would be
graded against an expectation invented separately, which measures nothing.

**Not every scenario is answerable, and that is recorded rather than hidden.**
Six tools are registered, and they estimate uplift, elasticity, forecasts,
allocations, prices and scenarios. None of them diagnoses a stockout, a
competitor price cut or a lost-distribution shock. For those scenarios the
correct behaviour is to *decline* - to say the available evidence cannot
establish the cause - and confabulating a confident answer is the failure.

So questions carry an ``answerable`` flag and are scored on different things:

*Answerable* — did it select the right tools, gather evidence, and get the
direction right?

*Unanswerable* — did it abstain? A system that answers everything confidently is
worse than one that knows its own coverage, because the second can be trusted
about the first.

**Entities are carried, not parsed.** Each question names the product and window
its scenario actually used, so a scored run is checking whether the agent looked
where the effect was planted rather than whether it guessed an id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.schemas.domain import IntentType

#: Where Step 2 recorded what it injected.
DEFAULT_SCENARIO_CONFIG = Path("data/local/ground_truth/scenario_config.json")


@dataclass(frozen=True)
class GoldenQuestion:
    """One graded question, derived from an injected scenario."""

    question_id: str
    #: The scenario this was built from, so a failure is traceable to the truth.
    scenario_id: str
    label: str
    question: str

    expected_intent: IntentType
    #: Tools the answer cannot be established without. Missing one is a failure
    #: of the plan, not of the prose.
    required_tools: frozenset[str] = frozenset()
    #: Tools whose use signals fan-out rather than judgement. Calling one is not
    #: fatal but costs points: the property being protected is the *minimum
    #: sufficient* workflow, and a graph that calls everything looks impressive
    #: in a demo and is wrong.
    discouraged_tools: frozenset[str] = frozenset()

    #: What the truth says. `None` where the scenario fixes no direction.
    expected_direction: str | None = None
    #: Whether the registered tools can establish the answer at all. False makes
    #: abstention the correct behaviour and confidence the failure.
    answerable: bool = True
    #: Why not, stated so the gap is legible rather than implied by a flag.
    coverage_gap: str | None = None

    product_ids: tuple[str, ...] = ()
    store_ids: tuple[str, ...] = ()
    region: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    #: Free-text notes carried into the report for a human reader.
    notes: str = ""

    def entities(self) -> dict[str, list[str]]:
        """The entities a correct plan should scope its tool calls to."""
        found: dict[str, list[str]] = {}
        if self.product_ids:
            found["products"] = list(self.product_ids)
        if self.store_ids:
            found["stores"] = list(self.store_ids)
        if self.region:
            found["regions"] = [self.region]
        return found


# --------------------------------------------------------------------------
# Scenario label -> question shape
# --------------------------------------------------------------------------
#
# One entry per injected label. Keeping this as data rather than a chain of
# conditionals makes the coverage gaps countable: anything mapped to
# `answerable=False` is a capability the platform does not have yet, and the
# report says how many there are.


@dataclass(frozen=True)
class _QuestionShape:
    template: str
    intent: IntentType
    required_tools: frozenset[str] = frozenset()
    discouraged_tools: frozenset[str] = frozenset()
    direction: str | None = None
    answerable: bool = True
    coverage_gap: str | None = None
    notes: str = ""


_SHAPES: dict[str, _QuestionShape] = {
    "successful_promo": _QuestionShape(
        template=(
            "Did the promotion on product {product} between {start} and {end} "
            "generate incremental profit, and was it worth running?"
        ),
        intent=IntentType.PROMOTION_DECISION,
        required_tools=frozenset({"estimate_promo_uplift"}),
        discouraged_tools=frozenset({"forecast_demand", "allocate_promotion_budget"}),
        direction="positive",
        notes="Shallow discount; incremental margin should exceed spend.",
    ),
    "bad_promo": _QuestionShape(
        template=(
            "Product {product} was promoted at a deep discount between {start} "
            "and {end}. Sales rose. Should we run it again?"
        ),
        intent=IntentType.PROMOTION_DECISION,
        required_tools=frozenset({"estimate_promo_uplift"}),
        discouraged_tools=frozenset({"forecast_demand"}),
        # The trap: uplift is genuinely positive while the promotion destroys
        # value. An answer that stops at "sales rose" has missed the question.
        direction="positive_uplift_negative_value",
        notes=(
            "Uplift is positive and ROI is poor. Reading the uplift alone gives "
            "the wrong recommendation."
        ),
    ),
    "price_increase": _QuestionShape(
        template=(
            "We raised the price of product {product} on {start}. How did demand "
            "respond, and was the increase the right call?"
        ),
        intent=IntentType.PRICE_DECISION,
        required_tools=frozenset({"estimate_price_elasticity"}),
        discouraged_tools=frozenset({"allocate_promotion_budget"}),
        direction="negative",
        notes="Own demand falls; the revenue direction depends on elasticity.",
    ),
    "stockout": _QuestionShape(
        template=(
            "Sales of product {product} fell sharply between {start} and {end}. "
            "What caused the decline?"
        ),
        intent=IntentType.ROOT_CAUSE,
        answerable=False,
        coverage_gap=(
            "No availability or distribution diagnostic is registered. The "
            "decline is a supply constraint with latent demand unchanged, and "
            "no combination of the six analytical tools can separate that from "
            "a demand fall."
        ),
        notes="Attributing this to demand would be wrong in a costly direction.",
    ),
    "competitor_price_cut": _QuestionShape(
        template=(
            "Demand for product {product} fell from {start} while our own price "
            "was unchanged. Why?"
        ),
        intent=IntentType.ROOT_CAUSE,
        answerable=False,
        coverage_gap=(
            "Competitor pricing is in the data but no tool exposes it. "
            "Own-price elasticity cannot explain a decline at an unchanged own "
            "price, so the tools available can only rule out the wrong cause."
        ),
    ),
    "regional_shock": _QuestionShape(
        template=(
            "{region} region sales dropped between {start} and {end}. What "
            "happened, and what should we do about it?"
        ),
        intent=IntentType.ROOT_CAUSE,
        answerable=False,
        coverage_gap=(
            "The decline is mostly lost distribution rather than softer demand. "
            "Separating the two needs a listing-count diagnostic that is not "
            "registered as a tool."
        ),
    ),
    "seasonal_peak": _QuestionShape(
        template=(
            "What demand should we plan for over the festival period beginning "
            "{start}?"
        ),
        intent=IntentType.FORECAST,
        required_tools=frozenset({"forecast_demand"}),
        discouraged_tools=frozenset(
            {"estimate_promo_uplift", "allocate_promotion_budget", "optimize_price"}
        ),
        direction="positive",
        notes="The minimum-sufficient case: one tool answers it, and it should stop.",
    ),
    "product_launch": _QuestionShape(
        template=(
            "Product {product} launched partway through the history. What should "
            "we forecast for it?"
        ),
        intent=IntentType.FORECAST,
        required_tools=frozenset({"forecast_demand"}),
        discouraged_tools=frozenset({"estimate_price_elasticity"}),
        notes=(
            "Short history with a distribution ramp. A forecast that treats "
            "pre-launch zeros as weak demand is wrong."
        ),
    ),
}


def _format(shape: _QuestionShape, scenario: dict[str, Any]) -> str:
    products = scenario.get("product_ids") or []
    return shape.template.format(
        product=products[0] if products else "the affected products",
        region=scenario.get("region") or "The affected",
        start=scenario.get("start_date") or "the start of the window",
        end=scenario.get("end_date") or "the end of the window",
    )


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value))


def build_golden_set(scenarios: list[dict[str, Any]]) -> list[GoldenQuestion]:
    """Turn injected scenarios into graded questions.

    Scenarios with no mapped shape are skipped rather than guessed at - an
    unmapped label means nobody decided what the right answer looks like, and
    grading against an undecided expectation is worse than not grading.
    """
    questions: list[GoldenQuestion] = []

    for scenario in scenarios:
        shape = _SHAPES.get(str(scenario.get("label")))
        if shape is None:
            continue

        scenario_id = str(scenario["scenario_id"])
        questions.append(
            GoldenQuestion(
                question_id=f"Q-{scenario_id}",
                scenario_id=scenario_id,
                label=str(scenario["label"]),
                question=_format(shape, scenario),
                expected_intent=shape.intent,
                required_tools=shape.required_tools,
                discouraged_tools=shape.discouraged_tools,
                expected_direction=shape.direction,
                answerable=shape.answerable,
                coverage_gap=shape.coverage_gap,
                product_ids=tuple(scenario.get("product_ids") or []),
                # Truncated: the regional shock lists hundreds, and carrying them
                # all into a question would bloat every report that prints it.
                store_ids=tuple((scenario.get("store_ids") or [])[:5]),
                region=scenario.get("region"),
                start_date=_parse_date(scenario.get("start_date")),
                end_date=_parse_date(scenario.get("end_date")),
                notes=shape.notes,
            )
        )

    return questions


@lru_cache(maxsize=4)
def load_golden_set(path: Path | None = None) -> tuple[GoldenQuestion, ...]:
    """Load the golden set from Step 2's scenario record.

    Cached because the evaluation, its tests and the CLI all ask for it, and it
    is a pure function of a file that does not change during a run.
    """
    source = path or DEFAULT_SCENARIO_CONFIG
    if not source.exists():
        raise FileNotFoundError(
            f"scenario config not found at {source}. Generate the dataset first: "
            f"uv run python scripts/generate_data.py"
        )

    payload = json.loads(source.read_text(encoding="utf-8"))
    return tuple(build_golden_set(payload.get("scenarios", [])))


def coverage_summary(questions: tuple[GoldenQuestion, ...]) -> dict[str, Any]:
    """What the golden set can and cannot grade.

    Reported alongside every score so a headline number is never read without
    the denominator it was computed over.
    """
    answerable = [q for q in questions if q.answerable]
    gaps: dict[str, int] = {}
    for question in questions:
        if not question.answerable:
            gaps[question.label] = gaps.get(question.label, 0) + 1

    return {
        "questions": len(questions),
        "answerable": len(answerable),
        "abstention_expected": len(questions) - len(answerable),
        "labels": sorted({q.label for q in questions}),
        "coverage_gaps": gaps,
    }


__all__ = [
    "DEFAULT_SCENARIO_CONFIG",
    "GoldenQuestion",
    "build_golden_set",
    "coverage_summary",
    "load_golden_set",
]
