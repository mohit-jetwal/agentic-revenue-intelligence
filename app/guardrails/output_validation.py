"""Checking that every number in a recommendation came from a tool.

This is the hallucination control that is **architectural rather than
prompted**. The system prompt tells the model never to state a number that did
not come from a tool result; this verifies it did not, and a prompt instruction
without a check is a hope.

**How it works.** Every numeral in the generated text is extracted, and each is
looked for in the tool results the investigation actually collected — at the
precision it was written, and at plausible roundings of the underlying value. A
number that appears nowhere is *unsourced*, and the recommendation carries a
warning naming it.

**What it deliberately does not do is block.** Three reasons.

*False positives are certain.* A recommendation legitimately contains numbers
that are not tool outputs: a year, a horizon in days, a percentage the user
supplied, "the top 3 products". Blocking on those would make the check
unusable, and an unusable check gets switched off.

*The arithmetic is real.* "150 incremental units at a 40 margin is 6,000 of
profit" involves a number no tool returned. Recomputation from sourced values is
legitimate, and distinguishing it from invention needs the reasoning the check
does not have.

*A labelled number is more useful than a missing one.* Flagging "this figure
does not appear in any tool result" tells a reviewer exactly where to look.
Suppressing the sentence tells them nothing.

So it reports, and the Critic decides. What it catches is the failure that
matters: a confident figure that exists nowhere in the evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.observability.logging import get_logger
from app.schemas.tool_contract import ToolResult

logger = get_logger(__name__)

#: Numerals in prose, with an optional magnitude suffix.
#:
#: The suffix group is load-bearing. A recommendation writes 1,427,355 as
#: "1.43M", and without capturing the M the numeral is 1.43 - which falls under
#: the structural floor and is skipped, so the commonest way of writing a large
#: figure would never be checked at all.
_NUMBER = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(bn|[KkMmBb])?\b")

#: Suffix to multiplier. Case matters only for disambiguation, not meaning.
_MAGNITUDE: dict[str, float] = {
    "k": 1e3,
    "K": 1e3,
    "m": 1e6,
    "M": 1e6,
    "b": 1e9,
    "B": 1e9,
    "bn": 1e9,
}

#: Numbers below this are almost always structural rather than measured - a
#: step count, a year, "the top 3". Checking them produces noise that buries
#: the real finding.
_IGNORE_BELOW = 100.0

#: Years appear constantly in date ranges and are never tool outputs.
_YEAR_RANGE = (1990.0, 2100.0)

#: Relative tolerance when matching a quoted figure to a source value. Covers
#: the rounding a readable sentence applies: 1,427,355 written as 1.43M, or
#: 0.6761 as 68%.
_TOLERANCE = 0.02


@dataclass
class UnsourcedNumber:
    """A figure in the output that no tool result supports."""

    value: float
    #: The sentence it appeared in, so a reviewer can see the claim.
    context: str

    def describe(self) -> str:
        return f"{self.value:,.4g} in: \"{self.context.strip()[:120]}\""


@dataclass
class ValidationReport:
    """What the check found."""

    checked: int = 0
    sourced: int = 0
    unsourced: list[UnsourcedNumber] = field(default_factory=list)
    #: Numbers skipped as structural - years, small counts.
    skipped: int = 0

    @property
    def clean(self) -> bool:
        return not self.unsourced

    @property
    def sourced_share(self) -> float:
        return self.sourced / self.checked if self.checked else 1.0

    def warnings(self) -> list[str]:
        """Warnings to attach to the recommendation, most specific first."""
        if self.clean:
            return []
        listed = "; ".join(item.describe() for item in self.unsourced[:5])
        more = (
            f" and {len(self.unsourced) - 5} more"
            if len(self.unsourced) > 5
            else ""
        )
        return [
            f"UNSOURCED FIGURES: {len(self.unsourced)} number(s) in this "
            f"recommendation do not appear in any tool result - {listed}{more}. "
            f"They may be arithmetic derived from sourced values, or they may "
            f"be invented. Verify before acting on them."
        ]

    def summary(self) -> str:
        return (
            f"{self.sourced}/{self.checked} figures sourced "
            f"({self.skipped} structural, {len(self.unsourced)} unsourced)"
        )


def collect_source_values(results: list[ToolResult]) -> set[float]:
    """Every number a tool actually returned.

    Walks the whole result payload rather than a known set of fields, because
    the recommendation may legitimately quote any of them - an interval bound, a
    per-event ROI, a segment uplift - and enumerating them would need updating
    every time a tool gains a field.
    """
    values: set[float] = set()
    for result in results:
        _walk(result.result, values)
        _walk(result.model_dump(include={"confidence"}), values)
    return values


def _walk(node: Any, into: set[float], depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(node, bool):
        return
    if isinstance(node, int | float):
        into.add(float(node))
        return
    if isinstance(node, dict):
        for value in node.values():
            _walk(value, into, depth + 1)
    elif isinstance(node, list | tuple):
        for item in node:
            _walk(item, into, depth + 1)


def validate_output(text: str, results: list[ToolResult]) -> ValidationReport:
    """Check every figure in ``text`` against what the tools returned."""
    sources = collect_source_values(results)
    report = ValidationReport()

    for match in _NUMBER.finditer(text):
        raw = match.group(1).rstrip(".").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue

        suffix = match.group(2)
        scaled = value * _MAGNITUDE[suffix] if suffix else value
        percentage = _is_percentage(text, match.end())

        # Neither a suffix nor a percent sign can be structural: "1.43M" and
        # "68%" are both measurements, though 1.43 and 68 on their own fall
        # under the floor. Without these two exemptions the check would skip
        # the two commonest ways a finding is actually written.
        if percentage and _is_confidence_level(text, value, match.start(), match.end()):
            report.skipped += 1
            continue
        if suffix is None and not percentage and _is_structural(value):
            report.skipped += 1
            continue

        report.checked += 1
        if _has_source(scaled, sources, percentage=percentage):
            report.sourced += 1
        else:
            report.unsourced.append(
                UnsourcedNumber(
                    value=scaled, context=_sentence_around(text, match.start())
                )
            )

    logger.info(
        "output_validation.checked",
        checked=report.checked,
        unsourced=len(report.unsourced),
        sources=len(sources),
    )
    return report


def _is_structural(value: float) -> bool:
    """Whether a number is bookkeeping rather than a measurement."""
    magnitude = abs(value)
    if _YEAR_RANGE[0] <= magnitude <= _YEAR_RANGE[1] and float(value).is_integer():
        return True
    return magnitude < _IGNORE_BELOW


def _is_percentage(text: str, position: int) -> bool:
    return text[position : position + 1] == "%"


#: Conventional confidence levels. "95%" in "95% interval" names the level, not
#: a measured quantity, and no tool returns it as a value.
_CONFIDENCE_LEVELS = frozenset({80.0, 90.0, 95.0, 99.0})

#: Words that mark a nearby percentage as a confidence level.
_INTERVAL_WORDS = ("interval", "confidence", "credible", " ci", "ci ")


def _is_confidence_level(text: str, value: float, start: int, end: int) -> bool:
    """Whether a percentage names an interval's level rather than a finding.

    Scoped by both the value and the surrounding words: a genuine finding of
    exactly 95% next to the word "confidence" is possible but far rarer than the
    phrase "95% confidence interval", and requiring both keeps the exemption
    from swallowing ordinary percentages.
    """
    if value not in _CONFIDENCE_LEVELS:
        return False
    window = text[max(0, start - 30) : end + 30].lower()
    return any(word in window for word in _INTERVAL_WORDS)


def _has_source(value: float, sources: set[float], *, percentage: bool) -> bool:
    """Whether a source value matches, allowing for readable rounding.

    A recommendation writes 1,427,355 as "1.43M" and 0.6761 as "68%". Requiring
    an exact match would flag every well-written sentence, so the comparison is
    relative and, for percentages, also checks the fractional form the tools
    actually return.
    """
    candidates = [value, -value]
    if percentage:
        # Tools return 0.68; prose says 68%.
        candidates.extend([value / 100.0, -value / 100.0])

    for candidate in candidates:
        for source in sources:
            if _close(candidate, source):
                return True
            # Prose scales large figures: "1.43M" against a source of 1,427,355.
            for scale in (1e3, 1e6, 1e9):
                if _close(candidate * scale, source):
                    return True
    return False


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    magnitude = max(abs(a), abs(b))
    if magnitude == 0:
        return True
    return abs(a - b) / magnitude <= _TOLERANCE


def _sentence_around(text: str, position: int) -> str:
    """The sentence containing ``position``, for the reviewer's context."""
    start = max(text.rfind(".", 0, position), text.rfind("\n", 0, position)) + 1
    end = text.find(".", position)
    return text[start : end if end != -1 else len(text)]


__all__ = [
    "UnsourcedNumber",
    "ValidationReport",
    "collect_source_values",
    "validate_output",
]
