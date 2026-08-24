"""Execution budget enforcement (section 18 of the brief).

An agent that re-plans is an agent that can loop forever. The failure is not
hypothetical: a Supervisor whose Critic keeps returning "insufficient evidence"
will re-plan, re-run tools, and burn tokens until something external stops it.

Four independent limits, because they fail differently:

* ``max_iterations``  - catches a planning loop that makes no tool calls at all.
* ``max_tool_calls``  - catches a plan that fans out too wide.
* ``max_execution_seconds`` - catches a slow tool the other limits never reach.
* ``max_token_budget`` - catches context growth, the one that costs money.

Exceeding a budget is not an error to hide. The Supervisor should catch
:class:`BudgetExceededError`, stop investigating, and return the best recommendation
supported by the evidence gathered so far, explicitly flagged as incomplete.
A truthful partial answer beats an infinite loop, and beats a confident answer
built on an investigation that never finished.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config.settings import AgentSettings


class BudgetExceededError(RuntimeError):
    """Raised when an execution limit is reached."""

    def __init__(self, limit_name: str, limit: float, observed: float) -> None:
        super().__init__(
            f"Agent budget exceeded: {limit_name} limit is {limit}, reached {observed}."
        )
        self.limit_name = limit_name
        self.limit = limit
        self.observed = observed


@dataclass
class BudgetTracker:
    """Tracks and enforces one investigation's resource budget."""

    max_iterations: int
    max_tool_calls: int
    max_execution_seconds: float
    max_token_budget: int

    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls, settings: AgentSettings) -> BudgetTracker:
        return cls(
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            max_execution_seconds=settings.max_execution_seconds,
            max_token_budget=settings.max_token_budget,
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    # -- consumption --------------------------------------------------------

    def record_iteration(self) -> None:
        self.iterations += 1
        self.check()

    def record_tool_call(self, count: int = 1) -> None:
        self.tool_calls += count
        self.check()

    def record_tokens(self, tokens: int) -> None:
        self.tokens_used += tokens
        self.check()

    # -- enforcement --------------------------------------------------------

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if any limit has been reached.

        Time is checked first: it is the limit most likely to be breached
        without any counter moving.
        """
        if self.elapsed_seconds > self.max_execution_seconds:
            raise BudgetExceededError(
                "max_execution_seconds", self.max_execution_seconds, round(self.elapsed_seconds, 2)
            )
        if self.iterations > self.max_iterations:
            raise BudgetExceededError("max_iterations", self.max_iterations, self.iterations)
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceededError("max_tool_calls", self.max_tool_calls, self.tool_calls)
        if self.tokens_used > self.max_token_budget:
            raise BudgetExceededError("max_token_budget", self.max_token_budget, self.tokens_used)

    def would_exceed(self, *, tool_calls: int = 0, tokens: int = 0) -> bool:
        """Check a prospective spend without consuming it.

        Lets the Supervisor decline to start a step it cannot afford to finish,
        rather than aborting halfway through and wasting what it already spent.
        """
        return (
            self.elapsed_seconds > self.max_execution_seconds
            or self.tool_calls + tool_calls > self.max_tool_calls
            or self.tokens_used + tokens > self.max_token_budget
        )

    def remaining(self) -> dict[str, float]:
        """Headroom on each limit. Surfaced in the trace for debugging."""
        return {
            "iterations": max(0, self.max_iterations - self.iterations),
            "tool_calls": max(0, self.max_tool_calls - self.tool_calls),
            "seconds": max(0.0, round(self.max_execution_seconds - self.elapsed_seconds, 2)),
            "tokens": max(0, self.max_token_budget - self.tokens_used),
        }
