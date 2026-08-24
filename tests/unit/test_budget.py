"""Execution budget enforcement.

Each limit is tested independently because they catch different failure shapes:
a planning loop that calls no tools, a plan that fans out too wide, a slow tool
the counters never reach, and unbounded context growth.
"""

from __future__ import annotations

import pytest

from app.config.settings import AgentSettings
from app.guardrails.budget import BudgetExceededError, BudgetTracker

pytestmark = pytest.mark.unit


def _tracker(**overrides: float) -> BudgetTracker:
    defaults: dict[str, float] = {
        "max_iterations": 3,
        "max_tool_calls": 5,
        "max_execution_seconds": 60.0,
        "max_token_budget": 1000,
    }
    defaults.update(overrides)
    return BudgetTracker(
        max_iterations=int(defaults["max_iterations"]),
        max_tool_calls=int(defaults["max_tool_calls"]),
        max_execution_seconds=defaults["max_execution_seconds"],
        max_token_budget=int(defaults["max_token_budget"]),
    )


def test_built_from_settings() -> None:
    tracker = BudgetTracker.from_settings(AgentSettings(_env_file=None))
    assert tracker.max_iterations > 0
    assert tracker.tool_calls == 0


def test_within_budget_does_not_raise() -> None:
    tracker = _tracker()
    for _ in range(3):
        tracker.record_iteration()
    assert tracker.iterations == 3


def test_iteration_limit_trips() -> None:
    """Catches a planning loop that never calls a tool."""
    tracker = _tracker(max_iterations=2)
    tracker.record_iteration()
    tracker.record_iteration()
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_iteration()
    assert exc_info.value.limit_name == "max_iterations"


def test_tool_call_limit_trips() -> None:
    """Catches a plan that fans out too wide."""
    tracker = _tracker(max_tool_calls=2)
    tracker.record_tool_call()
    tracker.record_tool_call()
    with pytest.raises(BudgetExceededError, match="max_tool_calls"):
        tracker.record_tool_call()


def test_tool_call_limit_trips_on_batch() -> None:
    tracker = _tracker(max_tool_calls=3)
    with pytest.raises(BudgetExceededError, match="max_tool_calls"):
        tracker.record_tool_call(count=4)


def test_token_limit_trips() -> None:
    """Catches unbounded context growth - the limit that costs money."""
    tracker = _tracker(max_token_budget=100)
    tracker.record_tokens(90)
    with pytest.raises(BudgetExceededError, match="max_token_budget"):
        tracker.record_tokens(20)


def test_time_limit_trips() -> None:
    """Catches a slow tool the counters would never reach."""
    tracker = _tracker(max_execution_seconds=0.0)
    with pytest.raises(BudgetExceededError, match="max_execution_seconds"):
        tracker.check()


def test_time_is_checked_before_counters() -> None:
    """Time is the limit most likely to breach with no counter moving."""
    tracker = _tracker(max_execution_seconds=0.0, max_iterations=0)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check()
    assert exc_info.value.limit_name == "max_execution_seconds"


def test_exception_reports_limit_and_observed_value() -> None:
    tracker = _tracker(max_tool_calls=1)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_tool_call(count=5)

    error = exc_info.value
    assert error.limit == 1
    assert error.observed == 5
    assert "max_tool_calls" in str(error)


def test_would_exceed_does_not_consume() -> None:
    """Lets the Supervisor decline a step it cannot afford to finish."""
    tracker = _tracker(max_tool_calls=2)
    tracker.record_tool_call()

    assert tracker.would_exceed(tool_calls=2) is True
    assert tracker.would_exceed(tool_calls=1) is False
    assert tracker.tool_calls == 1  # unchanged


def test_remaining_reports_headroom() -> None:
    tracker = _tracker(max_tool_calls=5, max_token_budget=1000)
    tracker.record_tool_call(count=2)
    tracker.record_tokens(400)

    remaining = tracker.remaining()
    assert remaining["tool_calls"] == 3
    assert remaining["tokens"] == 600


def test_remaining_never_goes_negative() -> None:
    tracker = _tracker(max_tool_calls=1)
    tracker.tool_calls = 9  # simulate an over-run recorded elsewhere
    assert tracker.remaining()["tool_calls"] == 0
