"""Deterministic offline LLM provider.

Exists so the agent layer is testable. Every agent test, the golden-set
evaluation and CI run against this: no API key, no network, no cost, and the
same answer every time.

That is not a convenience. An agent suite that costs money per run gets run less
often, and a non-deterministic one produces failures nobody can reproduce - which
is precisely how a re-planning bug survives to production.

**Two modes.**

*Scripted.* A test registers the exact object a call should return, keyed by the
response model and optionally by a substring of the last user message. This is
how a test says "when the Supervisor plans a forecast question, return this
plan" and then asserts on what the graph did with it.

*Synthesised.* With nothing registered, a minimal valid instance is constructed
from the model's JSON schema. Enough for the graph to run end to end, and
deliberately bland - a synthesised plan has no steps, so a test that forgot to
register one fails on an empty plan rather than passing on a plausible guess.

**What it does not do** is pretend to reason. It never invents a number, because
returning something numeric-looking would let a test pass while the real system
would refuse.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config.settings import LLMSettings
from app.llm.base import (
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    Message,
    TokenUsage,
    ToolCall,
)
from app.observability.logging import get_logger
from app.schemas.tool_contract import ToolSpec

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass
class ScriptedCall:
    """One registered response."""

    #: Return this when the last user message contains this substring. ``None``
    #: matches any message for the model.
    when_contains: str | None
    value: Any


@dataclass
class StubProvider(LLMProvider):
    """An LLM that returns what it was told to, and nothing else."""

    settings: LLMSettings | None = None
    #: response model name -> registered calls, in registration order.
    _scripted: dict[str, list[ScriptedCall]] = field(default_factory=dict)
    _text: list[ScriptedCall] = field(default_factory=list)
    _tool_calls: list[ScriptedCall] = field(default_factory=list)
    #: Every call made, for tests that assert on how the agent used the model.
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def planner_model_name(self) -> str:
        return "stub"

    # -- scripting ----------------------------------------------------------

    def script_structured(
        self, value: BaseModel, *, when_contains: str | None = None
    ) -> StubProvider:
        """Return ``value`` from ``complete_structured`` for its own type."""
        key = type(value).__name__
        self._scripted.setdefault(key, []).append(ScriptedCall(when_contains, value))
        return self

    def script_text(self, text: str, *, when_contains: str | None = None) -> StubProvider:
        self._text.append(ScriptedCall(when_contains, text))
        return self

    def script_tool_calls(
        self, tool_calls: list[ToolCall], *, when_contains: str | None = None
    ) -> StubProvider:
        self._tool_calls.append(ScriptedCall(when_contains, tool_calls))
        return self

    def reset(self) -> None:
        self._scripted.clear()
        self._text.clear()
        self._tool_calls.clear()
        self.calls.clear()

    # -- provider surface ---------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self._record("complete", messages, system)
        text = self._match(self._text, messages)
        return self._response(text if text is not None else "")

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[TModel],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[TModel, LLMResponse]:
        self._record("complete_structured", messages, system, model=response_model.__name__)

        scripted = self._match(self._scripted.get(response_model.__name__, []), messages)
        if scripted is not None:
            if not isinstance(scripted, response_model):
                raise LLMResponseError(
                    f"scripted value is a {type(scripted).__name__}, not a "
                    f"{response_model.__name__}"
                )
            return scripted, self._response("", tool_name="emit_structured_response")

        synthesised = _synthesise(response_model)
        logger.info("stub.synthesised", model=response_model.__name__)
        return synthesised, self._response("", tool_name="emit_structured_response")

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self._record(
            "complete_with_tools", messages, system, tools=[t.name for t in tools]
        )
        scripted = self._match(self._tool_calls, messages)
        return self._response("", tool_calls=scripted or [])

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def health_check(self) -> tuple[bool, str]:
        return True, "stub (deterministic, offline)"

    # -- internals ----------------------------------------------------------

    def _record(
        self, method: str, messages: list[Message], system: str | None, **extra: Any
    ) -> None:
        self.calls.append(
            {
                "method": method,
                "messages": len(messages),
                "last_user": _last_user(messages),
                "has_system": system is not None,
                **extra,
            }
        )

    @staticmethod
    def _match(scripted: list[ScriptedCall], messages: list[Message]) -> Any:
        """First registration whose filter matches. Specific before general.

        Sorting so that ``when_contains`` entries are tried before catch-alls
        lets a test register a default and then override it for one question,
        which is the natural way to write a suite.
        """
        if not scripted:
            return None
        last = _last_user(messages).lower()
        ordered = sorted(scripted, key=lambda c: c.when_contains is None)
        for call in ordered:
            if call.when_contains is None or call.when_contains.lower() in last:
                return call.value
        return None

    def _response(
        self,
        text: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_name: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text=text,
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            tool_calls=tool_calls or [],
            stop_reason="tool_use" if tool_name or tool_calls else "end_turn",
            model="stub",
        )


def _last_user(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _synthesise[T: BaseModel](model: type[T]) -> T:
    """Build a minimal valid instance from the model's field defaults.

    Deliberately bland. A synthesised plan has no steps and a synthesised
    critique is not valid, so a test that forgot to register a response fails on
    an obviously empty object rather than passing on a plausible-looking guess.
    """
    values: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if not info.is_required():
            continue
        values[name] = _default_for(info.annotation)
    return model(**values)


def _default_for(annotation: Any) -> Any:
    """A minimal value for a required field of the given type."""
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}

    # Optional[X] and unions: None is valid if permitted, else use the first arm.
    args = getattr(annotation, "__args__", ())
    if args:
        if type(None) in args:
            return None
        return _default_for(args[0])

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _synthesise(annotation)
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 0
        if issubclass(annotation, float):
            return 0.0
        if issubclass(annotation, str):
            # StrEnum members are strings; take the first so the value is valid.
            members = getattr(annotation, "__members__", None)
            if members:
                return next(iter(members.values()))
            return ""
    return None


def build_stub_provider(
    settings: LLMSettings | None = None,
    *,
    configure: Callable[[StubProvider], None] | None = None,
) -> StubProvider:
    """Construct a stub, optionally pre-scripted."""
    provider = StubProvider(settings=settings)
    if configure:
        configure(provider)
    return provider


__all__ = ["ScriptedCall", "StubProvider", "build_stub_provider"]
