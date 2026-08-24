"""LLM provider abstraction.

The platform is built on Claude, but the reasoning layer is isolated behind this
interface so that model choice is a configuration decision rather than an
architectural one. That isolation earns its keep in two ordinary situations:
swapping planner and worker models independently, and running the evaluation
suite against a recorded/stub provider with no network calls.

What this interface deliberately does *not* offer is a "just give me text"
escape hatch for analytical work. Numbers come from tools. The provider surface
is limited to producing text, producing validated structured output, and
selecting tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.schemas.tool_contract import ToolSpec

TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMNotConfiguredError(LLMError):
    """Raised when no API key is available."""


class LLMTimeoutError(LLMError):
    """Raised when the provider exceeded its configured timeout."""


class LLMResponseError(LLMError):
    """Raised when the response could not be parsed into the requested shape."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked to invoke."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Uniform provider response."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    #: Identifier of the prompt version used, for reproducibility.
    prompt_version: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Message(BaseModel):
    """A single conversational turn."""

    role: str
    content: str


class LLMProvider(ABC):
    """Interface every LLM backend must satisfy."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model in use. Recorded in traces."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-form text completion. Used for narrative synthesis only."""

    @abstractmethod
    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[TModel],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[TModel, LLMResponse]:
        """Completion validated against a Pydantic model.

        This is the workhorse for planning, intent classification and critique -
        anywhere downstream code branches on the model's answer. Returning a
        validated object rather than text is what stops a malformed plan from
        propagating into the graph as a string that nobody checked.
        """

    @abstractmethod
    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Completion where the model may request tool invocations.

        The provider returns the requested calls; it never executes them.
        Execution stays with the caller so permission checks and budget
        accounting happen in one auditable place.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Approximate token count, used for budget accounting."""

    def health_check(self) -> tuple[bool, str]:
        """Cheap configuration probe. Must not make a network call."""
        return True, f"{type(self).__name__} ({self.model_name})"
