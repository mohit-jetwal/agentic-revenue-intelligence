"""Claude provider.

Four decisions worth defending.

**Structured output uses a single forced tool, not JSON-in-a-prompt.** The
target Pydantic model's JSON schema becomes a tool's input schema and
``tool_choice`` forces it. The API then *constrains* the output shape rather
than being asked politely for it, which is the difference between a parse error
once a week and a parse error never. Asking for JSON in the prompt and calling
``json.loads`` works until the model prefixes "Here is the JSON:" and it does not
fail loudly - it fails as a malformed plan propagating into the graph as a string
nobody checked.

**The system prompt and tool specs are cached.** They are large, stable, and
re-sent on every turn of a long investigation - exactly the shape prompt caching
exists for. A twelve-step investigation re-sends the same 3,000-token tool
manifest twelve times without it.

**Retries live on the SDK client, not here.** The SDK already implements
exponential backoff with jitter and respects ``retry-after``. Hand-rolling that
produces a worse version that ignores the header the server sent.

**Message content is never logged.** Token counts, model, stop reason and prompt
version are. Business data must not leak into logs, and a log line is the easiest
place for it to escape.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config.settings import LLMSettings
from app.llm.base import (
    LLMNotConfiguredError,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    Message,
    TokenUsage,
    ToolCall,
)
from app.observability.logging import get_logger
from app.schemas.tool_contract import ToolSpec

logger = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: Name of the synthetic tool used to force structured output.
_STRUCTURED_TOOL = "emit_structured_response"


class ClaudeProvider(LLMProvider):
    """Anthropic Claude backend."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        # Created lazily so importing this module - and constructing the
        # container - never requires a key.
        self._client: Any | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def planner_model_name(self) -> str:
        """Model used for planning and re-planning, where reasoning matters most."""
        return self._settings.planner_model

    # -- client -------------------------------------------------------------

    def _require_key(self) -> str:
        secret = self._settings.api_key
        key = secret.get_secret_value() if secret is not None else ""
        if not key:
            raise LLMNotConfiguredError(
                "No Claude API key configured. Set LLM__API_KEY in .env "
                "(see .env.example), or set LLM__PROVIDER=stub to run offline."
            )
        return key

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise LLMNotConfiguredError(
                    "the `anthropic` package is not installed"
                ) from exc

            self._client = Anthropic(
                api_key=self._require_key(),
                timeout=float(self._settings.timeout_seconds),
                max_retries=self._settings.max_retries,
            )
        return self._client

    # -- completions --------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-form text. Narrative synthesis only - never for a number."""
        response = self._call(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._to_response(response)

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

        The workhorse for planning, intent classification and critique - anywhere
        downstream code branches on the answer. Returning a validated object
        rather than text is what stops a malformed plan from propagating into
        the graph as a string nobody checked.
        """
        schema = _tool_schema(response_model)
        raw = self._call(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[
                {
                    "name": _STRUCTURED_TOOL,
                    "description": (
                        f"Emit the response as a {response_model.__name__}. "
                        f"This is the only valid way to answer."
                    ),
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL},
        )
        response = self._to_response(raw)

        payload = next(
            (call.arguments for call in response.tool_calls if call.name == _STRUCTURED_TOOL),
            None,
        )
        if payload is None:
            raise LLMResponseError(
                f"the model returned no {_STRUCTURED_TOOL} call despite "
                f"tool_choice forcing it; stop_reason={response.stop_reason}"
            )

        try:
            return response_model.model_validate(payload), response
        except ValidationError as exc:
            # The schema constrained the shape and the content still failed
            # validation - a constraint the JSON schema cannot express, such as
            # a bounded float or a cross-field rule.
            raise LLMResponseError(
                f"the model's {response_model.__name__} failed validation: {exc}"
            ) from exc

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

        The provider returns the requested calls and **never executes them**.
        Execution stays with the caller so permission checks and budget
        accounting happen in one auditable place.
        """
        raw = self._call(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[_to_anthropic_tool(spec) for spec in tools],
        )
        return self._to_response(raw)

    # -- internals ----------------------------------------------------------

    def _call(
        self,
        messages: list[Message],
        *,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": max_tokens or self._settings.max_tokens,
            "temperature": (
                temperature if temperature is not None else self._settings.temperature
            ),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        if system:
            # Cached: the system prompt is large, stable, and re-sent every turn.
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            # The last tool carries the cache breakpoint, so the whole manifest
            # above it is cached as one prefix.
            cached = [dict(tool) for tool in tools]
            cached[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = cached
        if tool_choice:
            payload["tool_choice"] = tool_choice

        try:
            return self.client.messages.create(**payload)
        except Exception as exc:
            name = type(exc).__name__
            if "Timeout" in name or "timeout" in str(exc).lower():
                raise LLMTimeoutError(
                    f"Claude request exceeded {self._settings.timeout_seconds}s"
                ) from exc
            raise LLMResponseError(f"Claude request failed: {name}: {exc}") from exc

    def _to_response(self, raw: Any) -> LLMResponse:
        """Normalise the SDK response into the provider-agnostic shape."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in getattr(raw, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        raw_usage = getattr(raw, "usage", None)
        usage = TokenUsage(
            input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0),
        )

        # Token counts, model and stop reason only. Never message content.
        logger.info(
            "llm.response",
            model=str(getattr(raw, "model", self._settings.model)),
            stop_reason=str(getattr(raw, "stop_reason", "")),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            tool_calls=len(tool_calls),
        )

        return LLMResponse(
            text="".join(text_parts),
            usage=usage,
            tool_calls=tool_calls,
            stop_reason=str(getattr(raw, "stop_reason", "") or ""),
            model=str(getattr(raw, "model", self._settings.model)),
        )

    def count_tokens(self, text: str) -> int:
        """Rough estimate, deliberately.

        ~4 characters per token is close enough for budget guarding, which needs
        to catch runaway growth rather than bill anyone. The SDK's counting
        endpoint is a network round trip, and spending one to decide whether to
        spend another is the wrong trade inside a loop.
        """
        return max(1, len(text) // 4)

    def health_check(self) -> tuple[bool, str]:
        """Configuration probe. Makes no network call."""
        if not self._settings.is_configured:
            return False, "LLM__API_KEY not set (use LLM__PROVIDER=stub to run offline)"
        return True, f"claude ({self._settings.model} / planner {self._settings.planner_model})"


def _tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a Pydantic model, flattened for the tools API.

    ``$defs`` and ``$ref`` are inlined because the Anthropic tools API does not
    resolve local references. A nested model would otherwise arrive as a bare
    ``$ref`` the API cannot follow, and the failure looks like the model
    ignoring the schema rather than the schema being unusable.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    return _inline_refs(schema, definitions) if definitions else schema


def _inline_refs(node: Any, definitions: dict[str, Any], depth: int = 0) -> Any:
    """Recursively replace ``$ref`` with the referenced definition."""
    if depth > 12:
        # A self-referential model would recurse forever. Twelve levels is far
        # beyond anything the agent schemas need.
        return node
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = definitions.get(ref.split("/")[-1], {})
            merged = {**_inline_refs(target, definitions, depth + 1)}
            merged.update({k: v for k, v in node.items() if k != "$ref"})
            return merged
        return {k: _inline_refs(v, definitions, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, definitions, depth + 1) for item in node]
    return node


def _to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    """Translate the platform's tool spec into the API's shape."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


__all__ = ["ClaudeProvider"]
