"""LLM providers: the Claude translation layer and the offline stub.

The Claude tests exercise everything except the network call: schema flattening,
response normalisation, prompt caching, and error mapping. The one thing they do
not do is spend money.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.config.settings import LLMSettings
from app.llm.base import (
    LLMNotConfiguredError,
    LLMRefusalError,
    LLMResponseError,
    LLMTimeoutError,
    Message,
    ToolCall,
)
from app.llm.claude import ClaudeProvider, _inline_refs, _tool_schema
from app.llm.stub import ScriptedCall, StubProvider
from app.schemas.tool_contract import ToolSpec

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class Nested(BaseModel):
    label: str
    weight: float = 1.0


class Answer(BaseModel):
    """A model with a nested type, which is what breaks a naive schema dump."""

    verdict: str
    score: float = Field(ge=0.0, le=1.0)
    details: list[Nested] = Field(default_factory=list)


def fake_message(
    *,
    text: str = "",
    tool_use: dict[str, Any] | None = None,
    stop: str = "end_turn",
    stop_details: dict[str, str] | None = None,
) -> SimpleNamespace:
    """A stand-in for the SDK's response object."""
    content: list[SimpleNamespace] = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    if tool_use:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name=tool_use["name"],
                input=tool_use["input"],
            )
        )
    return SimpleNamespace(
        content=content,
        stop_reason=stop,
        stop_details=SimpleNamespace(**stop_details) if stop_details else None,
        model="claude-sonnet-5",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=40,
            cache_read_input_tokens=90,
            cache_creation_input_tokens=0,
        ),
    )


class RecordingClient:
    """Captures the payload instead of sending it."""

    def __init__(self, response: Any) -> None:
        self.payload: dict[str, Any] = {}
        self.messages = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **payload: Any) -> Any:
        self.payload = payload
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def provider_with(response: Any) -> tuple[ClaudeProvider, RecordingClient]:
    settings = LLMSettings(api_key="test-key")  # type: ignore[arg-type]
    provider = ClaudeProvider(settings)
    client = RecordingClient(response)
    provider._client = client
    return provider, client


# --------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------


class TestSchemaFlattening:
    def test_nested_refs_are_inlined(self) -> None:
        """The tools API does not resolve local ``$ref``.

        A nested model would otherwise arrive as a bare reference the API cannot
        follow, and the failure looks like the model ignoring the schema rather
        than the schema being unusable.
        """
        schema = _tool_schema(Answer)
        assert "$defs" not in schema
        assert "$ref" not in str(schema)
        items = schema["properties"]["details"]["items"]
        assert "label" in items["properties"]

    def test_flat_models_pass_through(self) -> None:
        schema = _tool_schema(Nested)
        assert set(schema["properties"]) == {"label", "weight"}

    def test_recursion_is_bounded(self) -> None:
        """A self-referential model must not recurse forever."""
        node = {"$ref": "#/$defs/Loop"}
        definitions = {"Loop": {"properties": {"next": {"$ref": "#/$defs/Loop"}}}}
        assert _inline_refs(node, definitions) is not None


class TestStructuredOutput:
    def test_forces_a_single_tool(self) -> None:
        """Constraining the shape beats asking for JSON and parsing it."""
        provider, client = provider_with(
            fake_message(
                tool_use={
                    "name": "emit_structured_response",
                    "input": {"verdict": "ok", "score": 0.8, "details": []},
                },
                stop="tool_use",
            )
        )
        result, response = provider.complete_structured(
            [Message(role="user", content="judge this")], Answer
        )

        assert result.verdict == "ok"
        assert client.payload["tool_choice"] == {
            "type": "tool",
            "name": "emit_structured_response",
        }
        assert response.usage.input_tokens == 120

    def test_missing_tool_call_is_an_error(self) -> None:
        """tool_choice forced it, so its absence is a provider failure rather
        than something to paper over with a default."""
        provider, _ = provider_with(fake_message(text="I'd rather not"))
        with pytest.raises(LLMResponseError, match="no emit_structured_response"):
            provider.complete_structured([Message(role="user", content="x")], Answer)

    def test_a_refusal_is_a_clear_error_not_a_validation_dump(self) -> None:
        """A refusal is the model declining the request, not the schema failing
        to constrain a response. Retrying the identical call fails identically,
        so the caller needs to know *why*, not see a Pydantic error about
        missing fields that were never going to be there."""
        provider, _ = provider_with(
            fake_message(
                tool_use={
                    "name": "emit_structured_response",
                    "input": {"rationale": "partial"},
                },
                stop="refusal",
                stop_details={
                    "category": "reasoning_extraction",
                    "explanation": "asked to reproduce internal reasoning",
                },
            )
        )
        with pytest.raises(LLMRefusalError) as excinfo:
            provider.complete_structured([Message(role="user", content="x")], Answer)

        assert excinfo.value.category == "reasoning_extraction"
        assert "reproduce internal reasoning" in (excinfo.value.explanation or "")

    def test_invalid_content_fails_validation(self) -> None:
        """The schema constrains shape, not every constraint. A bounded float is
        exactly the sort of rule JSON schema cannot always express."""
        provider, _ = provider_with(
            fake_message(
                tool_use={
                    "name": "emit_structured_response",
                    "input": {"verdict": "ok", "score": 99.0},
                },
                stop="tool_use",
            )
        )
        with pytest.raises(LLMResponseError, match="failed validation"):
            provider.complete_structured([Message(role="user", content="x")], Answer)


class TestPromptCaching:
    def test_system_prompt_is_cached(self) -> None:
        """It is large, stable, and re-sent every turn of an investigation."""
        provider, client = provider_with(fake_message(text="hi"))
        provider.complete([Message(role="user", content="q")], system="You are...")

        assert client.payload["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_manifest_is_cached_at_the_last_tool(self) -> None:
        """One breakpoint caches the whole manifest above it as a prefix."""
        provider, client = provider_with(fake_message())
        specs = [
            ToolSpec(name=f"t{i}", description="d", input_schema={"type": "object"})
            for i in range(3)
        ]
        provider.complete_with_tools([Message(role="user", content="q")], specs)

        tools = client.payload["tools"]
        assert "cache_control" not in tools[0]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_means_no_system_key(self) -> None:
        provider, client = provider_with(fake_message())
        provider.complete([Message(role="user", content="q")])
        assert "system" not in client.payload


class TestNoSamplingTemperature:
    """The Claude 5 family's `Messages.create()` does not declare a
    `temperature` parameter at all - passing one is a client-side `TypeError`
    raised before any request is sent. `temperature` stays an accepted argument
    on every public method for ABC and stub parity; it must simply never reach
    the wire."""

    def test_temperature_never_reaches_the_payload(self) -> None:
        provider, client = provider_with(fake_message())
        provider.complete([Message(role="user", content="q")], temperature=0.7)

        assert "temperature" not in client.payload

    def test_the_default_settings_temperature_is_also_withheld(self) -> None:
        provider, client = provider_with(fake_message())
        provider.complete([Message(role="user", content="q")])

        assert "temperature" not in client.payload


class TestResponseNormalisation:
    def test_text_and_tool_calls_are_separated(self) -> None:
        provider, _ = provider_with(
            fake_message(
                text="thinking",
                tool_use={"name": "forecast_demand", "input": {"product_id": "P1"}},
                stop="tool_use",
            )
        )
        response = provider.complete_with_tools([Message(role="user", content="q")], [])

        assert response.text == "thinking"
        assert response.wants_tools
        assert response.tool_calls[0].name == "forecast_demand"
        assert response.tool_calls[0].arguments == {"product_id": "P1"}

    def test_cache_tokens_are_recorded(self) -> None:
        """Cache reads are what make a long investigation affordable, so they
        have to be visible in the usage record."""
        provider, _ = provider_with(fake_message(text="x"))
        response = provider.complete([Message(role="user", content="q")])
        assert response.usage.cache_read_tokens == 90

    def test_provider_never_executes_tools(self) -> None:
        """Execution stays with the caller so permission checks and budget
        accounting happen in one auditable place."""
        provider, _ = provider_with(
            fake_message(
                tool_use={"name": "forecast_demand", "input": {}}, stop="tool_use"
            )
        )
        response = provider.complete_with_tools([Message(role="user", content="q")], [])
        assert isinstance(response.tool_calls[0], ToolCall)


class TestErrors:
    def test_missing_key_names_the_offline_option(self) -> None:
        # `_env_file=None`: a bare `LLMSettings()` reads the real `.env` file
        # directly, independent of `os.environ`, so this test's premise - no
        # key configured - would silently stop holding on any machine where a
        # developer has actually set one up.
        provider = ClaudeProvider(LLMSettings(_env_file=None))  # type: ignore[call-arg]
        with pytest.raises(LLMNotConfiguredError, match="LLM__PROVIDER=stub"):
            provider._require_key()

    def test_timeouts_are_mapped(self) -> None:
        class APITimeoutError(Exception):
            pass

        provider, _ = provider_with(APITimeoutError("deadline exceeded"))
        with pytest.raises(LLMTimeoutError):
            provider.complete([Message(role="user", content="q")])

    def test_other_failures_become_response_errors(self) -> None:
        provider, _ = provider_with(RuntimeError("500 from upstream"))
        with pytest.raises(LLMResponseError, match="Claude request failed"):
            provider.complete([Message(role="user", content="q")])

    def test_health_check_makes_no_network_call(self) -> None:
        ok, detail = ClaudeProvider(LLMSettings(_env_file=None)).health_check()  # type: ignore[call-arg]
        assert not ok
        assert "stub" in detail


# --------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------


class TestStub:
    def test_returns_the_scripted_object(self) -> None:
        stub = StubProvider().script_structured(Answer(verdict="planned", score=0.9))
        result, _ = stub.complete_structured([Message(role="user", content="q")], Answer)
        assert result.verdict == "planned"

    def test_is_deterministic(self) -> None:
        """The property the whole provider exists for."""
        stub = StubProvider().script_structured(Answer(verdict="same", score=0.5))
        first, _ = stub.complete_structured([Message(role="user", content="q")], Answer)
        second, _ = stub.complete_structured([Message(role="user", content="q")], Answer)
        assert first == second

    def test_routes_on_message_content(self) -> None:
        stub = (
            StubProvider()
            .script_structured(Answer(verdict="forecast", score=0.9), when_contains="forecast")
            .script_structured(Answer(verdict="uplift", score=0.9), when_contains="promotion")
        )
        forecast, _ = stub.complete_structured(
            [Message(role="user", content="give me a forecast")], Answer
        )
        uplift, _ = stub.complete_structured(
            [Message(role="user", content="did the promotion work")], Answer
        )
        assert forecast.verdict == "forecast"
        assert uplift.verdict == "uplift"

    def test_specific_beats_catch_all(self) -> None:
        """Lets a suite register a default and override it for one question."""
        stub = (
            StubProvider()
            .script_structured(Answer(verdict="default", score=0.1))
            .script_structured(Answer(verdict="special", score=0.9), when_contains="urgent")
        )
        result, _ = stub.complete_structured(
            [Message(role="user", content="this is urgent")], Answer
        )
        assert result.verdict == "special"

    def test_synthesises_when_nothing_is_registered(self) -> None:
        """Deliberately bland, so a forgotten registration fails on an obviously
        empty object rather than passing on a plausible guess."""
        result, _ = StubProvider().complete_structured(
            [Message(role="user", content="q")], Answer
        )
        assert result.verdict == ""
        assert result.details == []

    def test_wrong_scripted_type_is_refused(self) -> None:
        """A test that registers the wrong type should fail loudly, not hand
        the graph an object of a shape it will misread later."""
        stub = StubProvider()
        stub._scripted["Answer"] = [ScriptedCall(None, Nested(label="x"))]

        with pytest.raises(LLMResponseError, match="not a Answer"):
            stub.complete_structured([Message(role="user", content="q")], Answer)

    def test_records_calls_for_assertions(self) -> None:
        stub = StubProvider()
        stub.complete([Message(role="user", content="hello")], system="sys")
        assert stub.calls[0]["method"] == "complete"
        assert stub.calls[0]["has_system"]
        assert stub.calls[0]["last_user"] == "hello"

    def test_scripted_tool_calls(self) -> None:
        stub = StubProvider().script_tool_calls(
            [ToolCall(id="1", name="forecast_demand", arguments={"product_id": "P1"})]
        )
        response = stub.complete_with_tools([Message(role="user", content="q")], [])
        assert response.wants_tools
        assert response.tool_calls[0].name == "forecast_demand"

    def test_needs_no_key(self) -> None:
        ok, detail = StubProvider().health_check()
        assert ok
        assert "offline" in detail

    def test_reset_clears_everything(self) -> None:
        stub = StubProvider().script_text("x")
        stub.complete([Message(role="user", content="q")])
        stub.reset()
        assert stub.calls == []
        assert stub._text == []


class TestContainerSelection:
    """Selection goes through the real configuration path.

    Settings are frozen, so these set the environment variable rather than
    mutating an instance - which also means they exercise the mechanism a
    deployment actually uses.
    """

    def test_stub_is_selected_by_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.services.container import Container
        from tests.conftest import build_settings

        monkeypatch.setenv("LLM__PROVIDER", "stub")
        assert isinstance(Container(build_settings()).llm_provider, StubProvider)

    def test_claude_is_the_default(self) -> None:
        from app.services.container import Container
        from tests.conftest import build_settings

        assert isinstance(Container(build_settings()).llm_provider, ClaudeProvider)
