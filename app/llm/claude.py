"""Claude provider.

Skeleton for Stage 1 Step 14. Construction and configuration checks work today
so the container and ``GET /health`` can report accurately; the request methods
are implemented in Step 14.

Implementation notes for Step 14:

* Use the official ``anthropic`` SDK. Retries and timeouts are configured on the
  client rather than hand-rolled - the SDK already implements exponential
  backoff with jitter and respects ``retry-after``.
* ``complete_structured`` should use tool-calling with a single forced tool whose
  input schema is the target Pydantic model's JSON schema. That is materially
  more reliable than asking for JSON in the prompt and parsing it, because the
  API constrains the output shape rather than merely requesting it.
* Cache the system prompt and the tool specs with prompt caching: they are large,
  stable, and re-sent on every turn of a long investigation.
* Never log message content at INFO. Log token counts, model, stop reason and
  prompt version instead - business data must not leak into logs.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.config.settings import LLMSettings
from app.llm.base import (
    LLMNotConfiguredError,
    LLMProvider,
    LLMResponse,
    Message,
)
from app.schemas.tool_contract import ToolSpec

TModel = TypeVar("TModel", bound=BaseModel)

_STEP = "Stage 1 Step 14 (Claude integration)"


class ClaudeProvider(LLMProvider):
    """Anthropic Claude backend."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        # The client is created lazily in Step 14 so that importing this module
        # (and constructing the container) never requires a key.
        self._client: object | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def planner_model_name(self) -> str:
        """Model used for planning and re-planning."""
        return self._settings.planner_model

    def _require_key(self) -> str:
        secret = self._settings.api_key
        key = secret.get_secret_value() if secret is not None else ""
        if not key:
            raise LLMNotConfiguredError(
                "No Claude API key configured. Set LLM__API_KEY in .env "
                "(see .env.example). Not required until Stage 1 Step 14."
            )
        return key

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"ClaudeProvider.{method}() is implemented in {_STEP}")

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        raise self._not_yet("complete")

    def complete_structured(
        self,
        messages: list[Message],
        response_model: type[TModel],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[TModel, LLMResponse]:
        raise self._not_yet("complete_structured")

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        raise self._not_yet("complete_with_tools")

    def count_tokens(self, text: str) -> int:
        """Rough estimate until the SDK's token counting endpoint is wired in.

        ~4 characters per token is close enough for budget guarding, which only
        needs to catch runaway growth, not to bill anyone.
        """
        return max(1, len(text) // 4)

    def health_check(self) -> tuple[bool, str]:
        if not self._settings.is_configured:
            return False, "LLM__API_KEY not set (not required until Step 14)"
        return True, f"claude ({self._settings.model} / planner {self._settings.planner_model})"
