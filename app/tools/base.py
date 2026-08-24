"""Analytical tool base class - the enforcement point for the tool contract.

This is one of the three seams the whole architecture rests on.

The platform's core safety property is that Claude interprets numbers but never
produces them. Prompting cannot guarantee that. What guarantees it is this: an
agent's only route to a number is :meth:`AnalyticalTool.run`, and ``run`` is
concrete and final. Subclasses implement :meth:`_execute` and never construct a
:class:`ToolResult` themselves, so no tool can return a bare float, an unversioned
figure, or an untimed result - the wrapper always attaches status, provenance,
timing and trace id.

The wrapper also converts exceptions into a *structured* failure rather than
letting them escape. That matters for agentic behaviour: a raised exception
kills the graph, whereas a ``ToolResult`` with a recoverable error code lets the
Supervisor re-plan around the failure, which is exactly the behaviour section 18
asks for.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, final

from pydantic import BaseModel, ValidationError

from app.observability.context import get_trace_id, new_trace_id
from app.observability.logging import get_logger
from app.observability.metrics import METRICS
from app.schemas.tool_contract import (
    ModelProvenance,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
)

logger = get_logger(__name__)

Permission = Literal["read_analytics", "read_documents", "run_model", "optimise"]


class ToolExecutionError(Exception):
    """Raised inside :meth:`AnalyticalTool._execute` to signal a typed failure.

    Prefer this over a bare exception: it lets a tool distinguish "not enough
    data for this product" (recoverable - the Supervisor should try something
    else) from an unexpected internal fault.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.INTERNAL_ERROR,
        recoverable: bool = True,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.detail = detail or {}


class ToolOutput[TOut: BaseModel](BaseModel):
    """What :meth:`AnalyticalTool._execute` returns.

    Separating the payload from its metadata is what keeps the wrapper honest:
    ``_execute`` supplies the numbers and the caveats, the wrapper supplies the
    envelope. A tool cannot forge a status or a timing.
    """

    payload: TOut
    provenance: ModelProvenance | None = None
    confidence: float | None = None
    assumptions: list[str] = []
    warnings: list[str] = []


class AnalyticalTool[TIn: BaseModel, TOut: BaseModel](ABC):
    """Base class for every analytical capability exposed to an agent.

    Subclasses declare ``name``, ``description``, ``input_schema``,
    ``output_schema`` and implement :meth:`_execute`.
    """

    #: Stable tool name. This is what Claude sees and what the Supervisor plans with.
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    #: Coarse permission tag, enforced by the guardrail layer (Step 20).
    permission: ClassVar[Permission] = "run_model"
    #: Wall-clock ceiling. Enforced by subclasses / the executor in later steps.
    timeout_seconds: ClassVar[float] = 60.0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Fail loudly at import time if a subclass forgets its contract.

        A tool missing its schemas would otherwise fail much later, mid
        investigation, with a confusing AttributeError.
        """
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        required = ("name", "description", "input_schema", "output_schema")
        missing = [attr for attr in required if not getattr(cls, attr, None)]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define class attributes: {', '.join(missing)}"
            )

    # -- subclass hook ------------------------------------------------------

    @abstractmethod
    def _execute(self, payload: TIn) -> ToolOutput[TOut]:
        """Perform the analysis. Receives validated input.

        Raise :class:`ToolExecutionError` for expected failures. Any other
        exception is caught by :meth:`run` and reported as a non-recoverable
        internal error.
        """

    # -- final, non-overridable entrypoint ----------------------------------

    @final
    def run(self, raw_input: dict[str, Any] | TIn) -> ToolResult:
        """Validate, execute, and wrap. The only public way to invoke a tool.

        Never raises: every outcome is expressed as a :class:`ToolResult`.
        """
        trace_id = get_trace_id() or new_trace_id()
        started = time.perf_counter()
        log = logger.bind(tool=self.name, trace_id=trace_id)

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        # --- input validation ---
        try:
            payload = (
                raw_input
                if isinstance(raw_input, self.input_schema)
                else self.input_schema.model_validate(raw_input)
            )
        except ValidationError as exc:
            log.warning("tool.invalid_input", errors=exc.error_count())
            return ToolResult.failure(
                tool_name=self.name,
                code=ToolErrorCode.INVALID_INPUT,
                message=f"Input failed validation for tool '{self.name}'.",
                recoverable=True,
                detail={"validation_errors": exc.errors(include_url=False)},
                execution_time_ms=elapsed_ms(),
                trace_id=trace_id,
            )

        # --- execution ---
        try:
            output = self._execute(payload)  # type: ignore[arg-type]
        except ToolExecutionError as exc:
            log.warning("tool.failed", code=exc.code.value, error=str(exc))
            METRICS.increment("tool_calls_total")
            return ToolResult.failure(
                tool_name=self.name,
                code=exc.code,
                message=str(exc),
                recoverable=exc.recoverable,
                detail=exc.detail,
                execution_time_ms=elapsed_ms(),
                trace_id=trace_id,
            )
        except Exception as exc:
            # Deliberate blanket boundary. An unexpected fault must not kill the
            # graph: log with traceback, return a non-recoverable error, and let
            # the Supervisor decide whether to re-plan around it.
            log.exception("tool.internal_error", error=str(exc))
            METRICS.increment("tool_calls_total")
            return ToolResult.failure(
                tool_name=self.name,
                code=ToolErrorCode.INTERNAL_ERROR,
                message=f"Tool '{self.name}' failed unexpectedly: {exc}",
                recoverable=False,
                execution_time_ms=elapsed_ms(),
                trace_id=trace_id,
            )

        duration = elapsed_ms()
        METRICS.increment("tool_calls_total")
        log.info("tool.completed", duration_ms=duration)

        return ToolResult.success(
            tool_name=self.name,
            result=output.payload.model_dump(mode="json"),
            provenance=output.provenance,
            confidence=output.confidence,
            assumptions=output.assumptions,
            warnings=output.warnings,
            execution_time_ms=duration,
            trace_id=trace_id,
        )

    # -- tool-calling metadata ---------------------------------------------

    @classmethod
    def spec(cls) -> ToolSpec:
        """Describe this tool for Claude's tool-calling API.

        Generated from the Pydantic input schema so the JSON schema the model is
        shown and the schema we validate against cannot drift apart.
        """
        return ToolSpec(
            name=cls.name,
            description=cls.description.strip(),
            input_schema=cls.input_schema.model_json_schema(),
            permission=cls.permission,
        )
