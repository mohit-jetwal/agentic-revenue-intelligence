"""The analytical tool contract.

This module defines the *only* shape in which a numerical result may reach the
LLM. It implements section 13 of the brief.

Why this matters more than it looks: the central safety property of this
platform is that Claude reasons about numbers but never produces them. That
property is not enforceable by prompting alone - a prompt is a request, not a
guarantee. It is enforced structurally: every analytical capability is a
:class:`~app.tools.base.AnalyticalTool` whose ``run()`` returns a
:class:`ToolResult`, and the fields that carry numbers (``result``) are always
accompanied by the provenance needed to audit them (``model_name``,
``model_version``, ``dataset_version``, ``mlflow_run_id``).

An agent that wants a number has exactly one way to get one, and that way
always records where it came from.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(StrEnum):
    """Outcome of a tool invocation."""

    SUCCESS = "success"
    #: Completed, but with caveats the agent must surface (see ``warnings``).
    PARTIAL = "partial"
    ERROR = "error"
    #: Rejected before execution: bad input, missing data, or denied permission.
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"


class ToolErrorCode(StrEnum):
    """Machine-readable failure reasons, so the Supervisor can re-plan.

    The distinction that matters: ``INSUFFICIENT_DATA`` and ``MODEL_NOT_FOUND``
    are recoverable by choosing a different analysis, whereas ``INTERNAL_ERROR``
    is not. Re-planning logic keys off this, not off the message text.
    """

    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_INPUT = "invalid_input"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_NOT_APPROVED = "model_not_approved"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL_ERROR = "internal_error"


class ToolError(BaseModel):
    """Structured error detail attached to a failed :class:`ToolResult`."""

    model_config = ConfigDict(frozen=True)

    code: ToolErrorCode
    message: str
    #: True when a different plan could still succeed - drives re-planning.
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)


class ModelProvenance(BaseModel):
    """Where a number came from. Required for every numerical result.

    Section 29 of the brief: an agent must never present a figure it cannot
    attribute to an approved, versioned model run.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    model_version: str
    #: Version/hash of the dataset the model was fitted or scored against.
    dataset_version: str | None = None
    #: Populated once MLflow tracking is wired in (Stage 1 Step 12).
    mlflow_run_id: str | None = None
    #: Registry stage the model was loaded from, e.g. "Production".
    model_stage: str | None = None


class ToolResult(BaseModel):
    """Uniform envelope returned by every analytical tool.

    Matches section 13 of the brief. ``result`` holds the tool-specific payload
    (itself a validated Pydantic model, serialised); everything else is the
    metadata that makes the payload auditable and interpretable.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: ToolStatus
    tool_name: str
    model_name: str | None = None
    model_version: str | None = None
    dataset_version: str | None = None

    #: The tool-specific payload. Empty on error.
    result: dict[str, Any] = Field(default_factory=dict)

    #: Tool-reported confidence in [0, 1]. ``None`` when not meaningful
    #: (e.g. a deterministic SQL aggregation, where "confidence" would be noise).
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    #: Modelling assumptions the agent must carry into its recommendation.
    assumptions: list[str] = Field(default_factory=list)
    #: Caveats that do not invalidate the result but must be surfaced.
    warnings: list[str] = Field(default_factory=list)

    execution_time_ms: int = Field(default=0, ge=0)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    error: ToolError | None = None

    @property
    def ok(self) -> bool:
        return self.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    @property
    def is_recoverable_failure(self) -> bool:
        """True when re-planning could plausibly succeed with a different tool."""
        return not self.ok and (self.error is None or self.error.recoverable)

    # -- constructors -------------------------------------------------------
    # Tools never build a ToolResult by hand; the base class wrapper calls
    # these. They exist as classmethods so the shape stays in one place.

    @classmethod
    def success(
        cls,
        *,
        tool_name: str,
        result: dict[str, Any],
        provenance: ModelProvenance | None = None,
        confidence: float | None = None,
        assumptions: list[str] | None = None,
        warnings: list[str] | None = None,
        execution_time_ms: int = 0,
        trace_id: str | None = None,
    ) -> ToolResult:
        status = ToolStatus.PARTIAL if warnings else ToolStatus.SUCCESS
        return cls(
            status=status,
            tool_name=tool_name,
            model_name=provenance.model_name if provenance else None,
            model_version=provenance.model_version if provenance else None,
            dataset_version=provenance.dataset_version if provenance else None,
            result=result,
            confidence=confidence,
            assumptions=assumptions or [],
            warnings=warnings or [],
            execution_time_ms=execution_time_ms,
            trace_id=trace_id or str(uuid.uuid4()),
        )

    @classmethod
    def failure(
        cls,
        *,
        tool_name: str,
        code: ToolErrorCode,
        message: str,
        recoverable: bool = True,
        detail: dict[str, Any] | None = None,
        execution_time_ms: int = 0,
        trace_id: str | None = None,
    ) -> ToolResult:
        status = ToolStatus.TIMEOUT if code is ToolErrorCode.TIMEOUT else ToolStatus.ERROR
        if code is ToolErrorCode.INVALID_INPUT:
            status = ToolStatus.INVALID_INPUT
        return cls(
            status=status,
            tool_name=tool_name,
            error=ToolError(
                code=code,
                message=message,
                recoverable=recoverable,
                detail=detail or {},
            ),
            execution_time_ms=execution_time_ms,
            trace_id=trace_id or str(uuid.uuid4()),
        )

    def for_llm(self) -> dict[str, Any]:
        """Compact representation handed to Claude.

        Deliberately excludes ``created_at`` and internal error detail: the
        model needs the numbers and their provenance, not our plumbing. Keeping
        this narrow also keeps token cost down on long investigations.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "tool_name": self.tool_name,
            "result": self.result,
        }
        if self.model_name:
            payload["model"] = f"{self.model_name}:{self.model_version}"
        if self.dataset_version:
            payload["dataset_version"] = self.dataset_version
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.assumptions:
            payload["assumptions"] = self.assumptions
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.error is not None:
            payload["error"] = {
                "code": self.error.code.value,
                "message": self.error.message,
                "recoverable": self.error.recoverable,
            }
        return payload


class ToolSpec(BaseModel):
    """Declarative description of a tool, used for Claude tool-calling.

    Built from the tool's Pydantic input schema so the JSON schema Claude sees
    and the schema we validate against can never drift apart.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    #: Coarse permission tag; the guardrail layer (Step 20) enforces it.
    permission: Literal["read_analytics", "read_documents", "run_model", "optimise"] = (
        "run_model"
    )
