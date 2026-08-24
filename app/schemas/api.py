"""Request and response models for the HTTP API (section 20 of the brief)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_state import Recommendation
from app.schemas.domain import BusinessObjective, IntentType


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Uniform error body. Every non-2xx response uses this shape."""

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str | None = None
    trace_id: str | None = None
    #: For 501 stubs: which implementation step will provide this endpoint.
    implemented_in: str | None = None


# ---------------------------------------------------------------------------
# Health / metrics
# ---------------------------------------------------------------------------
class DependencyStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class DependencyCheck(BaseModel):
    name: str
    status: DependencyStatus
    detail: str | None = None


class HealthResponse(BaseModel):
    status: DependencyStatus
    name: str
    version: str
    environment: str
    dependencies: list[DependencyCheck] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Lightweight JSON counters.

    Deliberately not a Prometheus exposition format: in Stage 1 nothing scrapes
    it, and in Stage 2 monitoring is Databricks-native. See README.
    """

    uptime_seconds: float
    requests_total: int
    requests_failed: int
    investigations_started: int
    investigations_completed: int
    tool_calls_total: int
    tokens_used_total: int


# ---------------------------------------------------------------------------
# Chat / investigation
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    investigation_id: str
    trace_id: str
    answer: str
    intent: IntentType | None = None
    recommendation: Recommendation | None = None


class InvestigateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    objective: BusinessObjective | None = None
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    user_id: str | None = None


class InvestigationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class InvestigationResponse(BaseModel):
    investigation_id: str
    trace_id: str
    status: InvestigationStatus
    question: str
    intent: IntentType | None = None
    objective: BusinessObjective | None = None
    recommendation: Recommendation | None = None
    created_at: datetime
    completed_at: datetime | None = None


class TraceEvent(BaseModel):
    """One entry in the agentic trace shown to the user (section 22).

    ``reasoning_summary`` is a short, user-facing rationale. Private
    chain-of-thought is never stored here or returned by the API.
    """

    sequence: int
    timestamp: datetime
    event_type: str
    actor: str
    summary: str
    tool_name: str | None = None
    duration_ms: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    investigation_id: str
    trace_id: str
    events: list[TraceEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
class ScenarioRequest(BaseModel):
    """A what-if question (section 8, Scenario Engine)."""

    description: str = Field(min_length=3, max_length=1000)
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None
    price_change_pct: float | None = Field(default=None, ge=-90.0, le=200.0)
    promotion_spend_change: float | None = None
    competitor_price_change_pct: float | None = Field(default=None, ge=-90.0, le=200.0)
    inventory_change_pct: float | None = Field(default=None, ge=-100.0, le=500.0)
    horizon_days: int = Field(default=30, gt=0, le=365)


# ---------------------------------------------------------------------------
# Models / feedback
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    stage: str
    description: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    trained_at: datetime | None = None
    approved: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    investigation_id: str
    helpful: bool
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    user_id: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    received: bool = True
