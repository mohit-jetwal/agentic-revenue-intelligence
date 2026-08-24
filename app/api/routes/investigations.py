"""Investigation, chat and scenario endpoints.

Registered from Step 1 so the full API surface from section 20 is visible in
OpenAPI, and every stub names the step that implements it. A 501 that says
"Stage 1 Step 19" is a roadmap; a missing route is just a 404.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.errors import NotImplementedYetError
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    InvestigateRequest,
    InvestigationResponse,
    ModelsResponse,
    ScenarioRequest,
    TraceResponse,
)

router = APIRouter(tags=["investigations"])

_STUB_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_501_NOT_IMPLEMENTED: {
        "model": ErrorResponse,
        "description": "Endpoint scheduled for a later implementation step.",
    }
}

_API_STEP = "Stage 1 Step 19 (FastAPI application)"
_AGENT_STEP = "Stage 1 Step 16 (LangGraph Supervisor)"
_SCENARIO_STEP = "Stage 1 Step 11 (scenario simulation engine)"
_MLFLOW_STEP = "Stage 1 Step 12 (MLflow)"


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses=_STUB_RESPONSES,
    summary="Ask a business question",
)
def chat(request: ChatRequest) -> ChatResponse:
    """Conversational entry point. Routes to the Supervisor agent."""
    raise NotImplementedYetError("POST /chat", _AGENT_STEP)


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    responses=_STUB_RESPONSES,
    summary="Run a scoped investigation",
)
def investigate(request: InvestigateRequest) -> InvestigationResponse:
    """Run a full agentic investigation with explicit scope filters."""
    raise NotImplementedYetError("POST /investigate", _AGENT_STEP)


@router.get(
    "/investigation/{investigation_id}",
    response_model=InvestigationResponse,
    responses=_STUB_RESPONSES,
    summary="Fetch an investigation",
)
def get_investigation(investigation_id: str) -> InvestigationResponse:
    """Retrieve a previous investigation and its recommendation."""
    raise NotImplementedYetError("GET /investigation/{id}", _API_STEP)


@router.get(
    "/investigation/{investigation_id}/trace",
    response_model=TraceResponse,
    responses=_STUB_RESPONSES,
    summary="Fetch the agentic trace",
)
def get_trace(investigation_id: str) -> TraceResponse:
    """Return the plan, tool calls, re-planning events and critic verdict.

    Returns user-facing reasoning summaries only. Private chain-of-thought is
    never persisted or exposed.
    """
    raise NotImplementedYetError("GET /investigation/{id}/trace", _API_STEP)


@router.post(
    "/scenario",
    responses=_STUB_RESPONSES,
    summary="Simulate a what-if scenario",
)
def scenario(request: ScenarioRequest) -> dict[str, object]:
    """Project the impact of price, promotion, competitor or inventory changes."""
    raise NotImplementedYetError("POST /scenario", _SCENARIO_STEP)


@router.get(
    "/models",
    response_model=ModelsResponse,
    responses=_STUB_RESPONSES,
    summary="List registered models",
)
def list_models() -> ModelsResponse:
    """Registered analytical models with version, stage and metrics."""
    raise NotImplementedYetError("GET /models", _MLFLOW_STEP)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses=_STUB_RESPONSES,
    summary="Submit feedback on a recommendation",
)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback, used later for agent evaluation."""
    raise NotImplementedYetError("POST /feedback", _API_STEP)
