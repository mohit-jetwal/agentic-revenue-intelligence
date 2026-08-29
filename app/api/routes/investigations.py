"""Investigation, chat and scenario endpoints.

The contracts were declared in Step 1 and returned 501 until the agent behind
them existed. Step 14 implements them against the same
:class:`InvestigationService` the CLI and the UI use, so there is one path
through the system rather than three that drift.

**Investigations run synchronously.** A background queue would be the right
answer at production volumes, and it is the wrong answer here: it would add a
worker, a result store and a polling contract to serve a demo where the answer
takes seconds. ``GET /investigation/{id}`` exists and works, so the async
version is a change of caller behaviour rather than a rewrite.

**A failed investigation returns 200 with a `failed` status**, not a 5xx. The
request was handled correctly; the investigation is what did not conclude, and
that distinction is what lets a caller tell "the platform is down" from "the
evidence did not support an answer".
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import ContainerDep
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

_NOT_FOUND: dict[int | str, dict[str, object]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No investigation with that id.",
    }
}

_MLFLOW_STEP = "Stage 2 (Databricks Model Registry)"


@router.post("/chat", response_model=ChatResponse, summary="Ask a business question")
def chat(request: ChatRequest, container: ContainerDep) -> ChatResponse:
    """Conversational entry point. Routes to the Supervisor agent."""
    outcome = container.investigation_service.run(
        request.question, user_id=request.user_id, session_id=request.session_id
    )
    return ChatResponse(
        investigation_id=outcome.investigation_id,
        trace_id=outcome.trace_id,
        answer=outcome.answer,
        intent=outcome.intent,  # type: ignore[arg-type]
        recommendation=outcome.recommendation,
    )


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    summary="Run a scoped investigation",
)
def investigate(
    request: InvestigateRequest, container: ContainerDep
) -> InvestigationResponse:
    """Run a full agentic investigation with explicit scope filters.

    The scope is handed to the agent rather than extracted from prose. A caller
    who already knows which products they mean should not have their filter
    depend on the model reading it correctly.
    """
    outcome = container.investigation_service.run(
        request.question,
        user_id=request.user_id,
        scope={
            "objective": request.objective,
            "products": request.product_ids,
            "stores": request.store_ids,
            "region": request.region,
            "start_date": request.start_date,
            "end_date": request.end_date,
        },
    )

    stored = container.investigation_service.get(outcome.investigation_id)
    if stored is not None:
        return stored

    # The store is unavailable but the investigation ran. Returning what is in
    # hand beats a 500 that discards a completed answer over a logging failure.
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="investigation completed but could not be persisted",
    )


@router.get(
    "/investigation/{investigation_id}",
    response_model=InvestigationResponse,
    responses=_NOT_FOUND,
    summary="Fetch an investigation",
)
def get_investigation(
    investigation_id: str, container: ContainerDep
) -> InvestigationResponse:
    """Retrieve a previous investigation and its recommendation."""
    found = container.investigation_service.get(investigation_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no investigation {investigation_id}",
        )
    return found


@router.get(
    "/investigation/{investigation_id}/trace",
    response_model=TraceResponse,
    responses=_NOT_FOUND,
    summary="Fetch the agentic trace",
)
def get_trace(investigation_id: str, container: ContainerDep) -> TraceResponse:
    """Return the plan, tool calls, re-planning events and critic verdict.

    Returns user-facing reasoning summaries only. Private chain-of-thought is
    never persisted or exposed.
    """
    found = container.investigation_service.get_trace(investigation_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no investigation {investigation_id}",
        )
    return found


@router.post("/scenario", summary="Simulate a what-if scenario")
def scenario(request: ScenarioRequest, container: ContainerDep) -> dict[str, Any]:
    """Project the impact of price, promotion, competitor or inventory changes.

    Calls the scenario tool directly rather than through the agent: the caller
    has already specified every lever, so there is nothing left to reason about
    and a planning round trip would add latency and cost for no decision.
    """
    if not container.tool_registry.has("simulate_scenario"):
        raise NotImplementedYetError("POST /scenario", "Stage 1 Step 9")

    payload, ignored = _scenario_payload(request)
    result = container.tool_registry.get("simulate_scenario").run(payload)
    if result.status == "error":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=result.error.message if result.error else "scenario failed",
        )
    return {
        "trace_id": result.trace_id,
        "result": result.result,
        "assumptions": result.assumptions,
        # Anything the engine could not model is reported, never dropped. A
        # projection that silently ignored the inventory change the caller asked
        # about would answer a different question than the one posed.
        "warnings": list(result.warnings) + ignored,
    }


#: Request field -> the lever the scenario engine models.
_LEVERS: tuple[tuple[str, str], ...] = (
    ("price_change_pct", "price"),
    ("competitor_price_change_pct", "competitor_price"),
)


def _scenario_payload(request: ScenarioRequest) -> tuple[dict[str, Any], list[str]]:
    """Translate the API request into the tool's input.

    The two shapes differ in three ways, and each is handled explicitly rather
    than by hoping the field names line up: the API takes percentages where the
    engine takes fractions, the API takes a list of products where the engine
    projects one, and the API accepts two levers the engine does not model.
    """
    ignored: list[str] = []

    if not request.product_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "product_ids is required: the scenario engine projects through "
                "one product's demand equation and has no basis for a portfolio "
                "aggregate"
            ),
        )
    if len(request.product_ids) > 1:
        ignored.append(
            f"only {request.product_ids[0]} was projected; the engine models one "
            f"product at a time and cannot aggregate across "
            f"{len(request.product_ids)} without double-counting substitution"
        )
    if request.store_ids:
        ignored.append(
            "store_ids was ignored: the demand equation is fitted at product "
            "level, so a store filter would narrow the data without changing "
            "the projection"
        )
    if request.promotion_spend_change is not None:
        ignored.append(
            "promotion_spend_change was ignored: the engine takes a percentage "
            "depth change, and converting a spend amount to a depth needs a "
            "promotion plan that was not supplied"
        )
    if request.inventory_change_pct is not None:
        ignored.append(
            "inventory_change_pct was ignored: no availability model is "
            "registered, so an inventory lever has nothing to project through"
        )

    levers = [
        # Percent in, fraction out. The API says -5.0 and the engine means -0.05.
        {"lever": lever, "change_pct": getattr(request, field) / 100.0}
        for field, lever in _LEVERS
        if getattr(request, field) is not None
    ]
    if not levers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "no modellable lever supplied. Set price_change_pct or "
                "competitor_price_change_pct."
            ),
        )

    return (
        {
            "product_id": request.product_ids[0],
            "levers": levers,
            "region": request.region,
            "horizon_days": request.horizon_days,
            "scenario_name": request.description[:80],
        },
        ignored,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    responses=_STUB_RESPONSES,
    summary="List registered models",
)
def list_models(container: ContainerDep) -> ModelsResponse:
    """Registered analytical models with version and stage.

    Metrics are deliberately absent rather than invented. The local MLflow store
    records them per run; surfacing a training metric here as though it were a
    live production figure is exactly the kind of number that gets quoted in a
    meeting and cannot be traced back.
    """
    try:
        return ModelsResponse(models=container.model_registry.list_models())
    except NotImplementedError as exc:
        raise NotImplementedYetError("GET /models", _MLFLOW_STEP) from exc


@router.post(
    "/feedback", response_model=FeedbackResponse, summary="Submit feedback"
)
def feedback(request: FeedbackRequest, container: ContainerDep) -> FeedbackResponse:
    """Record user feedback, used later for agent evaluation.

    Accepted for an investigation id the store does not have. Feedback is the
    only human-labelled signal this platform produces, and rejecting it because
    a demo database was reset would discard the scarcest data here.
    """
    feedback_id = container.investigation_store.add_feedback(
        investigation_id=request.investigation_id,
        helpful=request.helpful,
        rating=request.rating,
        comment=request.comment,
        user_id=request.user_id,
    )
    return FeedbackResponse(feedback_id=feedback_id)

