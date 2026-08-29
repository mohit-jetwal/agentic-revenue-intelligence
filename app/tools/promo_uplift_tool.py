"""The promo uplift tool (brief sections 30, 39).

The contract a future Claude agent calls. No LangGraph here and none needed:
``AnalyticalTool`` already owns validation, timing, tracing and error wrapping,
so this class writes ``_execute`` and four class attributes.

**Why the agent must not compute this itself.** An LLM can subtract two averages,
and that is precisely the danger - the arithmetic is trivial and the answer is
wrong. Uplift requires a counterfactual that is absent from every dataset, a
control group chosen under stated assumptions, an adjustment for confounders
that must exclude mediators, and a validation suite that can *reject* the whole
estimate. None of that is inferable from a table of numbers in a context window.
A model asked "how much did this promotion generate" will produce a confident
figure by comparing promoted days to unpromoted ones, which on this data
overstates by around 45 percentage points.

So the tool returns **evidence**, not just a number:

* ``validation_status`` - whether the causal assumptions held. The agent must
  not describe a ``failed`` estimate as causal, and the description says so.
* ``treatment_definition`` - what the number means. Two uplift figures computed
  under different definitions are not comparable.
* ``assumptions`` - the conditions under which the estimate *is* the effect.
* ``method_reason`` - why this estimator was chosen over the others.
* the naive comparison, marked ineligible, so the agent can state the size of
  the error it avoided.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import get_logger
from app.schemas.promo_uplift import UpliftErrorResponse, UpliftRequest
from app.schemas.tool_contract import ModelProvenance, ToolErrorCode
from app.services.promo_uplift_service import PromoUpliftService
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput

logger = get_logger(__name__)


class PromoUpliftInput(BaseModel):
    """What the agent asks for."""

    model_config = ConfigDict(frozen=True)

    promotion_id: str | None = Field(
        default=None,
        description=(
            "A specific promotion to measure. Omit to aggregate across every "
            "promotion matching the other filters."
        ),
    )
    product_id: str | None = Field(default=None, description="Restrict to one product.")
    store_id: str | None = Field(default=None, description="Restrict to one store.")
    region: str | None = Field(default=None, description="Restrict to one region.")
    category: str | None = Field(default=None, description="Restrict to one category.")
    start_date: date | None = Field(
        default=None, description="Only promotions starting on or after this date."
    )
    end_date: date | None = Field(
        default=None, description="Only promotions ending on or before this date."
    )
    include_segments: bool = Field(
        default=True,
        description=(
            "Include uplift broken down by category, region and mechanic. Use "
            "this to answer which segments respond best."
        ),
    )
    include_events: bool = Field(
        default=True,
        description="Include per-promotion incremental profit and ROI, ranked.",
    )
    max_events: int = Field(
        default=50, gt=0, le=1000, description="Cap on individual promotions returned."
    )


class PromoUpliftOutput(BaseModel):
    """What the agent gets back."""

    model_config = ConfigDict(protected_namespaces=())

    #: The estimand. Required reading before the number below means anything.
    treatment_definition: str
    method: str
    method_reason: str

    incremental_units: float
    incremental_revenue: float
    incremental_profit: float
    promotion_spend: float
    roi: float | None
    uplift_pct: float
    baseline_units: float
    observed_units: float

    confidence_interval: dict[str, float] | None = None
    #: passed | warnings | failed. A `failed` estimate must not be described as
    #: causal, however reasonable the number looks.
    validation_status: str = "not_assessed"
    events_analysed: int = 0

    events: list[dict[str, object]] = Field(default_factory=list)
    segments: list[dict[str, object]] = Field(default_factory=list)
    #: Every estimator's result, including the naive comparison marked
    #: ineligible - so the agent can quantify the bias it avoided rather than
    #: assert that one exists.
    method_comparison: list[dict[str, object]] = Field(default_factory=list)


class PromoUpliftTool(AnalyticalTool[PromoUpliftInput, PromoUpliftOutput]):
    """Measure the incremental sales and profit a promotion caused."""

    name = "estimate_promo_uplift"
    description = (
        "Estimate the INCREMENTAL sales, profit and ROI caused by a promotion - "
        "what happened versus what would have happened without it. This is a "
        "causal estimate, not a comparison of promoted to unpromoted sales: it "
        "adjusts for the fact that promotions are scheduled into strong demand "
        "periods and that shoppers buy ahead, both of which make a naive "
        "comparison overstate incrementality substantially. Returns incremental "
        "units, revenue, profit and ROI with a measured confidence interval, "
        "per-promotion detail ranked by ROI, segment-level effects, and a "
        "validation status saying whether the causal assumptions held. Use it "
        "for 'how much did promotion X generate', 'which products give the best "
        "promotional return', and 'was this promotion worth it'. Do NOT compute "
        "uplift yourself from sales figures - the counterfactual is not in the "
        "data."
    )
    input_schema = PromoUpliftInput
    output_schema = PromoUpliftOutput
    permission = "run_model"
    timeout_seconds = 180.0

    def __init__(self, service: PromoUpliftService) -> None:
        self._service = service

    def _execute(self, payload: PromoUpliftInput) -> ToolOutput[PromoUpliftOutput]:
        response = self._service.estimate_uplift(
            UpliftRequest(
                promotion_ids=[payload.promotion_id] if payload.promotion_id else None,
                product_ids=[payload.product_id] if payload.product_id else None,
                store_ids=[payload.store_id] if payload.store_id else None,
                region=payload.region,
                category=payload.category,
                analysis_start_date=payload.start_date,
                analysis_end_date=payload.end_date,
                include_segments=payload.include_segments,
                include_events=payload.include_events,
                max_events=payload.max_events,
            )
        )

        # `isinstance` rather than a status-string check: it narrows the union
        # for the type checker as well as at runtime.
        if isinstance(response, UpliftErrorResponse):
            raise ToolExecutionError(
                response.message,
                code=_ERROR_CODES.get(response.error_code, ToolErrorCode.INTERNAL_ERROR),
                recoverable=response.recoverable,
                detail=response.detail,
            )

        interval = None
        if response.confidence_interval is not None:
            interval = {
                "lower": round(response.confidence_interval.lower, 4),
                "upper": round(response.confidence_interval.upper, 4),
                "confidence_level": response.confidence_interval.confidence_level,
            }

        output = PromoUpliftOutput(
            treatment_definition=response.treatment_definition,
            method=response.method,
            method_reason=response.method_reason,
            incremental_units=round(response.incremental_units, 1),
            incremental_revenue=round(response.incremental_revenue, 2),
            incremental_profit=round(response.incremental_profit, 2),
            promotion_spend=round(response.promotion_spend, 2),
            roi=round(response.roi, 3) if response.roi is not None else None,
            uplift_pct=round(response.uplift_pct, 4),
            baseline_units=round(response.baseline_units, 1),
            observed_units=round(response.observed_units, 1),
            confidence_interval=interval,
            validation_status=response.validation_status,
            events_analysed=response.events_analysed,
            events=[
                {
                    "promotion_id": e.promotion_id,
                    "product_id": e.product_id,
                    "store_id": e.store_id,
                    "incremental_units": round(e.incremental_units, 1),
                    "incremental_profit": round(e.incremental_profit, 2),
                    "roi": round(e.roi, 3) if e.roi is not None else None,
                    "value_destroying": e.value_destroying,
                }
                for e in response.events
            ],
            segments=[
                {
                    "dimension": s.dimension,
                    "segment": s.segment,
                    "uplift_pct": round(s.uplift_pct, 4)
                    if s.uplift_pct is not None
                    else None,
                    "classification": s.classification,
                    "action": s.action,
                    "n_treated": s.n_treated,
                }
                for s in response.segments
            ],
            method_comparison=[
                {
                    "method": m.method,
                    "uplift_pct": round(m.uplift_pct, 4),
                    "eligible": m.eligible,
                }
                for m in response.comparison
            ],
        )

        warnings = list(response.warnings)
        if not response.is_causal:
            # Promoted to the front. A supervisor reading a truncated warning
            # list must not miss the one that says the number is not causal.
            warnings.insert(
                0,
                "CAUSAL VALIDATION FAILED. This figure is reported so it can be "
                "inspected, and must NOT be described as the effect the "
                "promotion caused.",
            )

        return ToolOutput(
            payload=output,
            provenance=ModelProvenance(
                model_name=response.model_name,
                model_version=response.model_version,
                dataset_version=response.dataset_version,
            ),
            # No `confidence` scalar. The honest expression of uncertainty here
            # is the measured interval on the effect, which is in the payload. A
            # 0-1 score would be an invented summary of a quantity that was
            # actually estimated.
            confidence=None,
            assumptions=list(response.assumptions),
            warnings=warnings,
        )


#: Service error codes to the tool taxonomy.
_ERROR_CODES = {
    "model_not_found": ToolErrorCode.MODEL_NOT_FOUND,
    "model_not_fitted": ToolErrorCode.MODEL_NOT_FOUND,
    "insufficient_data": ToolErrorCode.INSUFFICIENT_DATA,
    "no_control_group": ToolErrorCode.INSUFFICIENT_DATA,
    "assumptions_violated": ToolErrorCode.INSUFFICIENT_DATA,
    "invalid_treatment": ToolErrorCode.INVALID_INPUT,
    "invalid_input": ToolErrorCode.INVALID_INPUT,
    "uplift_failed": ToolErrorCode.INTERNAL_ERROR,
}


__all__ = ["PromoUpliftInput", "PromoUpliftOutput", "PromoUpliftTool"]
