"""The elasticity tool.

Answers "how does demand respond to price" and "what does this product compete
with". The contract a future Claude agent calls for any pricing question.

**What the agent must take from this and what it must not.** The number to act
on is ``is_elastic``, not the coefficient: elastic demand (|e| > 1) means a price
rise *reduces* revenue, and that single bit determines the direction of every
pricing recommendation. The coefficient's magnitude is a local approximation
around the observed price range and does not extrapolate to a price nobody has
charged.

The ``method`` field is required in the output for a reason. On this data the
naive estimator recovers only about 56% of the true elasticity — it makes
products look less price-sensitive than they are, which encourages exactly the
wrong recommendation. An elasticity that travels without saying how it was
identified is not evidence.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.observability.logging import get_logger
from app.schemas.elasticity import ElasticityErrorResponse, ElasticityRequest
from app.schemas.tool_contract import ModelProvenance, ToolErrorCode
from app.services.elasticity_service import ElasticityService
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput

logger = get_logger(__name__)


class ElasticityInput(BaseModel):
    """What the agent asks for."""

    model_config = ConfigDict(frozen=True)

    product_id: str = Field(description="Product to estimate price elasticity for.")
    region: str | None = Field(default=None, description="Restrict to one region.")
    store_id: str | None = Field(default=None, description="Restrict to one store.")
    start_date: date | None = Field(default=None, description="Start of the estimation window.")
    end_date: date | None = Field(default=None, description="End of the estimation window.")
    include_cross_price: bool = Field(
        default=False,
        description=(
            "Also return substitutes and complements. Use this for "
            "cannibalisation and assortment questions."
        ),
    )
    include_comparison: bool = Field(
        default=False,
        description=(
            "Return every estimator side by side, including ones known to be "
            "biased. Use only when the question is about method, not price."
        ),
    )


class ElasticityOutput(BaseModel):
    """What the agent gets back."""

    model_config = ConfigDict(protected_namespaces=())

    product_id: str
    elasticity: float
    #: The bit a pricing decision turns on.
    is_elastic: bool
    #: Plain-language reading: "a price rise reduces revenue" / "raises revenue".
    revenue_direction: str

    confidence_interval: dict[str, float] | None = None
    p_value: float | None = None
    sample_size: int = 0

    method: str
    method_reason: str = ""

    substitutes: list[str] = Field(default_factory=list)
    complements: list[str] = Field(default_factory=list)
    cross_price: list[dict[str, object]] = Field(default_factory=list)
    pairs_tested: int = 0

    comparison: list[dict[str, object]] = Field(default_factory=list)


class ElasticityTool(AnalyticalTool[ElasticityInput, ElasticityOutput]):
    """Estimate how demand responds to price."""

    name = "estimate_price_elasticity"
    description = (
        "Estimate own-price elasticity for a product: the percentage change in "
        "demand per percentage change in price. Returns the coefficient, whether "
        "demand is ELASTIC (|e| > 1, meaning a price rise reduces revenue) or "
        "inelastic (a price rise raises revenue), a confidence interval, and the "
        "method used to identify it. Optionally returns substitutes and "
        "complements for cannibalisation and assortment questions. Use this for "
        "'should we raise the price', 'how price-sensitive is this product', and "
        "'what does this compete with'. The elasticity is a local approximation "
        "around observed prices and does not extrapolate to prices never charged. "
        "Do NOT infer elasticity yourself from a price and volume change - price "
        "moves with demand, so the naive calculation understates sensitivity."
    )
    input_schema = ElasticityInput
    output_schema = ElasticityOutput
    permission = "run_model"
    timeout_seconds = 120.0

    def __init__(self, service: ElasticityService) -> None:
        self._service = service

    def _execute(self, payload: ElasticityInput) -> ToolOutput[ElasticityOutput]:
        response = self._service.estimate(
            ElasticityRequest(
                product_id=payload.product_id,
                store_ids=[payload.store_id] if payload.store_id else None,
                region=payload.region,
                start_date=payload.start_date,
                end_date=payload.end_date,
                include_comparison=payload.include_comparison,
                include_cross_price=payload.include_cross_price,
            )
        )

        # `isinstance` rather than a status-string check: it narrows the union
        # for the type checker as well as at runtime.
        if isinstance(response, ElasticityErrorResponse):
            raise ToolExecutionError(
                response.message,
                code=_ERROR_CODES.get(response.error_code, ToolErrorCode.INTERNAL_ERROR),
                recoverable=response.recoverable,
                detail=response.detail,
            )

        interval = None
        if response.confidence_interval is not None:
            interval = {
                "lower": round(response.confidence_interval[0], 4),
                "upper": round(response.confidence_interval[1], 4),
            }

        output = ElasticityOutput(
            product_id=response.product_id,
            elasticity=round(response.elasticity, 4),
            is_elastic=response.is_elastic,
            revenue_direction=response.revenue_direction,
            confidence_interval=interval,
            p_value=response.p_value,
            sample_size=response.sample_size,
            method=response.method,
            method_reason=response.method_reason,
            substitutes=list(response.substitutes),
            complements=list(response.complements),
            cross_price=[
                {
                    "product_id": record.source_product_id,
                    "cross_elasticity": round(record.cross_elasticity, 4),
                    "relationship": record.relationship_type,
                    "strength": record.strength,
                    "significant": record.is_significant,
                }
                for record in response.cross_price
                if record.is_significant
            ],
            pairs_tested=response.pairs_tested,
            comparison=[
                {
                    "method": m.method,
                    "elasticity": round(m.elasticity, 4),
                    "selectable": m.selectable,
                }
                for m in response.comparison
            ],
        )

        return ToolOutput(
            payload=output,
            provenance=ModelProvenance(
                model_name=response.model_name,
                model_version=response.model_version,
                dataset_version=response.dataset_version,
            ),
            # No confidence scalar. The measured expression of uncertainty is
            # the interval on the coefficient, which is in the payload; a 0-1
            # score would be an invented summary of something already estimated.
            confidence=None,
            assumptions=list(response.assumptions),
            warnings=list(response.warnings),
        )


#: Service error codes to the tool taxonomy.
_ERROR_CODES = {
    "insufficient_data": ToolErrorCode.INSUFFICIENT_DATA,
    "invalid_input": ToolErrorCode.INVALID_INPUT,
    "elasticity_failed": ToolErrorCode.INTERNAL_ERROR,
}


__all__ = ["ElasticityInput", "ElasticityOutput", "ElasticityTool"]
