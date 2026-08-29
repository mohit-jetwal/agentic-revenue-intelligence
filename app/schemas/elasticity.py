"""Elasticity request/response contracts.

Shaped like :mod:`app.schemas.forecast` and :mod:`app.schemas.promo_uplift` so
the API, the tool layer and any future agent see one house style.

The field that does not appear in the other two: ``method`` is **required**, not
optional. An elasticity of -2.5 estimated by naive OLS and one estimated by panel
fixed effects are not the same claim - on this data the naive estimator recovers
only about 56% of the true elasticity - and a number that travels without saying
how it was identified invites exactly the wrong pricing decision.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ElasticityRequest(BaseModel):
    """What to estimate elasticity for."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    store_ids: list[str] | None = None
    region: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    #: Include the full method comparison. Useful for a report, noisy for an
    #: agent that only needs the number it should act on.
    include_comparison: bool = False
    #: Also estimate cross-price relationships for this product.
    include_cross_price: bool = False
    max_candidates: int = Field(default=40, gt=0, le=200)


class MethodEstimate(BaseModel):
    """One estimator's line in the comparison."""

    method: str
    elasticity: float
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_obs: int = 0
    #: False for estimators known to be biased on this data. They are reported
    #: for contrast, never selected.
    selectable: bool = True


class CrossPriceRecord(BaseModel):
    """One directed cross-price relationship."""

    source_product_id: str
    cross_elasticity: float
    relationship_type: str
    strength: str | None = None
    p_value: float | None = None
    is_significant: bool = False
    sample_size: int = 0


class ElasticityResponse(BaseModel):
    """An elasticity estimate with the evidence behind it."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "success"
    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str

    product_id: str
    region: str | None = None

    elasticity: float
    #: |e| > 1. This is the flag a pricing decision turns on: elastic demand
    #: means a price rise reduces revenue.
    is_elastic: bool
    confidence_interval: tuple[float, float] | None = None
    standard_error: float | None = None
    p_value: float | None = None
    r_squared: float | None = None
    sample_size: int = 0

    #: How the estimate was identified. Required - see the module docstring.
    method: str
    method_reason: str = ""
    estimation_window: tuple[date, date] | None = None

    comparison: list[MethodEstimate] = Field(default_factory=list)
    cross_price: list[CrossPriceRecord] = Field(default_factory=list)
    substitutes: list[str] = Field(default_factory=list)
    complements: list[str] = Field(default_factory=list)
    pairs_tested: int = 0

    diagnostics: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_ms: int = 0

    @property
    def revenue_direction(self) -> str:
        """What a price rise does to revenue. The business reading of |e|."""
        return "reduces revenue" if self.is_elastic else "raises revenue"


class ElasticityErrorResponse(BaseModel):
    """A structured refusal an agent can re-plan around."""

    status: str = "error"
    error_code: str
    message: str
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int = 0


__all__ = [
    "CrossPriceRecord",
    "ElasticityErrorResponse",
    "ElasticityRequest",
    "ElasticityResponse",
    "MethodEstimate",
]
