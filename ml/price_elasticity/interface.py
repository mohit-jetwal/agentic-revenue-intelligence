"""Own-price elasticity model interface (Stage 1 Step 8).

Estimates the percentage change in demand per percentage change in price.

The modelling hazard this interface is shaped around is price endogeneity.
Prices are not set at random - they are raised into strong demand and cut into
weak demand - so a naive regression of quantity on price recovers the pricing
manager's behaviour, not the consumer's. The bias is toward zero, which makes
products look less price-sensitive than they are and encourages exactly the
wrong recommendation. Hence ``method`` and ``diagnostics`` are part of the
result: how the estimate was identified is as important as its value.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from ml.base import AnalyticalModel


class ElasticityResult(BaseModel):
    """An own-price elasticity estimate with its diagnostics."""

    product_id: str
    region: str | None = None
    store_id: str | None = None
    segment: str | None = None

    #: Negative for a normal good. -1.42 means a 1% price rise is associated
    #: with roughly a 1.42% demand fall.
    elasticity: float
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    standard_error: float | None = None
    r_squared: float | None = None
    sample_size: int = Field(ge=0)

    #: Elastic demand (|e| > 1) means a price rise reduces revenue; inelastic
    #: means it raises revenue. This flag is what a pricing decision turns on.
    is_elastic: bool | None = None

    #: e.g. "log_log_ols", "panel_fe", "iv_2sls".
    method: str | None = None
    estimation_window: tuple[date, date] | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PriceElasticityModel(AnalyticalModel[ElasticityResult]):
    """Own-price elasticity estimator."""

    name = "price_elasticity"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        product_id: str,
        region: str | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ElasticityResult:
        """Estimate own-price elasticity for the product and slice."""
