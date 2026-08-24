"""Promotion uplift model interface (Stage 1 Step 6).

Estimates the *incremental* sales caused by a promotion.

The distinction this model exists to enforce: sales during a promotion are not
uplift. Promotions run when demand is already expected to be strong (seasonal
peaks, paydays), and they pull forward purchases that would have happened later.
A naive during-versus-before comparison captures both effects and reports them
as incremental, systematically overstating ROI. Hence an explicit baseline, an
explicit control group where one exists, and an explicit pull-forward term.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from app.schemas.domain import PromotionType
from ml.base import AnalyticalModel


class UpliftResult(BaseModel):
    """Incremental effect of a promotion, with its uncertainty."""

    promotion_id: str | None = None
    product_id: str | None = None
    store_id: str | None = None
    region: str | None = None
    promotion_type: PromotionType | None = None
    start_date: date
    end_date: date

    baseline_units: float = Field(ge=0)
    actual_units: float = Field(ge=0)
    incremental_units: float
    uplift_pct: float

    incremental_revenue: float
    incremental_profit: float
    promotion_spend: float = Field(ge=0)
    #: incremental_profit / promotion_spend. Below 1.0 means the promotion
    #: destroyed value even if uplift was positive.
    roi: float | None = None

    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None

    #: Units borrowed from future periods rather than genuinely new. Subtracted
    #: from incremental_units when estimable.
    pull_forward_units: float | None = None
    #: Volume lost on other products in the same category.
    cannibalisation_units: float | None = None

    method: str | None = None
    control_group_size: int | None = None
    treatment_group_size: int | None = None
    assumptions: list[str] = Field(default_factory=list)


class PromoUpliftModel(AnalyticalModel[UpliftResult]):
    """Causal estimator of promotional incrementality."""

    name = "promo_uplift"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        promotion_id: str | None = None,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> UpliftResult:
        """Estimate incremental sales attributable to the promotion."""
