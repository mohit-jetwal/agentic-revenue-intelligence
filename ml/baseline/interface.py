"""Baseline sales model interface (Stage 1 Step 4).

Estimates what sales *would have been* absent promotions and abnormal events.

This is the most reused model in the platform: promotion uplift is measured
against it, root cause analysis compares actual to it, and the scenario engine
starts from it. Getting it wrong propagates everywhere, which is why it is built
first and why its output carries an interval rather than a point estimate.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from ml.base import AnalyticalModel


class BaselineResult(BaseModel):
    """Expected vs actual sales for one slice."""

    product_id: str | None = None
    store_id: str | None = None
    region: str | None = None
    start_date: date
    end_date: date

    baseline_units: float = Field(ge=0)
    baseline_revenue: float = Field(ge=0)
    actual_units: float = Field(ge=0)
    actual_revenue: float = Field(ge=0)

    #: actual - baseline. Negative means underperformance.
    units_gap: float
    revenue_gap: float
    #: Gap as a fraction of baseline, e.g. -0.12 for a 12% shortfall.
    revenue_gap_pct: float

    baseline_lower: float | None = None
    baseline_upper: float | None = None
    #: True when the gap falls outside the prediction interval - i.e. the
    #: shortfall is larger than normal variation and worth investigating.
    is_significant: bool = False


class BaselineSalesModel(AnalyticalModel[BaselineResult]):
    """Expected-sales estimator."""

    name = "baseline_sales"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
    ) -> BaselineResult:
        """Estimate baseline and compare it to actuals for the slice."""
