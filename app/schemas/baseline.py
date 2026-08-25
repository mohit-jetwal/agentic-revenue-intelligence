"""Baseline service request/response contracts (brief sections 17, 33).

Shaped so the Step 13 tool layer can wrap it without translation: the response
carries the same provenance fields as
:class:`~app.schemas.tool_contract.ToolResult` - model name, model version,
dataset version, assumptions, warnings, execution time - because a number
reaching Claude has to arrive with its lineage attached or it cannot be
attributed.

Deliberately *not* a ``ToolResult`` yet. Step 13 owns the tool layer, and
building the wrapper here would put agent concerns inside a service that has no
business knowing about agents.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.baseline.interface import BaselineResult


class BaselineRequest(BaseModel):
    """A baseline query (brief section 33)."""

    model_config = ConfigDict(frozen=True)

    start_date: date
    end_date: date
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None
    #: Reproduce what the model would have said on this date. Defaults to
    #: ``end_date`` - a baseline for a past window normally uses everything
    #: known by the end of it.
    as_of_date: date | None = None
    #: Return per-row records as well as the aggregate. Off by default: a
    #: multi-month, multi-product slice is tens of thousands of rows, and an
    #: agent asking "what was the baseline" wants the number, not the panel.
    include_records: bool = False
    max_records: int = Field(default=5_000, gt=0, le=100_000)


class BaselineRecord(BaseModel):
    """One product-store-day baseline (brief section 17)."""

    model_config = ConfigDict(protected_namespaces=())

    date: date
    product_id: str
    store_id: str

    actual_units: float
    baseline_units: float
    sales_gap: float
    sales_gap_pct: float | None = None

    baseline_lower: float | None = None
    baseline_upper: float | None = None
    #: The actual falls outside the prediction interval, so the gap exceeds the
    #: model's normal error. Only meaningful because coverage was measured.
    is_significant: bool | None = None

    promotion_flag: bool | None = None
    stockout_flag: bool | None = None
    #: True when the cold-start fallback produced this row rather than the model.
    fallback_used: bool = False


class BaselineMetricsSummary(BaseModel):
    """Accuracy of the model behind this response, on its test window.

    Reported with every response so a caller can weigh the number. A baseline
    with 30% WMAPE and one with 8% support very different conclusions, and
    hiding that distinction behind a bare figure invites over-confidence.
    """

    test_wmape: float | None = None
    test_mae: float | None = None
    test_bias_pct: float | None = None
    interval_coverage: float | None = None
    interval_nominal: float | None = None
    backtest_mean_wmape: float | None = None
    backtest_stable: bool | None = None


class BaselineResponse(BaseModel):
    """Structured baseline output (brief section 33)."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "success"

    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str

    #: Slice-level result - the Step 1 contract other models consume.
    result: BaselineResult | None = None
    #: Per-row detail, when requested.
    records: list[BaselineRecord] = Field(default_factory=list)
    record_count: int = 0
    #: Rows that used the cold-start fallback.
    fallback_rows: int = 0

    metrics: BaselineMetricsSummary = Field(default_factory=BaselineMetricsSummary)

    #: Modelling assumptions a caller must carry into any conclusion. Populated
    #: from what the model actually did, not written by hand.
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    execution_time_ms: int = 0

    def summary(self) -> str:
        if self.result is None:
            return f"{self.status}: no result"
        r = self.result
        return (
            f"{r.start_date}..{r.end_date}: actual {r.actual_units:,.0f} vs baseline "
            f"{r.baseline_units:,.0f} units (gap {r.units_gap:+,.0f}, "
            f"{r.revenue_gap_pct:+.1%} revenue)"
        )


class BaselineErrorResponse(BaseModel):
    """Failure, in the same shape so a caller needs one branch, not two."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "error"
    error_code: str
    message: str
    #: Whether a different request could succeed - drives re-planning in Step 16.
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int = 0
