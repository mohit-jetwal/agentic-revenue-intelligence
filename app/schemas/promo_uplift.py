"""Promo uplift request/response contracts (brief section 28).

Shaped like :mod:`app.schemas.forecast` so the API, the tool layer and any
future agent see one house style. Three fields exist here that a forecast
response has no need for, and each carries something a causal number cannot
travel without.

``validation_status`` - whether the causal assumptions held. A forecast is more
or less accurate; a causal estimate is either identified or it is not, and that
is a different kind of statement. A caller must be able to tell "we measured
+18%" from "we computed +18% but the design does not support calling it causal".

``assumptions`` - not a disclaimer. These are the conditions under which the
number *is* the causal effect, and they are the first thing a reviewer should
attack.

``confidence_interval`` - present only when it was computed. Never fabricated,
same rule as ``confidence`` in the forecast contract. An estimator without a
usable variance returns ``None``, and the response says so rather than emitting
a plausible-looking band.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UpliftRequest(BaseModel):
    """What to estimate uplift for."""

    model_config = ConfigDict(frozen=True)

    promotion_ids: list[str] | None = None
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None
    category: str | None = None

    analysis_start_date: date | None = None
    analysis_end_date: date | None = None

    #: Include the post-promotion washout, giving net incrementality rather than
    #: the gross effect during the event. Net is the honest default for a
    #: profitability question; gross is what most reporting shows.
    include_pull_forward: bool = True
    include_segments: bool = True
    include_events: bool = True
    max_events: int = Field(default=500, gt=0, le=10_000)


class UpliftIntervalRecord(BaseModel):
    """A measured interval. Absent rather than invented."""

    lower: float
    upper: float
    confidence_level: float


class UpliftEventRecord(BaseModel):
    """One promotion's estimated impact."""

    promotion_id: str
    product_id: str
    store_id: str
    treated_days: int
    incremental_units: float
    incremental_revenue: float
    incremental_profit: float
    promotion_spend: float | None = None
    roi: float | None = None
    value_destroying: bool = False


class UpliftSegmentRecord(BaseModel):
    """Conditional effect for one segment."""

    segment: str
    dimension: str
    n_treated: int
    uplift_pct: float | None = None
    classification: str = "uncertain"
    action: str = ""
    estimable: bool = True


class MethodComparisonRecord(BaseModel):
    """One estimator's line in the comparison."""

    method: str
    uplift_pct: float
    ci_lower_pct: float | None = None
    ci_upper_pct: float | None = None
    incremental_units: float | None = None
    incremental_profit: float | None = None
    roi: float | None = None
    validation_status: str = "not_assessed"
    eligible: bool = True


class UpliftDiagnostics(BaseModel):
    """The evidence that the estimate is causal, or is not."""

    overlap_trimmed_share: float | None = None
    effective_sample_fraction: float | None = None
    max_standardised_difference: float | None = None
    unbalanced_covariates: int = 0
    placebo_effect_pct: float | None = None
    placebo_passed: bool | None = None
    sensitivity_spread_pct: float | None = None
    parallel_trends_p: float | None = None
    parallel_trends_passed: bool | None = None
    propensity_auc: float | None = None


class UpliftResponse(BaseModel):
    """A causal estimate, with everything needed to judge it."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = "success"
    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str

    #: The estimand, in one sentence. An uplift number is uninterpretable
    #: without it, so it is required rather than optional.
    treatment_definition: str
    method: str
    method_reason: str = ""

    baseline_units: float
    observed_units: float
    incremental_units: float
    uplift_pct: float
    incremental_revenue: float
    incremental_profit: float
    promotion_spend: float = 0.0
    roi: float | None = None

    confidence_interval: UpliftIntervalRecord | None = None
    #: Gross effect during the event, before pull-forward is paid back. Reported
    #: alongside the net figure so the difference between "sold more this week"
    #: and "sold more overall" is visible rather than a matter of which window
    #: someone chose.
    gross_uplift_pct: float | None = None
    pull_forward_units: float | None = None

    events_analysed: int = 0
    treated_days: int = 0
    control_days: int = 0

    events: list[UpliftEventRecord] = Field(default_factory=list)
    segments: list[UpliftSegmentRecord] = Field(default_factory=list)
    comparison: list[MethodComparisonRecord] = Field(default_factory=list)
    diagnostics: UpliftDiagnostics = Field(default_factory=UpliftDiagnostics)

    #: passed | warnings | failed. See the module docstring.
    validation_status: str = "not_assessed"
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    execution_time_ms: int = 0

    @property
    def is_causal(self) -> bool:
        """Whether this may be presented as a causal effect.

        A ``failed`` status still returns the number - suppressing it would not
        stop anyone computing a worse one - but it must not be described as
        causal, and this is the property that decides.
        """
        return self.validation_status in {"passed", "warnings"}


class UpliftErrorResponse(BaseModel):
    """A structured refusal an agent can re-plan around."""

    status: str = "error"
    error_code: str
    message: str
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int = 0


__all__ = [
    "MethodComparisonRecord",
    "UpliftDiagnostics",
    "UpliftErrorResponse",
    "UpliftEventRecord",
    "UpliftIntervalRecord",
    "UpliftRequest",
    "UpliftResponse",
    "UpliftSegmentRecord",
]
