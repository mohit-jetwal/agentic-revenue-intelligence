"""Typed failures for the promo uplift capability.

Same design as :mod:`ml.forecasting.exceptions`, and for the same reason: a
service deciding whether a failure is *recoverable* should read a class
attribute, not a message string.

**The inheritance is load-bearing.** Everything that means "the data cannot
support this estimate" subclasses :class:`ml.base.InsufficientDataError`, and
:class:`UpliftModelUnavailableError` subclasses
:class:`ml.base.ModelNotFittedError`. The service maps them to error codes
through those two base classes, so a new subclass here is routed correctly
without touching the service.

One class deserves its own note. :class:`CausalAssumptionsViolatedError` is the
failure this capability exists to be able to raise. A promo uplift model that
never refuses is not a causal model - it is a regression with a causal label on
it. When overlap fails, or the treated and control groups are not comparable
after adjustment, the honest answer is "this design does not identify the
effect", not a number with a wide interval. The estimate that *would* have been
produced rides along in ``detail`` so a human can look at it, clearly marked as
unidentified.
"""

from __future__ import annotations

from typing import Any

from ml.base import InsufficientDataError, ModelNotFittedError


class PromoUpliftError(Exception):
    """Base for every promo uplift failure.

    ``recoverable`` means: could a *different request* succeed against the same
    system? A narrower date range, a coarser grain or a different promotion
    might; a missing baseline model never will until someone trains one.
    """

    #: Stable identifier surfaced to callers and agents.
    code: str = "uplift_failed"
    #: Whether re-planning could succeed. See the class docstring.
    recoverable: bool = True

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, for a structured error response."""
        return {
            "error_code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "detail": self.detail,
        }


class NoControlGroupError(PromoUpliftError, InsufficientDataError):
    """No usable control observations for the requested treatment.

    Causal inference without a control group is not possible - there is nothing
    to compare against, so any "uplift" would be a comparison of the treated
    period against itself.

    This happens for real: a product promoted continuously for the whole window
    has no unpromoted days of its own, and if every store ran the same promotion
    there is no cross-sectional control either. Widening the date range or
    dropping to a category-level comparison can recover it, which is why this is
    recoverable and why ``detail`` reports how many controls *were* found.
    """

    code = "no_control_group"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        treated_rows: int | None = None,
        control_rows: int | None = None,
        required_control_rows: int | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message,
            treated_rows=treated_rows,
            control_rows=control_rows,
            required_control_rows=required_control_rows,
            **detail,
        )


class InsufficientPrePeriodError(PromoUpliftError, InsufficientDataError):
    """Not enough pre-treatment history to build covariates or test trends.

    Every covariate this model conditions on is measured *before* the promotion
    starts. A promotion beginning two weeks after a product launched has no
    28-day trailing demand, no prior promotion frequency and no pre-trend to
    test - so the adjustment set is mostly missing and the parallel-trends check
    cannot run at all.

    Returning an estimate anyway would mean silently conditioning on whatever
    happened to be non-null, which varies per promotion and makes the estimates
    non-comparable to each other.
    """

    code = "insufficient_data"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        available_days: int | None = None,
        required_days: int | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message, available_days=available_days, required_days=required_days, **detail
        )


class UnknownPromotionError(PromoUpliftError, InsufficientDataError):
    """The requested promotion, product or store is not in the data.

    Refused rather than answered with a zero uplift. "This promotion had no
    effect" and "this promotion does not exist" are different findings, and a
    category manager acting on the first when the second is true would conclude
    a mechanic does not work on no evidence at all.
    """

    code = "insufficient_data"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        promotion_ids: list[str] | None = None,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message,
            promotion_ids=promotion_ids,
            product_ids=product_ids,
            store_ids=store_ids,
            **detail,
        )


class CausalAssumptionsViolatedError(PromoUpliftError):
    """The design does not identify a causal effect on this data.

    Raised when a diagnostic fails hard rather than merely warns: overlap is
    violated after trimming, covariate balance is still poor after weighting, or
    the placebo test finds an effect where none can exist.

    The estimate that would have been returned is carried in ``detail`` under
    ``unidentified_estimate``. Suppressing it entirely would be its own kind of
    dishonesty - the caller can compute a naive number in one line of SQL, so
    withholding ours does not protect them. Labelling it does.
    """

    code = "assumptions_violated"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        assumption: str | None = None,
        diagnostic: str | None = None,
        observed: float | None = None,
        threshold: float | None = None,
        **detail: Any,
    ) -> None:
        super().__init__(
            message,
            assumption=assumption,
            diagnostic=diagnostic,
            observed=observed,
            threshold=threshold,
            **detail,
        )


class OverlapViolationError(CausalAssumptionsViolatedError):
    """Treated and control units do not share covariate support.

    Positivity - every unit having a non-zero probability of either arm - is
    what makes a comparison possible at all. Where it fails, the estimator is
    extrapolating the outcome model into a region with no data from one arm, and
    inverse-probability weights explode: a propensity of 0.001 gives one
    observation a weight of 1000 and lets it dominate the entire estimate.

    A distinct class because the remedy is specific - trim, or restrict to the
    region of common support and say the estimand changed.
    """

    code = "assumptions_violated"
    recoverable = True


class UpliftModelUnavailableError(PromoUpliftError, ModelNotFittedError):
    """A required model is missing or unusable.

    Not recoverable: no reformulation of the request helps until someone trains
    one. Most often this is the *baseline* model rather than an uplift model -
    the baseline counterfactual method reuses Step 5's artifact, so an
    untrained baseline blocks that estimator specifically. The message names
    which model and the command that produces it.
    """

    code = "model_not_found"
    recoverable = False

    def __init__(self, message: str, *, model: str | None = None, **detail: Any) -> None:
        super().__init__(message, model=model, **detail)


class TreatmentDefinitionError(PromoUpliftError):
    """The treatment definition is inconsistent with the data.

    Overlapping promotions on one product-store-day, a promotion flagged with no
    discount, or a configured minimum depth that excludes every event. All make
    the treatment indicator ambiguous, and an ambiguous treatment makes the
    effect it identifies undefined rather than merely imprecise.
    """

    code = "invalid_treatment"
    recoverable = True

    def __init__(self, message: str, *, affected_rows: int | None = None, **detail: Any) -> None:
        super().__init__(message, affected_rows=affected_rows, **detail)


class InvalidUpliftRequestError(PromoUpliftError):
    """The request is malformed - a reversed date range, an unknown method."""

    code = "invalid_input"
    recoverable = True


class EstimationError(PromoUpliftError):
    """The estimator ran but could not produce a usable number.

    A singular design matrix, a propensity model that failed to converge, a
    bootstrap where every resample degenerated. Distinct from an assumptions
    violation: there the design is wrong, here the arithmetic failed.
    """

    code = "uplift_failed"
    recoverable = True

    def __init__(self, message: str, *, method: str | None = None, **detail: Any) -> None:
        super().__init__(message, method=method, **detail)


__all__ = [
    "CausalAssumptionsViolatedError",
    "EstimationError",
    "InsufficientPrePeriodError",
    "InvalidUpliftRequestError",
    "NoControlGroupError",
    "OverlapViolationError",
    "PromoUpliftError",
    "TreatmentDefinitionError",
    "UnknownPromotionError",
    "UpliftModelUnavailableError",
]
