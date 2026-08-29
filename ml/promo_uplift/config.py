"""Promo uplift configuration (brief sections 6, 7, 27).

Same pattern as :mod:`ml.forecasting.config`: values in YAML, shape enforced by
Pydantic, one hash into MLflow params.

There is a difference in *what* the configuration is for, though, and it matters.
A forecasting config tunes a predictor - change it and the forecast gets better
or worse. A causal config **defines the estimand**. Change ``washout_days`` from
0 to 10 and the number stops meaning "effect during the promotion" and starts
meaning "effect net of pull-forward". Both are legitimate; they are different
quantities, and an estimate is uninterpretable without knowing which one it is.

That is why :meth:`PromoUpliftConfig.treatment_definition` exists as a separate,
human-readable string. It is logged to MLflow, attached to every result, and
printed in the report - so an uplift number can never travel without the
definition that produced it.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "promo_uplift.yaml"


class PromoUpliftConfigError(ValueError):
    """Raised when the promo uplift configuration is internally inconsistent."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TreatmentConfig(_Base):
    """What counts as a promotion, and over what window."""

    min_discount_depth: float = Field(default=0.05, ge=0.0, lt=1.0)
    include_types: tuple[str, ...] = ()
    require_price_reduction: bool = False
    washout_days: int = Field(default=10, ge=0)
    min_duration_days: int = Field(default=2, gt=0)

    def describe(self) -> str:
        """One sentence a business reader can check the number against."""
        types = ", ".join(self.include_types) if self.include_types else "any mechanic"
        price = " with a price reduction" if self.require_price_reduction else ""
        return (
            f"treated = a promotion of {types}{price} with depth "
            f">= {self.min_discount_depth:.0%} running at least "
            f"{self.min_duration_days} days; effects measured over the event "
            f"window and a {self.washout_days}-day washout"
        )


class ControlsConfig(_Base):
    """Where the comparison observations come from."""

    same_series_window_days: int = Field(default=45, gt=0)
    use_cross_sectional_controls: bool = True
    min_control_rows: int = Field(default=30, gt=0)
    min_treated_rows: int = Field(default=5, gt=0)
    pre_period_days: int = Field(default=56, gt=0)

    def describe(self) -> str:
        source = "same listing" + (
            " plus never-treated listings in the same category and region"
            if self.use_cross_sectional_controls
            else ""
        )
        return (
            f"control = unpromoted days from the {source}, within "
            f"{self.same_series_window_days} days of the event and outside its "
            f"washout window"
        )


class StockoutConfig(_Base):
    """How censored outcomes are handled. See the YAML for why this is delicate."""

    exclude_censored_rows: bool = True
    differential_censoring_warn_pp: float = Field(default=0.05, ge=0.0, le=1.0)
    bracketing_sensitivity: bool = True


class PropensityConfig(_Base):
    """Treatment-assignment model and the overlap guardrails."""

    model: str = "logistic"
    clip: tuple[float, float] = (0.02, 0.98)
    max_trimmed_share: float = Field(default=0.10, ge=0.0, le=1.0)
    #: Percentile at which control weights are capped. ``None`` disables it. See
    #: :func:`ml.promo_uplift.propensity.att_weights` for the measurement that
    #: motivated a default rather than leaving weights raw.
    stabilise_weights_at: float | None = Field(default=99.0, gt=50.0, le=100.0)
    max_standardised_difference: float = Field(default=0.10, gt=0.0)
    min_effective_sample_fraction: float = Field(default=0.30, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_clip(self) -> PropensityConfig:
        low, high = self.clip
        if not 0.0 < low < high < 1.0:
            raise PromoUpliftConfigError(
                f"propensity.clip must satisfy 0 < low < high < 1; got {self.clip}"
            )
        return self


class EstimatorsConfig(_Base):
    """Which estimators run. Each is a different identifying assumption."""

    naive_during_vs_before: bool = True
    baseline_counterfactual: bool = True
    difference_in_differences: bool = True
    inverse_probability_weighting: bool = True
    augmented_ipw: bool = True
    dr_learner: bool = True

    def enabled(self) -> tuple[str, ...]:
        """Estimator names, weakest assumption set last.

        Ordered so the report reads as an argument: here is the naive number,
        here is what each additional adjustment does to it, here is the estimate
        that survives the most scrutiny.
        """
        order = (
            "naive_during_vs_before",
            "baseline_counterfactual",
            "difference_in_differences",
            "inverse_probability_weighting",
            "augmented_ipw",
            "dr_learner",
        )
        return tuple(name for name in order if getattr(self, name))


class OutcomeModelConfig(_Base):
    """Nuisance outcome model E[Y | X, T]."""

    transform: str = "log1p"
    estimator: str = "lightgbm"
    n_estimators: int = Field(default=300, gt=0)
    learning_rate: float = Field(default=0.05, gt=0.0)
    num_leaves: int = Field(default=31, gt=1)
    min_child_samples: int = Field(default=50, gt=0)
    seed: int = 42

    def params(self) -> dict[str, Any]:
        """Estimator keyword arguments, without the transform."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "random_state": self.seed,
        }


class CrossFittingConfig(_Base):
    n_folds: int = Field(default=5, gt=1)
    #: ``series`` holds out whole listings; ``time_blocks`` cuts contiguous date
    #: ranges. See :func:`ml.promo_uplift.estimators.assign_folds` for why the
    #: intuitive choice is the wrong default here.
    scheme: str = "series"

    @model_validator(mode="after")
    def _check_scheme(self) -> CrossFittingConfig:
        if self.scheme not in {"series", "time_blocks"}:
            raise PromoUpliftConfigError(
                f"cross_fitting.scheme must be 'series' or 'time_blocks'; "
                f"got {self.scheme!r}"
            )
        return self


class UncertaintyConfig(_Base):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    bootstrap_samples: int = Field(default=200, gt=0)
    bootstrap_unit: str = "series"
    seed: int = 42

    @property
    def confidence_level(self) -> float:
        return 1.0 - self.alpha


class ValidationConfig(_Base):
    placebo_shift_days: int = Field(default=30, gt=0)
    placebo_max_abs_effect: float = Field(default=0.05, ge=0.0)
    parallel_trends_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    sensitivity_washout_days: tuple[int, ...] = (0, 5, 10, 21)
    sensitivity_control_windows: tuple[int, ...] = (21, 45, 90)
    sensitivity_trim_levels: tuple[float, ...] = (0.01, 0.02, 0.05)
    stability_folds: int = Field(default=3, gt=0)


class BusinessConfig(_Base):
    margin_source: str = "sales_fact"
    default_gross_margin: float = Field(default=0.30, gt=0.0, lt=1.0)
    roi_break_even: float = 1.0


class SamplingConfig(_Base):
    n_series: int = Field(default=400, gt=0)
    stratify_by_volume: bool = True
    seed: int = 42
    max_events: int = Field(default=2000, gt=0)


class SyntheticConfig(_Base):
    n_series: int = Field(default=200, gt=0)
    n_days: int = Field(default=365, gt=0)
    seed: int = 42
    recovery_tolerance: float = Field(default=0.03, gt=0.0)


class PromoUpliftConfig(_Base):
    """The whole promo uplift configuration."""

    target: str = "units"
    grain: tuple[str, ...] = ("date", "product_id", "store_id")

    treatment: TreatmentConfig = Field(default_factory=TreatmentConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)
    stockouts: StockoutConfig = Field(default_factory=StockoutConfig)
    propensity: PropensityConfig = Field(default_factory=PropensityConfig)
    estimators: EstimatorsConfig = Field(default_factory=EstimatorsConfig)
    outcome_model: OutcomeModelConfig = Field(default_factory=OutcomeModelConfig)
    cross_fitting: CrossFittingConfig = Field(default_factory=CrossFittingConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> PromoUpliftConfig:
        # Covariates use trailing windows up to 56 days; a shorter pre-period
        # would leave the longest of them undefined for every promotion, so the
        # adjustment set would silently shrink rather than fail.
        if self.controls.pre_period_days < 56:
            raise PromoUpliftConfigError(
                f"controls.pre_period_days is {self.controls.pre_period_days} but "
                f"the pre-treatment covariates use a 56-day trailing window; the "
                f"longest of them would be undefined for every event"
            )
        # The control window has to clear the washout, or "control" days are
        # actually pull-forward days - depressed by the very promotion whose
        # effect they are supposed to anchor. That biases uplift upward.
        if self.controls.same_series_window_days <= self.treatment.washout_days:
            raise PromoUpliftConfigError(
                f"controls.same_series_window_days "
                f"({self.controls.same_series_window_days}) does not clear "
                f"treatment.washout_days ({self.treatment.washout_days}); control "
                f"rows would be drawn from the pull-forward dip"
            )
        # The placebo shift must land clear of the real event and its washout,
        # otherwise the "placebo" window contains genuine treatment effects and
        # the test is guaranteed to fail for the wrong reason.
        if self.validation.placebo_shift_days <= self.treatment.washout_days:
            raise PromoUpliftConfigError(
                f"validation.placebo_shift_days "
                f"({self.validation.placebo_shift_days}) is inside the "
                f"{self.treatment.washout_days}-day washout; the placebo window "
                f"would overlap real treatment effects"
            )
        if not self.estimators.enabled():
            raise PromoUpliftConfigError("no estimators are enabled")
        return self

    def treatment_definition(self) -> str:
        """The estimand, in one paragraph, for logging and for every result.

        Attached to results rather than kept in a document, because a number
        that travels without its definition will be compared against a number
        computed with a different one.
        """
        censoring = (
            "rows censored by stockout are excluded, so the estimand is the "
            "effect on sales among days where stock was available"
            if self.stockouts.exclude_censored_rows
            else "censored rows are retained"
        )
        return (
            f"{self.treatment.describe()}. "
            f"{self.controls.describe()}. "
            f"Estimand: ATT. {censoring}."
        )

    def fingerprint(self) -> str:
        """Short stable hash of the whole configuration, for MLflow params."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def smoke(self) -> PromoUpliftConfig:
        """A tiny variant for tests and correctness checks.

        Everything structural - the treatment windows, the control rules, the
        overlap guardrails, cross-fitting - is unchanged, so a bug in that
        machinery still surfaces here rather than only in a full run. Only the
        volume of data and the bootstrap count shrink.
        """
        return self.model_copy(
            update={
                "sampling": self.sampling.model_copy(
                    update={"n_series": 40, "max_events": 150}
                ),
                "synthetic": self.synthetic.model_copy(
                    update={"n_series": 60, "n_days": 240}
                ),
                "uncertainty": self.uncertainty.model_copy(
                    update={"bootstrap_samples": 40}
                ),
                "cross_fitting": self.cross_fitting.model_copy(update={"n_folds": 3}),
            }
        )


def load_promo_uplift_config(path: Path | None = None) -> PromoUpliftConfig:
    """Read and validate the promo uplift configuration."""
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise PromoUpliftConfigError(f"no promo uplift config at {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        return PromoUpliftConfig.model_validate(raw)
    except PromoUpliftConfigError:
        raise
    except ValueError as exc:
        raise PromoUpliftConfigError(
            f"invalid promo uplift config at {config_path}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_promo_uplift_config() -> PromoUpliftConfig:
    """Process-wide configuration, read once."""
    return load_promo_uplift_config()


def reset_promo_uplift_config_cache() -> None:
    """Clear the cache. Intended for tests."""
    get_promo_uplift_config.cache_clear()
