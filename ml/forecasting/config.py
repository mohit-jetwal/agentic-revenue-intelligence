"""Forecasting configuration (brief section 36).

Same pattern as ``features.contracts.config``: values in YAML, shape enforced by
Pydantic, so a typo fails on load with a field path rather than as a missing
column three joins later.

Two jobs beyond validation:

* **Reproducibility.** :meth:`ForecastConfig.fingerprint` hashes the whole
  configuration into one short string that goes into MLflow params. Two runs
  with the same fingerprint used the same setup; two runs with different
  fingerprints are not comparable, and the difference is discoverable rather
  than argued about.
* **Declaring the cost.** The defaults are chosen so a full run lands under
  ~25 minutes. Step 4 showed that when iteration takes hours, correctness bugs
  survive - so the sampling defaults here are a correctness measure, not a
  convenience.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "forecasting.yaml"


class ForecastConfigError(ValueError):
    """Raised when the forecasting configuration is internally inconsistent."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplingConfig(_Base):
    n_series: int = Field(default=800, gt=0)
    stratify_by_volume: bool = True
    seed: int = 42


class OriginConfig(_Base):
    stride_days: int = Field(default=7, gt=0)
    horizons_per_origin: int = Field(default=8, gt=0)
    warmup_days: int = Field(default=400, ge=0)


class TargetHandlingConfig(_Base):
    exclude_stockout_targets: bool = True
    exclude_stockout_origins: bool = False
    mask_censored_lags: bool = True


class ModelsConfig(_Base):
    horizon_seasonal_naive: bool = True
    horizon_naive: bool = True
    lightgbm: bool = True
    xgboost: bool = True
    statistical: bool = True

    def enabled(self) -> tuple[str, ...]:
        """Estimator names to train, simplest first.

        ``statistical`` is deliberately absent: it is fitted at aggregate grain
        by a separate path, not as a candidate in the product-store comparison.
        """
        order = ("horizon_naive", "horizon_seasonal_naive", "lightgbm", "xgboost")
        return tuple(name for name in order if getattr(self, name))


class ValidationConfig(_Base):
    method: str = "expanding_window"
    n_folds: int = Field(default=3, gt=0)
    test_days: int = Field(default=120, gt=0)
    valid_days: int = Field(default=90, gt=0)
    calibration_days: int = Field(default=60, gt=0)
    embargo_days: int = Field(default=90, ge=0)


class IntervalConfig(_Base):
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    horizon_buckets: tuple[tuple[int, int], ...] = (
        (1, 3), (4, 7), (8, 14), (15, 28), (29, 56), (57, 90),
    )
    calibration_origin_spacing_days: int = Field(default=7, ge=0)

    def bucket_for(self, horizon_step: int) -> str:
        """Label the bucket a horizon step falls in.

        Steps beyond the last bucket fall into it rather than producing a
        ``None`` label - an uncalibrated interval is worse than a slightly
        mis-bucketed one, and the horizon is bounded by config anyway.
        """
        for low, high in self.horizon_buckets:
            if low <= horizon_step <= high:
                return f"h{low}-{high}"
        low, high = self.horizon_buckets[-1]
        return f"h{low}-{high}"

    def bucket_labels(self) -> tuple[str, ...]:
        return tuple(f"h{low}-{high}" for low, high in self.horizon_buckets)


class StatisticalConfig(_Base):
    frequency: str = "weekly"
    levels: tuple[str, ...] = ("total", "region", "category", "category_region")
    bottom_sample_series: int = Field(default=50, ge=0)


def _default_search_space() -> dict[str, list[Any]]:
    """Fallback space when the YAML omits one.

    A named function rather than a lambda so the annotated return type reaches
    the type checker - an inline lambda infers `dict[str, object]` and fails to
    match the field.
    """
    return {
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "feature_fraction": [0.7, 0.85, 1.0],
        "bagging_fraction": [0.7, 0.85, 1.0],
        "lambda_l2": [0.0, 1.0, 5.0, 20.0],
    }


class TuningConfig(_Base):
    """Hyperparameter search settings.

    Small on purpose - see ``ml/forecasting/tuning.py`` for why a large sweep
    would be worse than none on this data.
    """

    n_trials: int = Field(default=20, gt=0, le=200)
    min_improvement_pp: float = Field(default=0.005, ge=0.0)
    space: dict[str, list[Any]] = Field(default_factory=_default_search_space)


class ExplainabilityConfig(_Base):
    top_n_features: int = Field(default=40, gt=0)
    permutation_repeats: int = Field(default=3, gt=0)


class ForecastConfig(_Base):
    """The whole forecasting configuration."""

    target: str = "units"
    grain: tuple[str, ...] = ("date", "product_id", "store_id")
    forecast_horizons: tuple[int, ...] = (7, 14, 28, 30, 90)
    max_horizon: int = Field(default=90, gt=0)

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    origins: OriginConfig = Field(default_factory=OriginConfig)
    target_handling: TargetHandlingConfig = Field(default_factory=TargetHandlingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    intervals: IntervalConfig = Field(default_factory=IntervalConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> ForecastConfig:
        if max(self.forecast_horizons) > self.max_horizon:
            raise ForecastConfigError(
                f"forecast_horizons reaches {max(self.forecast_horizons)} but "
                f"max_horizon is {self.max_horizon}; the model would never have "
                f"been trained at the longest horizon it is asked to serve"
            )
        # The buckets must span 1..max_horizon, or some horizon steps get their
        # interval from a bucket calibrated on different data.
        covered = self.intervals.horizon_buckets
        if covered[0][0] != 1:
            raise ForecastConfigError(
                f"horizon buckets start at {covered[0][0]}, not 1"
            )
        if covered[-1][1] < self.max_horizon:
            raise ForecastConfigError(
                f"horizon buckets stop at {covered[-1][1]} but max_horizon is "
                f"{self.max_horizon}; the longest horizons would be uncalibrated"
            )
        for (_, previous_high), (low, _) in pairwise(covered):
            if low != previous_high + 1:
                raise ForecastConfigError(
                    f"horizon buckets are not contiguous around {previous_high}/{low}"
                )
        if self.validation.embargo_days < self.max_horizon:
            raise ForecastConfigError(
                f"embargo is {self.validation.embargo_days} days but horizons "
                f"reach {self.max_horizon}; a training origin near the boundary "
                f"would have its target inside the evaluation window"
            )
        return self

    def fingerprint(self) -> str:
        """Short stable hash of the whole configuration, for MLflow params."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def smoke(self) -> ForecastConfig:
        """A tiny variant for tests and correctness checks.

        Fifty series and a fortnightly stride runs in under a minute, which is
        what makes it usable inside the test suite. Everything structural -
        splits, embargo, buckets, censoring - is unchanged, so a bug in that
        machinery still surfaces here rather than only in the full run.
        """
        return self.model_copy(
            update={
                "sampling": self.sampling.model_copy(update={"n_series": 50}),
                "origins": self.origins.model_copy(
                    # A wide stride keeps the expensive part cheap - building the
                    # feature history dominates the runtime - while drawing many
                    # horizons per origin keeps the *rows* plentiful, which is
                    # what the per-bucket assertions need. Four horizons per
                    # origin left ~60 rows in the shortest bucket, few enough
                    # that the horizon-monotonicity check flipped between runs
                    # on noise alone.
                    update={"stride_days": 14, "horizons_per_origin": 12}
                ),
                "validation": self.validation.model_copy(update={"n_folds": 2}),
            }
        )


def load_forecast_config(path: Path | None = None) -> ForecastConfig:
    """Read and validate the forecasting configuration."""
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        raise ForecastConfigError(f"no forecasting config at {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        return ForecastConfig.model_validate(raw)
    except ForecastConfigError:
        raise
    except ValueError as exc:
        raise ForecastConfigError(f"invalid forecasting config at {config_path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_forecast_config() -> ForecastConfig:
    """Process-wide configuration, read once."""
    return load_forecast_config()


def reset_forecast_config_cache() -> None:
    """Clear the cache. Intended for tests."""
    get_forecast_config.cache_clear()
