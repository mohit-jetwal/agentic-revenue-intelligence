"""Loader for ``configs/features/features.yaml``.

Same pattern as the dataset profiles in Step 2: values in YAML, shape enforced
by Pydantic, so a typo fails on load with a field path rather than as a missing
column three joins later.

The one non-obvious job here is cross-checking the config against the catalogue.
A feature name that does not exist is caught at load time - otherwise the column
simply never appears in the training frame, the model trains on whatever *did*
arrive, and nothing complains.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import PROJECT_ROOT
from features.contracts.catalogue import FEATURE_SPECS, forward_looking_features

CONFIG_PATH = PROJECT_ROOT / "configs" / "features" / "features.yaml"


class FeatureConfigError(ValueError):
    """Raised when the feature configuration is inconsistent with the catalogue."""


class DatasetSelection(BaseModel):
    """Which feature groups one dataset builder uses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: tuple[str, ...]
    target: str | None = None
    include_promotion_spend: bool = True
    exclude_promotional_rows: bool = False
    exclude_stockout_rows: bool = False
    pre_period_days: int = 28
    post_period_days: int = 14
    grain: str = "store"


class ValidationRules(BaseModel):
    """Guardrails applied to any built feature frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_forward_looking: tuple[str, ...] = ()
    max_null_rate: float = Field(default=0.60, ge=0.0, le=1.0)
    forbidden_columns: tuple[str, ...] = ()


class FeatureConfig(BaseModel):
    """The whole feature configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_version: str
    groups: dict[str, tuple[str, ...]]
    datasets: dict[str, DatasetSelection]
    validation: ValidationRules

    @model_validator(mode="after")
    def _check_against_catalogue(self) -> FeatureConfig:
        unknown = {
            name
            for features in self.groups.values()
            for name in features
            if name not in FEATURE_SPECS
        }
        if unknown:
            raise FeatureConfigError(
                f"features.yaml names features absent from the catalogue: "
                f"{sorted(unknown)}. Add them to features/contracts/catalogue.py "
                f"or correct the spelling."
            )

        missing_groups = {
            group
            for selection in self.datasets.values()
            for group in selection.groups
            if group not in self.groups
        }
        if missing_groups:
            raise FeatureConfigError(
                f"datasets reference undefined groups: {sorted(missing_groups)}"
            )

        # The allow-list must match the catalogue exactly. A mismatch in either
        # direction is a problem: an unlisted forward-looking feature is
        # unreviewed leakage, and a listed one that no longer looks forward means
        # the allow-list has gone stale and stopped being a real control.
        declared = {spec.name for spec in forward_looking_features()}
        allowed = set(self.validation.allowed_forward_looking)
        if declared != allowed:
            raise FeatureConfigError(
                f"forward-looking allow-list is out of step with the catalogue.\n"
                f"  in catalogue but not allowed: {sorted(declared - allowed)}\n"
                f"  allowed but not forward-looking: {sorted(allowed - declared)}"
            )

        return self

    def features_for(self, dataset: str) -> list[str]:
        """Flat, de-duplicated feature list for a dataset builder."""
        selection = self.datasets.get(dataset)
        if selection is None:
            raise FeatureConfigError(
                f"unknown dataset {dataset!r}; available: {sorted(self.datasets)}"
            )
        names: list[str] = []
        for group in selection.groups:
            names.extend(self.groups[group])
        return list(dict.fromkeys(names))

    def selection_for(self, dataset: str) -> DatasetSelection:
        selection = self.datasets.get(dataset)
        if selection is None:
            raise FeatureConfigError(
                f"unknown dataset {dataset!r}; available: {sorted(self.datasets)}"
            )
        return selection


@lru_cache(maxsize=1)
def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """Load and validate the feature configuration."""
    target = path or CONFIG_PATH
    if not target.is_file():
        raise FileNotFoundError(f"feature configuration not found at {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return FeatureConfig.model_validate(raw)


def reset_feature_config_cache() -> None:
    """Clear the cache. Intended for tests."""
    load_feature_config.cache_clear()
