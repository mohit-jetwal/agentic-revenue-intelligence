"""Feature repository interface (brief section 23).

The seam between feature engineering and models. A model asks for a named
feature set; it does not know whether that set was computed on the fly, read
from a local Parquet cache, or fetched from a Databricks Feature Table.

That indirection is what makes brief section 47 achievable - ``model.fit(X, y)``
should not care where ``X`` came from. It also gives Step 12 somewhere to record
lineage: every returned feature set carries a
:class:`~features.contracts.specs.FeatureSetMetadata` naming the dataset version,
feature version and as-of date it was built from.

The method names follow section 23 - ``get_demand_features`` and friends - but
they all return a :class:`FeatureSet`, so a caller gets the frame and its
provenance together rather than a bare DataFrame it then has to describe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from features.contracts.specs import FeatureSetMetadata


class FeatureRepositoryError(RuntimeError):
    """Raised when a feature set cannot be produced."""


class FeatureNotFoundError(FeatureRepositoryError):
    """Raised when a requested feature is absent from the built frame.

    Distinct from a generic error because it usually means the feature
    configuration and the catalogue have drifted apart, which has a specific fix.
    """


@dataclass(frozen=True)
class FeatureSet:
    """A feature frame together with its provenance.

    ``X`` and ``y`` are separated deliberately (brief section 36). Returning one
    combined frame and trusting the caller to drop the target is how the target
    ends up in the feature matrix, and that mistake produces a model with
    suspiciously perfect metrics.
    """

    features: pd.DataFrame
    metadata: FeatureSetMetadata
    target: pd.Series | None = None

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional in ML code
        return self.features

    @property
    def y(self) -> pd.Series | None:
        return self.target

    def __len__(self) -> int:
        return len(self.features)

    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def describe(self) -> str:
        lines = [
            f"{self.metadata.feature_set_name}: {len(self.features):,} rows "
            f"x {len(self.features.columns)} features",
            f"  feature_version : {self.metadata.feature_version}",
            f"  dataset_version : {self.metadata.dataset_version}",
            f"  as_of_date      : {self.metadata.as_of_date}",
            f"  window          : {self.metadata.start_date} -> {self.metadata.end_date}",
            f"  source_tables   : {', '.join(self.metadata.source_tables)}",
        ]
        if self.target is not None:
            lines.append(f"  target          : {self.metadata.target_name}")
        return "\n".join(lines)


class FeatureRepository(ABC):
    """Access to engineered feature sets.

    Implementations differ in *where* features come from, never in what they
    mean - the definitions live in ``features/contracts/catalogue.py`` and are
    shared.
    """

    @abstractmethod
    def get_demand_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Lags, rolling statistics and demand dynamics."""

    @abstractmethod
    def get_pricing_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Own price, discount depth, price index and movement."""

    @abstractmethod
    def get_promotion_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Promotion schedule and history."""

    @abstractmethod
    def get_inventory_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Availability and stockout history."""

    @abstractmethod
    def get_competitor_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Competitor price position."""

    @abstractmethod
    def get_training_features(
        self,
        *,
        dataset: str,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """A complete, model-ready feature set with its target separated.

        ``dataset`` names an entry in ``configs/features/features.yaml``, so
        which groups a model receives is configuration rather than a code
        change.
        """

    def health_check(self) -> tuple[bool, str]:
        """Cheap probe for diagnostics."""
        return True, type(self).__name__
