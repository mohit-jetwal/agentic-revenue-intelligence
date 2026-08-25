"""Feature repositories - the seam between features and models.

A model asks for a named feature set and receives a :class:`FeatureSet` carrying
the frame, the separated target, and the lineage metadata describing which
dataset version, feature version and as-of date produced it. It never learns
whether that came from local computation, a Parquet cache, or a Databricks
Feature Table.

* ``LocalFeatureRepository``      - Stage 1; computes on read, optional caching
* ``DatabricksFeatureRepository`` - Stage 2; declared, not implemented
"""

from features.repositories.base import (
    FeatureNotFoundError,
    FeatureRepository,
    FeatureRepositoryError,
    FeatureSet,
)
from features.repositories.databricks import DatabricksFeatureRepository
from features.repositories.local import LocalFeatureRepository

__all__ = [
    "DatabricksFeatureRepository",
    "FeatureNotFoundError",
    "FeatureRepository",
    "FeatureRepositoryError",
    "FeatureSet",
    "LocalFeatureRepository",
]
