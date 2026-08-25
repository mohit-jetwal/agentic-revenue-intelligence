"""Databricks feature repository (Stage 2).

Declared, not implemented - same reasoning as
:class:`~data.repositories.databricks.DatabricksDataRepository`. Writing the
signatures now proves the interface is satisfiable by a Feature Store rather
than only by local computation, and makes the migration reviewable instead of
aspirational.

Notes for Stage 2:

* Use ``databricks-feature-engineering`` (``FeatureEngineeringClient``) rather
  than reading feature tables as plain Delta. The client's ``create_training_set``
  performs the **point-in-time join** natively, which is the same guarantee
  :class:`~data.repositories.point_in_time.PointInTimeView` provides locally -
  and it does it in the engine rather than in pandas.
* Feature tables are keyed on ``(product_id, store_id, date)`` with a timestamp
  key, so as-of joins are a platform feature rather than application code. The
  local implementation's warm-up window and manual shifting collapse into
  ``FeatureLookup(timestamp_lookup_key="date")``.
* Register feature tables under ``<catalog>.features.<name>`` so lineage and
  access control come from Unity Catalog. A model logged with
  ``FeatureEngineeringClient.log_model`` records its feature lookups, which is
  what makes the model-to-feature-version link automatic rather than a
  convention someone has to maintain.
* The local ``FEATURE_VERSION`` maps onto the feature table's Delta version. Both
  answer "which definitions produced these numbers", so the lineage recorded in
  :class:`~features.contracts.specs.FeatureSetMetadata` carries across unchanged.

The definitions themselves - ``features/contracts/catalogue.py`` - do **not**
move. Only the computation and storage do. That is the point of the split.
"""

from __future__ import annotations

from datetime import date

from features.repositories.base import FeatureRepository, FeatureSet

_STAGE = (
    "Stage 2 (Databricks Feature Engineering). The local implementation in "
    "features/repositories/local.py is the reference for behaviour."
)


class DatabricksFeatureRepository(FeatureRepository):
    """Reads feature tables via Databricks Feature Engineering."""

    def __init__(self, *, catalog: str, schema: str = "features") -> None:
        self.catalog = catalog
        self.schema = schema

    def _table(self, name: str) -> str:
        """Fully-qualified Unity Catalog feature table name."""
        return f"{self.catalog}.{self.schema}.{name}"

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"DatabricksFeatureRepository.{method}() belongs to {_STAGE}")

    def get_demand_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_demand_features")

    def get_pricing_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_pricing_features")

    def get_promotion_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_promotion_features")

    def get_inventory_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_inventory_features")

    def get_competitor_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_competitor_features")

    def get_training_features(
        self,
        *,
        dataset: str,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        raise self._not_yet("get_training_features")

    def health_check(self) -> tuple[bool, str]:
        return False, f"Databricks feature repository not implemented. {_STAGE}"
