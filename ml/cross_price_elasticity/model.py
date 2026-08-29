"""The fitted cross-price elasticity model."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.base import InsufficientDataError, ModelMetadata
from ml.cross_price_elasticity.estimator import (
    PairEstimate,
    candidate_pairs,
    estimate_cross_elasticities,
)
from ml.cross_price_elasticity.interface import (
    CrossElasticityPair,
    CrossElasticityResult,
    CrossPriceElasticityModel,
)
from ml.price_elasticity.data import build_elasticity_panel
from ml.price_elasticity.estimator import estimation_window, prepare_panel

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"


class FittedCrossElasticityModel(CrossPriceElasticityModel):
    """Cross-price relationships, estimated on request."""

    name = "cross_price_elasticity"

    def __init__(
        self,
        repository: DataRepository,
        *,
        model_version: str = MODEL_VERSION,
    ) -> None:
        super().__init__(repository)
        self.version = model_version
        self._metadata = ModelMetadata(
            name=self.name,
            version=model_version,
            trained_at=datetime.now(UTC),
            approved=False,
        )

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """No-op: estimated on demand, nothing to persist."""
        return self.metadata

    def predict(  # type: ignore[override]
        self,
        *,
        product_id: str,
        candidate_product_ids: list[str] | None = None,
        region: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> CrossElasticityResult:
        """Estimate cross-price relationships for one focal product."""
        products = self._repository.get_products()
        relationships = self._safe_relationships()

        candidates = candidate_product_ids or candidate_pairs(
            product_id, products, relationships
        )
        if not candidates:
            raise InsufficientDataError(
                f"no candidate products for {product_id}. Cross-price effects are "
                f"only tested within a category or against a declared "
                f"relationship - a pair chosen at random is a coincidence that "
                f"survived a t-test, not a finding"
            )

        panel = build_elasticity_panel(
            self._repository,
            product_ids=[product_id, *candidates],
            region=region,
            start_date=start_date,
            end_date=end_date,
        )
        if panel.empty:
            raise InsufficientDataError(f"no sales rows for {product_id} and its candidates")

        # Two preparations, deliberately. The focal panel drops promoted rows -
        # a promotion on the focal product moves its demand through a mechanic
        # unrelated to any candidate's price. The source panels KEEP them,
        # because a candidate's promotional price cut is the largest single
        # source of the price variation that identifies the cross effect.
        # Measured: dropping them cut the identifying variation by 29% and lost
        # the true substitute entirely.
        focal_prepared = prepare_panel(panel, drop_promotions=True)
        source_prepared = prepare_panel(panel, drop_promotions=False)

        focal_panels = {
            str(pid): group
            for pid, group in focal_prepared.groupby("product_id", observed=True)
        }
        panels = {
            str(pid): group
            for pid, group in source_prepared.groupby("product_id", observed=True)
        }
        if product_id not in focal_panels:
            raise InsufficientDataError(
                f"no usable rows for {product_id} after dropping promoted, "
                f"censored and zero-unit days"
            )
        panels[product_id] = focal_panels[product_id]

        pairs, tested = estimate_cross_elasticities(panels, product_id, candidates)
        return self._to_result(
            pairs,
            product_id=product_id,
            region=region,
            tested=tested,
            frame=panels[product_id],
        )

    def _safe_relationships(self) -> pd.DataFrame:
        try:
            return self._repository.get_product_relationships()
        except (NotImplementedError, AttributeError, ValueError):
            return pd.DataFrame()

    def _to_result(
        self,
        pairs: list[PairEstimate],
        *,
        product_id: str,
        region: str | None,
        tested: int,
        frame: pd.DataFrame,
    ) -> CrossElasticityResult:
        records = [
            CrossElasticityPair(
                source_product_id=p.source_product_id,
                target_product_id=p.target_product_id,
                cross_elasticity=p.cross_elasticity,
                relationship_type=p.relationship,
                strength=p.strength,
                confidence_interval=p.confidence_interval(),
                p_value=p.adjusted_p_value if p.adjusted_p_value is not None else p.p_value,
                sample_size=p.n_obs,
                is_significant=p.is_significant,
            )
            for p in sorted(pairs, key=lambda p: abs(p.cross_elasticity), reverse=True)
        ]

        return CrossElasticityResult(
            product_id=product_id,
            region=region,
            pairs=records,
            substitutes=[
                r.source_product_id
                for r in records
                if r.relationship_type == "substitute" and r.is_significant
            ],
            complements=[
                r.source_product_id
                for r in records
                if r.relationship_type == "complement" and r.is_significant
            ],
            method="panel_fe_pairwise",
            estimation_window=estimation_window(frame),
            pairs_tested=tested,
            multiple_testing_correction="benjamini_hochberg",
            assumptions=[
                "Candidates are restricted to within-category products and "
                "declared relationships, chosen before looking at any outcome.",
                "The focal product's own log price is included, so shared price "
                "movement from category cost shocks does not load onto the cross "
                "coefficient.",
                "Prices are matched within store and day: substitution happens "
                "on a shelf, not against a national average.",
                "p-values are Benjamini-Hochberg adjusted across the candidates "
                f"tested ({tested}). Significance quoted against raw p-values "
                "would be meaningless at this many comparisons.",
            ],
            warnings=(
                [
                    f"only {len(records)} of {tested} candidate pairs had enough "
                    f"overlapping store-days to estimate"
                ]
                if len(records) < tested
                else []
            ),
        )


__all__ = ["MODEL_VERSION", "FittedCrossElasticityModel"]
