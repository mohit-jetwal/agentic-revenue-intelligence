"""Local feature repository, with optional materialisation.

Computes features through :class:`~features.engineering.engineer.FeatureEngineer`
and can cache the result to Parquet under ``data/local/features/``.

**Materialisation is off by default**, which is the important decision here.
Turning it on is the local analogue of a Databricks Feature Table and makes the
Stage 2 story concrete - but while feature definitions are still moving through
Steps 4-11, a stale cache would silently feed old features to a new model, and
the resulting metrics would be wrong in a way nothing would flag. Off by default
means the failure mode is "a few seconds slower", not "quietly incorrect".

The cache key includes the feature version, so bumping ``FEATURE_VERSION``
invalidates everything - which is the behaviour you want when the definitions
have changed underneath.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.point_in_time import PointInTimeView
from features.contracts.catalogue import FEATURE_GROUPS, FeatureGroupName
from features.contracts.config import FeatureConfig, load_feature_config
from features.contracts.specs import (
    FEATURE_VERSION,
    FeatureSetMetadata,
    current_code_version,
    hash_request,
)
from features.engineering.engineer import FeatureEngineer, FeatureRequest
from features.repositories.base import (
    FeatureNotFoundError,
    FeatureRepository,
    FeatureSet,
)

logger = get_logger(__name__)


class LocalFeatureRepository(FeatureRepository):
    """Builds feature sets locally, optionally caching them to Parquet."""

    def __init__(
        self,
        view: PointInTimeView,
        *,
        config: FeatureConfig | None = None,
        materialise: bool = False,
        cache_root: Path | None = None,
    ) -> None:
        self.view = view
        self.engineer = FeatureEngineer(view)
        self.config = config or load_feature_config()
        self.materialise = materialise
        self.cache_root = cache_root
        if self.materialise and self.cache_root is None:
            raise ValueError("materialise=True requires a cache_root")

    # -- cache --------------------------------------------------------------

    def _cache_path(self, metadata: FeatureSetMetadata) -> Path:
        # Explicit rather than `assert`: an assert is stripped under `python -O`,
        # so the guard would silently disappear in exactly the environment where
        # a wrong path matters most.
        if self.cache_root is None:
            raise ValueError("cache_root is not configured; materialisation is disabled")
        return Path(self.cache_root) / f"{metadata.cache_key()}.parquet"

    def _read_cache(self, metadata: FeatureSetMetadata) -> pd.DataFrame | None:
        if not self.materialise:
            return None
        path = self._cache_path(metadata)
        if not path.is_file():
            return None
        logger.info("features.cache_hit", key=metadata.cache_key())
        return pd.read_parquet(path)

    def _write_cache(self, metadata: FeatureSetMetadata, frame: pd.DataFrame) -> None:
        if not self.materialise:
            return
        path = self._cache_path(metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        # Metadata beside the data, so a cached artifact is self-describing -
        # a bare Parquet file with no provenance is exactly what this layer
        # exists to avoid producing.
        path.with_suffix(".metadata.json").write_text(metadata.to_json(), encoding="utf-8")
        logger.info("features.cached", key=metadata.cache_key(), rows=len(frame))

    # -- building -----------------------------------------------------------

    def _metadata(
        self,
        name: str,
        request: FeatureRequest,
        *,
        feature_names: tuple[str, ...] = (),
        target: str | None = None,
        rows: int = 0,
    ) -> FeatureSetMetadata:
        return FeatureSetMetadata(
            feature_set_name=name,
            feature_version=FEATURE_VERSION,
            dataset_version=self.view.dataset_version(),
            as_of_date=self.view.as_of_date,
            start_date=request.start_date,
            end_date=request.end_date,
            source_tables=tuple(request.source_tables()),
            feature_names=feature_names,
            target_name=target,
            row_count=rows,
            code_version=current_code_version(),
            request_hash=hash_request(
                {
                    "products": sorted(request.product_ids) if request.product_ids else None,
                    "stores": sorted(request.store_ids) if request.store_ids else None,
                    "region": request.region,
                    "start": str(request.start_date),
                    "end": str(request.end_date),
                    "spend": request.include_promotion_spend,
                }
            ),
        )

    def _build_group(
        self,
        name: str,
        group: FeatureGroupName,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None,
        store_ids: list[str] | None,
    ) -> FeatureSet:
        """Build one logical feature group, keeping the panel keys alongside."""
        started = time.perf_counter()

        request = FeatureRequest(
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
            # Groups are not independent: demand lags need sales, the price
            # index needs products, competitor features need own price. Building
            # everything and then projecting is both simpler and correct, at the
            # cost of some wasted computation on a single-group request.
        )
        metadata = self._metadata(name, request)

        cached = self._read_cache(metadata)
        panel = cached if cached is not None else self.engineer.build(request)

        if panel.empty:
            # An empty panel is a filter problem, not a drift problem. Saying so
            # matters: the two have completely different fixes, and blaming the
            # catalogue sends someone hunting through feature definitions when
            # the real cause is a product and a store that were never co-listed.
            logger.warning(
                "features.empty_group",
                group=group.value,
                products=len(product_ids) if product_ids else None,
                stores=len(store_ids) if store_ids else None,
                start_date=str(start_date),
                end_date=str(end_date),
            )
            return FeatureSet(
                features=panel,
                metadata=metadata.model_copy(update={"feature_names": (), "row_count": 0}),
            )

        wanted = [c for c in FEATURE_GROUPS[group].names() if c in panel.columns]
        if not wanted:
            raise FeatureNotFoundError(
                f"the panel has {len(panel):,} rows but produced no features from "
                f"group {group.value!r}. The catalogue and the engineering "
                f"primitives have drifted apart - expected any of "
                f"{FEATURE_GROUPS[group].names()[:5]}..."
            )

        keys = [c for c in ("date", "product_id", "store_id") if c in panel.columns]
        frame = panel[[*keys, *wanted]].copy()

        if cached is None:
            self._write_cache(metadata, panel)

        logger.debug(
            "features.group_built",
            group=group.value,
            rows=len(frame),
            features=len(wanted),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

        return FeatureSet(
            features=frame,
            metadata=metadata.model_copy(
                update={"feature_names": tuple(wanted), "row_count": len(frame)}
            ),
        )

    # -- section 23 interface ------------------------------------------------

    def get_demand_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        return self._build_group(
            "demand_features",
            FeatureGroupName.DEMAND,
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )

    def get_pricing_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        return self._build_group(
            "pricing_features",
            FeatureGroupName.PRICE,
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )

    def get_promotion_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        return self._build_group(
            "promotion_features",
            FeatureGroupName.PROMOTION,
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )

    def get_inventory_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        return self._build_group(
            "inventory_features",
            FeatureGroupName.INVENTORY,
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )

    def get_competitor_features(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        return self._build_group(
            "competitor_features",
            FeatureGroupName.COMPETITOR,
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )

    def get_training_features(
        self,
        *,
        dataset: str,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
    ) -> FeatureSet:
        """Complete feature set for a configured dataset, X and y separated."""
        selection = self.config.selection_for(dataset)
        wanted = self.config.features_for(dataset)

        request = FeatureRequest(
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
            include_promotion_spend=selection.include_promotion_spend,
        )
        metadata = self._metadata(dataset, request, target=selection.target)

        cached = self._read_cache(metadata)
        panel = cached if cached is not None else self.engineer.build(request)
        if cached is None:
            self._write_cache(metadata, panel)

        if panel.empty:
            return FeatureSet(features=panel, metadata=metadata)

        if selection.exclude_promotional_rows and "promotion_flag" in panel.columns:
            panel = panel[~panel["promotion_flag"].astype(bool)]
        if selection.exclude_stockout_rows and "stockout_flag" in panel.columns:
            panel = panel[~panel["stockout_flag"].astype(bool)]

        available = [c for c in wanted if c in panel.columns]
        missing = sorted(set(wanted) - set(available))
        if missing:
            # Warn rather than fail: `lag_364_units` is legitimately absent on a
            # short window, and refusing outright would make short-window
            # experimentation impossible. The names are logged so a genuine
            # drift between config and code is still visible.
            logger.warning("features.missing_from_panel", dataset=dataset, missing=missing[:20])

        keys = [c for c in ("date", "product_id", "store_id") if c in panel.columns]
        target_series: pd.Series | None = None
        if selection.target and selection.target in panel.columns:
            target_series = panel[selection.target].reset_index(drop=True)

        features = panel[[*keys, *available]].reset_index(drop=True)

        return FeatureSet(
            features=features,
            target=target_series,
            metadata=metadata.model_copy(
                update={"feature_names": tuple(available), "row_count": len(features)}
            ),
        )

    def health_check(self) -> tuple[bool, str]:
        state = "materialising" if self.materialise else "compute-on-read"
        return True, f"LocalFeatureRepository ({state}, as-of {self.view.as_of_date})"
