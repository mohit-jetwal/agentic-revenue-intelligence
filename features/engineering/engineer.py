"""FeatureEngineer - composes the primitives into feature groups.

Brief section 12 asks for reusable primitives rather than model-specific
pipelines, and that is what ``features/engineering`` provides. This class is the
thin layer that assembles them, so a caller says "give me demand and price
features" instead of remembering which six functions to call in which order.

Ordering is not arbitrary and is the reason this class exists rather than a
loose set of calls:

1. **Prepare the panel.** Sorting must happen before anything shifts, or lags
   point at whatever row happened to precede.
2. **Join breadth-adding data** (promotions, inventory, competitor) *before*
   computing history over it, so rolling counts see complete rows.
3. **Compute temporal features last**, once every source column is present.

Get that order wrong and the failure is silent - features are produced, they are
simply wrong.

Takes a :class:`~data.repositories.point_in_time.PointInTimeView`, never a bare
repository. That is the structural half of leakage prevention: the engineer has
no method that returns future observed data because the object it holds has
none.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.point_in_time import PointInTimeView
from features.engineering import demand as demand_features
from features.engineering import entity as entity_features
from features.engineering import inventory as inventory_features
from features.engineering import pricing as pricing_features
from features.engineering import promotion as promotion_features
from features.engineering import temporal as temporal_features
from features.engineering.panel import PANEL_KEYS, drop_target_derived, prepare_panel

logger = get_logger(__name__)


@dataclass
class FeatureRequest:
    """What to build, for whom, over what.

    A request object rather than a dozen keyword arguments: the five dataset
    builders each need a different subset, and a struct keeps their call sites
    readable and their intent inspectable.
    """

    start_date: date
    end_date: date
    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    region: str | None = None

    #: Which groups to include.
    demand: bool = True
    temporal: bool = True
    price: bool = True
    promotion: bool = True
    inventory: bool = True
    competitor: bool = True
    product: bool = True
    store: bool = True

    #: Realised promotional spend. Off for forward-looking feature sets, where
    #: spend has not been booked yet.
    include_promotion_spend: bool = True
    #: Drop columns that are arithmetic functions of the target on the same row
    #: (``revenue``, ``cost``, ``gross_profit``). Default on: leaving them means
    #: every future model has to remember that revenue is `units` in disguise,
    #: and one of them eventually will not. Turn off only for reporting.
    drop_target_derived: bool = True
    #: Extra history loaded before ``start_date`` so lags at the window's start
    #: are populated rather than null. Defaults to a year, covering the 364-day
    #: lag.
    warmup_days: int = 400
    #: Trim the warm-up rows before returning, so the caller gets exactly the
    #: window asked for with fully-formed features.
    trim_warmup: bool = True

    lags: Sequence[int] = field(default_factory=lambda: demand_features.DEFAULT_LAGS)
    windows: Sequence[int] = field(default_factory=lambda: demand_features.DEFAULT_WINDOWS)

    def source_tables(self) -> list[str]:
        """Tables this request reads, for lineage metadata."""
        tables = ["sales_daily"]
        if self.temporal:
            tables.append("calendar")
        if self.promotion:
            tables.append("promotions")
        if self.inventory:
            tables.append("inventory")
        if self.competitor:
            tables.append("competitor_pricing")
        if self.product:
            tables.append("products")
        if self.store:
            tables.append("stores")
        return tables


class FeatureEngineer:
    """Builds ML-ready feature frames from a point-in-time view."""

    def __init__(self, view: PointInTimeView, *, keys: Sequence[str] = PANEL_KEYS) -> None:
        if not isinstance(view, PointInTimeView):
            # A bare repository would silently disable the as-of cut, and every
            # feature built afterwards would be quietly contaminated. Refusing
            # here converts a subtle correctness bug into an obvious TypeError.
            raise TypeError(
                "FeatureEngineer requires a PointInTimeView so the as-of cut cannot "
                "be bypassed. Use repository.as_of(<date>) to obtain one."
            )
        self.view = view
        self.keys = tuple(keys)

    @property
    def as_of_date(self) -> date:
        return self.view.as_of_date

    def build(self, request: FeatureRequest) -> pd.DataFrame:
        """Assemble the requested feature groups into one panel."""
        started = time.perf_counter()
        load_start = request.start_date - pd.Timedelta(days=request.warmup_days).to_pytimedelta()

        panel = self._load_sales(request, load_start)
        if panel.empty:
            logger.warning("features.empty_panel", start_date=str(request.start_date))
            return panel

        panel = prepare_panel(panel, keys=self.keys)

        # Breadth first: every source column present before history is computed.
        if request.promotion:
            panel = self._add_promotions(panel, request, load_start)
        if request.inventory:
            panel = self._add_inventory(panel, request, load_start)
        if request.competitor:
            panel = self._add_competitor(panel, request, load_start)
        if request.product or request.price:
            panel = self._add_products(panel, request)
        if request.store:
            panel = self._add_stores(panel, request)

        # Depth second: temporal features over the now-complete rows.
        if request.demand:
            panel = demand_features.build_demand_features(
                panel, lags=request.lags, windows=request.windows, keys=self.keys
            )
        if request.price:
            panel = pricing_features.add_price_features(panel, keys=self.keys)
        if request.temporal:
            panel = self._add_temporal(panel, request, load_start)

        if request.trim_warmup:
            panel = panel[pd.to_datetime(panel["date"]).dt.date >= request.start_date].reset_index(
                drop=True
            )

        if request.drop_target_derived:
            # Last, so the columns are available to anything that legitimately
            # needs them during construction (margin, price recomputation) but
            # never survive into what a model receives.
            panel = drop_target_derived(panel)

        logger.info(
            "features.built",
            rows=len(panel),
            columns=len(panel.columns),
            as_of_date=str(self.as_of_date),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return panel

    # -- loaders ------------------------------------------------------------

    def _load_sales(self, request: FeatureRequest, load_start: date) -> pd.DataFrame:
        return self.view.get_sales(
            product_ids=request.product_ids,
            store_ids=request.store_ids,
            region=request.region,
            start_date=load_start,
            end_date=request.end_date,
            # Feature building legitimately needs more than the default cap; the
            # explicit value also means truncation here is opted into rather
            # than stumbled upon.
            max_rows=20_000_000,
        )

    def _add_promotions(
        self, panel: pd.DataFrame, request: FeatureRequest, load_start: date
    ) -> pd.DataFrame:
        promotions = self.view.get_promotions(
            product_ids=request.product_ids,
            store_ids=request.store_ids,
            start_date=load_start,
            end_date=request.end_date,
            max_rows=5_000_000,
        )
        panel = promotion_features.add_promotion_features(
            panel,
            promotions,
            keys=self.keys,
            include_spend=request.include_promotion_spend,
        )
        return promotion_features.add_time_to_next_promotion(panel, promotions)

    def _add_inventory(
        self, panel: pd.DataFrame, request: FeatureRequest, load_start: date
    ) -> pd.DataFrame:
        inventory = self.view.get_inventory(
            product_ids=request.product_ids,
            store_ids=request.store_ids,
            start_date=load_start,
            end_date=request.end_date,
            max_rows=20_000_000,
        )
        return inventory_features.add_inventory_features(panel, inventory, keys=self.keys)

    def _add_competitor(
        self, panel: pd.DataFrame, request: FeatureRequest, load_start: date
    ) -> pd.DataFrame:
        competitor = self.view.get_competitor_prices(
            product_ids=request.product_ids,
            start_date=load_start,
            end_date=request.end_date,
            max_rows=5_000_000,
        )
        return pricing_features.add_competitor_features(panel, competitor, keys=self.keys)

    def _add_products(self, panel: pd.DataFrame, request: FeatureRequest) -> pd.DataFrame:
        products = self.view.get_products(product_ids=request.product_ids)
        panel = entity_features.add_product_features(panel, products)
        if request.price:
            panel = pricing_features.add_price_index(panel, products)
        return panel

    def _add_stores(self, panel: pd.DataFrame, request: FeatureRequest) -> pd.DataFrame:
        stores = self.view.get_stores(store_ids=request.store_ids, region=request.region)
        return entity_features.add_store_features(panel, stores)

    def _add_temporal(
        self, panel: pd.DataFrame, request: FeatureRequest, load_start: date
    ) -> pd.DataFrame:
        calendar = self.view.get_calendar(start_date=load_start, end_date=request.end_date)
        panel = temporal_features.add_time_features(panel)
        panel = temporal_features.join_calendar_features(panel, calendar)
        return temporal_features.add_festival_proximity(panel, calendar)
