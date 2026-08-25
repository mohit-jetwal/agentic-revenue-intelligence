"""Model-ready dataset builders (brief sections 36-40).

Five builders, one per future model. Each takes a
:class:`~data.repositories.point_in_time.PointInTimeView`, returns a
:class:`~features.repositories.base.FeatureSet` with ``X`` and ``y`` separated,
and carries lineage metadata.

These are **not** models. They assemble the inputs a model will need and stop
there - fitting begins in Step 4. What they do encode is each model's *framing*:
which rows are eligible, what the target is, what has to be excluded to keep the
estimate honest. That framing is genuinely part of the data layer, because
getting it wrong produces a dataset on which no model can be correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.point_in_time import PointInTimeView
from features.contracts.config import load_feature_config
from features.contracts.specs import (
    FEATURE_VERSION,
    FeatureSetMetadata,
    current_code_version,
    hash_request,
)
from features.engineering.engineer import FeatureEngineer, FeatureRequest
from features.repositories.base import FeatureSet
from features.repositories.local import LocalFeatureRepository

logger = get_logger(__name__)


def _metadata(
    name: str,
    view: PointInTimeView,
    *,
    start_date: date,
    end_date: date,
    source_tables: tuple[str, ...],
    feature_names: tuple[str, ...] = (),
    target: str | None = None,
    rows: int = 0,
    extra: dict[str, object] | None = None,
) -> FeatureSetMetadata:
    return FeatureSetMetadata(
        feature_set_name=name,
        feature_version=FEATURE_VERSION,
        dataset_version=view.dataset_version(),
        as_of_date=view.as_of_date,
        start_date=start_date,
        end_date=end_date,
        source_tables=source_tables,
        feature_names=feature_names,
        target_name=target,
        row_count=rows,
        code_version=current_code_version(),
        request_hash=hash_request(extra or {}),
    )


# ---------------------------------------------------------------------------
# 36. Forecasting
# ---------------------------------------------------------------------------


def create_forecasting_dataset(
    view: PointInTimeView,
    *,
    train_start: date,
    train_end: date,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
) -> FeatureSet:
    """Demand forecasting dataset (brief section 36).

    Target is ``units``. Features are the full complement: lags, rolling
    statistics, price, promotion schedule, inventory, competitor and calendar.

    Realised promotional spend is **excluded** - it does not exist over a
    forecast horizon, and training on a feature that will be absent at inference
    guarantees a train/serve mismatch. This is why the config carries
    ``include_promotion_spend: false`` for this dataset rather than it being a
    detail buried in the builder.
    """
    repository = LocalFeatureRepository(view)
    feature_set = repository.get_training_features(
        dataset="forecasting",
        start_date=train_start,
        end_date=train_end,
        product_ids=product_ids,
        store_ids=store_ids,
    )

    logger.info(
        "dataset.forecasting_built",
        rows=len(feature_set),
        features=len(feature_set.feature_names()),
        as_of_date=str(view.as_of_date),
    )
    return feature_set


# ---------------------------------------------------------------------------
# 37. Price elasticity
# ---------------------------------------------------------------------------


def create_price_elasticity_dataset(
    view: PointInTimeView,
    *,
    start_date: date,
    end_date: date,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
    exclude_promotional: bool = True,
    exclude_stockouts: bool = True,
) -> FeatureSet:
    """Own-price elasticity dataset (brief section 37).

    Two exclusions, both load-bearing rather than tidying:

    * **Promotional rows** carry a price cut *and* an additive uplift at the same
      time. Regressing across them conflates the shopper's price response with
      the promotion's display and mechanic effects, inflating the apparent
      elasticity.
    * **Stockout rows** report supply, not demand. Including them biases the
      estimate toward zero for a reason that has nothing to do with price.

    Both default to ``True`` and are exposed so Step 8 can demonstrate the bias
    by turning them off - which is a more convincing argument than asserting it.
    """
    config = load_feature_config()
    selection = config.selection_for("price_elasticity")

    # Built directly rather than through the feature repository: this dataset
    # needs the log-log transforms and the row exclusions applied together, and
    # the repository's generic path would return the frame before either.
    engineer = FeatureEngineer(view)
    panel = engineer.build(
        FeatureRequest(
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )
    )
    if panel.empty:
        return FeatureSet(
            features=panel,
            metadata=_metadata(
                "price_elasticity",
                view,
                start_date=start_date,
                end_date=end_date,
                source_tables=("sales_daily", "pricing", "competitor_pricing"),
            ),
        )

    excluded_promo = 0
    excluded_stock = 0
    if exclude_promotional and "promotion_flag" in panel.columns:
        mask = ~panel["promotion_flag"].astype(bool)
        excluded_promo = int((~mask).sum())
        panel = panel[mask]
    if exclude_stockouts and "stockout_flag" in panel.columns:
        mask = ~panel["stockout_flag"].astype(bool)
        excluded_stock = int((~mask).sum())
        panel = panel[mask]

    # Log-log terms, so the coefficient on log price *is* the elasticity. Doing
    # it here rather than in the model keeps the transformation with the data
    # contract, where its zero-handling can be stated once.
    panel = panel[(panel["units"] > 0) & (panel["selling_price"] > 0)].copy()
    panel["log_units"] = np.log(panel["units"])
    panel["log_price"] = np.log(panel["selling_price"])
    if "competitor_price" in panel.columns:
        panel["log_competitor_price"] = np.log(panel["competitor_price"].replace(0.0, np.nan))

    wanted = [c for c in config.features_for("price_elasticity") if c in panel.columns]
    derived = [c for c in ("log_price", "log_competitor_price") if c in panel.columns]
    keys = [c for c in ("date", "product_id", "store_id") if c in panel.columns]
    # Region is a natural fixed-effect grouping for elasticity, so it is kept
    # even though it is not in the configured feature list.
    context = [c for c in ("region", "category") if c in panel.columns]

    features = panel[[*keys, *context, *wanted, *derived]].reset_index(drop=True)
    target = panel["log_units"].reset_index(drop=True)

    logger.info(
        "dataset.elasticity_built",
        rows=len(features),
        excluded_promotional=excluded_promo,
        excluded_stockout=excluded_stock,
    )

    return FeatureSet(
        features=features,
        target=target,
        metadata=_metadata(
            "price_elasticity",
            view,
            start_date=start_date,
            end_date=end_date,
            source_tables=("sales_daily", "pricing", "competitor_pricing", "products", "stores"),
            feature_names=tuple([*wanted, *derived]),
            target="log_units",
            rows=len(features),
            extra={
                "exclude_promotional": exclude_promotional,
                "exclude_stockouts": exclude_stockouts,
                "grain": selection.grain,
            },
        ),
    )


# ---------------------------------------------------------------------------
# 38. Promotion uplift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpliftWindows:
    """Period definitions for uplift measurement (brief section 38).

    * **pre-period**  - the ``pre_days`` before a promotion starts. The baseline
      the promotion is measured against, and it must end strictly before the
      promotion begins.
    * **treatment**   - the promotion window itself.
    * **post-period** - the ``post_days`` after it ends. Included because pantry
      loading depresses demand afterwards; ignoring it is precisely what makes a
      naive during-vs-before comparison overstate incrementality.
    """

    pre_days: int = 28
    post_days: int = 14


def create_promo_uplift_dataset(
    view: PointInTimeView,
    *,
    start_date: date,
    end_date: date,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
    windows: UpliftWindows | None = None,
) -> FeatureSet:
    """Promotion uplift dataset (brief section 38).

    Labels each row with its period relative to the nearest promotion, and marks
    treatment against control.

    **Control definition.** A control row is a product-store-date with no active
    promotion, for a product that *is* promoted somewhere in the window. Using
    never-promoted products as controls would be worse: promotions are targeted
    at products with particular demand profiles, so a never-promoted product is
    systematically different, and the comparison would measure product
    selection rather than promotional effect.
    """
    windows = windows or UpliftWindows()
    engineer = FeatureEngineer(view)

    panel = engineer.build(
        FeatureRequest(
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
        )
    )
    if panel.empty or "promotion_flag" not in panel.columns:
        return FeatureSet(
            features=panel,
            metadata=_metadata(
                "promo_uplift",
                view,
                start_date=start_date,
                end_date=end_date,
                source_tables=("sales_daily", "promotions"),
            ),
        )

    panel = panel.copy()
    panel["treatment"] = panel["promotion_flag"].astype(bool)

    # Period label. `days_since_promotion` and `days_until_promotion_end` are
    # already computed by the promotion primitives, so this is classification
    # rather than recomputation.
    period = pd.Series("baseline", index=panel.index, dtype="object")
    period[panel["treatment"]] = "treatment"

    days_to_next = panel.get("days_to_next_promotion")
    if days_to_next is not None:
        in_pre = (~panel["treatment"]) & days_to_next.between(0, windows.pre_days)
        period[in_pre] = "pre"

    days_since = panel.get("days_since_promotion")
    if days_since is not None:
        in_post = (~panel["treatment"]) & days_since.between(0, windows.post_days)
        period[in_post] = "post"

    panel["period"] = period

    # Products promoted at some point in the window are the eligible universe;
    # their unpromoted days are the controls.
    promoted_products = set(panel.loc[panel["treatment"], "product_id"].unique())
    panel["eligible_for_uplift"] = panel["product_id"].isin(promoted_products)

    config = load_feature_config()
    wanted = [c for c in config.features_for("promo_uplift") if c in panel.columns]
    keys = [c for c in ("date", "product_id", "store_id") if c in panel.columns]
    labels = ["treatment", "period", "eligible_for_uplift"]
    context = [c for c in ("promotion_id", "promotion_type", "region") if c in panel.columns]

    features = panel[[*keys, *labels, *context, *wanted]].reset_index(drop=True)
    target = panel["units"].reset_index(drop=True)

    logger.info(
        "dataset.uplift_built",
        rows=len(features),
        treatment_rows=int(panel["treatment"].sum()),
        control_rows=int((~panel["treatment"] & panel["eligible_for_uplift"]).sum()),
    )

    return FeatureSet(
        features=features,
        target=target,
        metadata=_metadata(
            "promo_uplift",
            view,
            start_date=start_date,
            end_date=end_date,
            source_tables=("sales_daily", "promotions", "pricing", "products", "stores"),
            feature_names=tuple(wanted),
            target="units",
            rows=len(features),
            extra={"pre_days": windows.pre_days, "post_days": windows.post_days},
        ),
    )


# ---------------------------------------------------------------------------
# 39. Cross-price
# ---------------------------------------------------------------------------


def create_cross_price_dataset(
    view: PointInTimeView,
    *,
    start_date: date,
    end_date: date,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
    max_pairs: int = 200,
) -> FeatureSet:
    """Cross-price dataset (brief section 39).

    One row per ``(date, store, product_a, product_b)`` where ``product_a``'s
    demand is paired with ``product_b``'s price.

    **Pair selection uses ``product_relationships``**, which is the point of
    that table. With N products there are N(N-1) ordered pairs - 89,700 at dev
    scale - and testing them all guarantees spurious findings at any conventional
    significance level. Restricting to declared candidates keeps the multiple-
    comparison problem tractable.

    **Grain is store-level**, because substitution happens on a shelf. Aggregating
    to product-date first averages away the store-level price differences that
    identify the effect - a lesson from Step 2's validation, where the same
    aggregation dropped sign agreement from 8/8 to 5/8.
    """
    relationships = view.get_product_relationships(product_ids=product_ids)
    if relationships.empty:
        return FeatureSet(
            features=pd.DataFrame(),
            metadata=_metadata(
                "cross_price",
                view,
                start_date=start_date,
                end_date=end_date,
                source_tables=("product_relationships",),
            ),
        )

    candidates = relationships[relationships["relationship_type"] != "unrelated"].copy()
    candidates = candidates.reindex(
        candidates["cross_elasticity"].abs().sort_values(ascending=False).index
    ).head(max_pairs)

    involved = sorted(set(candidates["product_a"]) | set(candidates["product_b"]))
    engineer = FeatureEngineer(view)
    panel = engineer.build(
        FeatureRequest(
            start_date=start_date,
            end_date=end_date,
            product_ids=involved,
            store_ids=store_ids,
            demand=False,
            inventory=False,
        )
    )
    if panel.empty:
        return FeatureSet(
            features=panel,
            metadata=_metadata(
                "cross_price",
                view,
                start_date=start_date,
                end_date=end_date,
                source_tables=("sales_daily", "pricing", "product_relationships"),
            ),
        )

    keep = [
        c
        for c in (
            "date",
            "store_id",
            "product_id",
            "units",
            "selling_price",
            "promotion_flag",
            "competitor_price",
            "region",
            "category",
        )
        if c in panel.columns
    ]
    slim = panel[keep]

    target_side = slim.rename(
        columns={
            "product_id": "product_a",
            "units": "demand_a",
            "selling_price": "price_a",
            "promotion_flag": "promotion_a",
        }
    )
    source_side = slim[
        [
            c
            for c in ("date", "store_id", "product_id", "selling_price", "promotion_flag")
            if c in slim.columns
        ]
    ].rename(
        columns={
            "product_id": "product_b",
            "selling_price": "price_b",
            "promotion_flag": "promotion_b",
        }
    )

    pairs = candidates[["product_a", "product_b", "relationship_type", "cross_elasticity"]]
    # `cross_elasticity` is the *declared* strength from the relationship table,
    # not ground truth - it is a candidate label for scoping, and Step 9 must
    # estimate the value independently.
    pairs = pairs.rename(columns={"cross_elasticity": "declared_strength"})

    merged = target_side.merge(pairs, on="product_a", how="inner")
    merged = merged.merge(source_side, on=["date", "store_id", "product_b"], how="inner")
    merged = merged[merged["demand_a"] > 0]

    if not merged.empty:
        merged["log_demand_a"] = np.log(merged["demand_a"])
        merged["log_price_a"] = np.log(merged["price_a"].replace(0.0, np.nan))
        merged["log_price_b"] = np.log(merged["price_b"].replace(0.0, np.nan))

    target = merged["log_demand_a"].reset_index(drop=True) if not merged.empty else None
    features = merged.reset_index(drop=True)

    logger.info(
        "dataset.cross_price_built",
        rows=len(features),
        pairs=int(features[["product_a", "product_b"]].drop_duplicates().shape[0])
        if not features.empty
        else 0,
    )

    return FeatureSet(
        features=features,
        target=target,
        metadata=_metadata(
            "cross_price",
            view,
            start_date=start_date,
            end_date=end_date,
            source_tables=("sales_daily", "pricing", "product_relationships", "competitor_pricing"),
            feature_names=("log_price_b", "log_price_a", "promotion_a", "promotion_b"),
            target="log_demand_a",
            rows=len(features),
            extra={"max_pairs": max_pairs},
        ),
    )


# ---------------------------------------------------------------------------
# 40. Trade promotion optimisation
# ---------------------------------------------------------------------------


def create_promo_optimization_dataset(
    view: PointInTimeView,
    *,
    start_date: date,
    end_date: date,
    product_ids: list[str] | None = None,
    lookback_days: int = 365,
) -> FeatureSet:
    """Trade promotion optimisation dataset (brief section 40).

    One row per ``(product, region)`` decision cell, carrying the historical ROI
    and margin an allocator needs.

    Deliberately **descriptive**: ``forecast_sales`` and ``uplift`` are left as
    placeholder columns, because they are *outputs* of Steps 5 and 6. Filling
    them here with a naive estimate would look complete and quietly become the
    number the optimiser trusts. An explicit null is the honest interface - Step
    7 populates them from the real models.
    """
    history_start = start_date - timedelta(days=lookback_days)

    trade = view.get_trade_promotions(
        product_ids=product_ids, start_date=history_start, end_date=end_date
    )
    products = view.get_products(product_ids=product_ids)

    if trade.empty:
        return FeatureSet(
            features=pd.DataFrame(),
            metadata=_metadata(
                "promo_optimization",
                view,
                start_date=start_date,
                end_date=end_date,
                source_tables=("trade_promotions", "products"),
            ),
        )

    cell = (
        trade.groupby(["product_id", "region"], observed=True)
        .agg(
            historical_roi=("roi", "mean"),
            historical_roi_std=("roi", "std"),
            historical_uplift=("actual_uplift", "mean"),
            promotion_cost=("actual_spend", "mean"),
            total_historical_spend=("actual_spend", "sum"),
            events=("trade_promo_id", "count"),
        )
        .reset_index()
    )

    cell = cell.merge(
        products[["product_id", "unit_cost", "base_price", "category"]],
        on="product_id",
        how="left",
    )
    cell["margin"] = (cell["base_price"] - cell["unit_cost"]) / cell["base_price"].replace(
        0.0, np.nan
    )

    # Spend bounds derived from history: a cell that has never taken more than
    # X is unlikely to absorb 10X, and an optimiser without bounds will try.
    cell["minimum_spend"] = 0.0
    cell["maximum_spend"] = cell["promotion_cost"] * 3.0

    # Populated by Steps 5 and 6 - see the docstring.
    cell["forecast_sales"] = np.nan
    cell["baseline_sales"] = np.nan
    cell["uplift"] = np.nan

    logger.info("dataset.optimization_built", cells=len(cell))

    return FeatureSet(
        features=cell,
        metadata=_metadata(
            "promo_optimization",
            view,
            start_date=start_date,
            end_date=end_date,
            source_tables=("trade_promotions", "products"),
            feature_names=tuple(cell.columns),
            rows=len(cell),
            extra={"lookback_days": lookback_days},
        ),
    )
