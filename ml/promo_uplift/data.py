"""Building the causal panel from the repository (brief section 5).

One join, in one place. The CLI, the training script and any notebook all call
:func:`build_uplift_panel`, because the join decides which rows count as treated
- and two copies of that would drift apart, leaving two different definitions of
the same effect in the same codebase.

The panel is deliberately **wider than the analysis window**. A promotion on the
first day of the requested range still needs the eight weeks before it: every
covariate is a trailing statistic, and without that history the treated rows are
dropped for an incomplete adjustment set. Requesting exactly the analysis window
is the most common way to end up with an estimate computed on a fraction of the
promotions you asked about.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.forecasting.sampling import SeriesSample
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import UnknownPromotionError

logger = get_logger(__name__)

#: Extra history fetched before the analysis window, in days. Covers the longest
#: trailing covariate (56 days) with room for the pre-trend test on top.
HISTORY_PAD_DAYS = 120

#: Explicit row cap for the bulk reads. The repository defaults to a 100,000-row
#: guard and raises rather than silently truncating - the right behaviour for an
#: interactive query, and the wrong one for a deliberate panel build, which is
#: why the limit is opted out of here rather than raised globally. A 400-pair
#: sample over three years is roughly 440,000 rows before the semi-join.
MAX_PANEL_ROWS = 20_000_000


def build_uplift_panel(
    repository: DataRepository,
    sample: SeriesSample | pd.DataFrame | None = None,
    *,
    config: PromoUpliftConfig | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Sales, prices, inventory and the promotion calendar, at the causal grain.

    ``sample`` is an optional product-store selection - either a
    :class:`~ml.forecasting.sampling.SeriesSample` from
    :func:`ml.forecasting.sampling.sample_series`, or a bare frame with
    ``product_id`` and ``store_id``. Passing one keeps a development run
    interactive; omitting it reads the full panel.
    """
    settings = config or get_promo_uplift_config()

    products = list(product_ids) if product_ids else None
    stores = list(store_ids) if store_ids else None

    pairs = sample.pairs if isinstance(sample, SeriesSample) else sample
    if pairs is not None and not pairs.empty:
        # Flattened to independent filters rather than an exact pair list: the
        # repository filters on each axis, and an exact semi-join is applied
        # below. Step 6 measured that pushing pairs down as a product of two
        # filters loads ~7x the rows, so the semi-join happens here instead.
        products = sorted(pairs["product_id"].astype(str).unique())
        stores = sorted(pairs["store_id"].astype(str).unique())

    fetch_start = start_date - timedelta(days=HISTORY_PAD_DAYS) if start_date else None

    sales = repository.get_sales(
        product_ids=products,
        store_ids=stores,
        start_date=fetch_start,
        end_date=end_date,
        max_rows=MAX_PANEL_ROWS,
    )
    if sales.empty:
        raise UnknownPromotionError(
            "no sales rows for the requested products, stores and dates",
            product_ids=products,
            store_ids=stores,
        )

    sales["date"] = pd.to_datetime(sales["date"])

    if pairs is not None and not pairs.empty:
        index = pd.MultiIndex.from_frame(
            pairs[["product_id", "store_id"]].astype(str)
        )
        sales = sales[
            pd.MultiIndex.from_frame(
                sales[["product_id", "store_id"]].astype(str)
            ).isin(index)
        ]

    panel = _attach_attributes(repository, sales)
    panel = _attach_event_spend(repository, panel, products, stores, fetch_start, end_date)

    logger.info(
        "promo_uplift.panel_built",
        rows=len(panel),
        products=int(panel["product_id"].nunique()),
        stores=int(panel["store_id"].nunique()),
        promoted_rows=int(panel["promotion_id"].notna().sum()),
    )
    _ = settings
    return panel.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)


def _attach_attributes(repository: DataRepository, sales: pd.DataFrame) -> pd.DataFrame:
    """Join the product and store attributes the adjustment set needs.

    Category and region are not decoration: they stratify the cross-sectional
    control pool and they carry the seasonal confounder, since promotion timing
    is targeted at each category's own peak.
    """
    panel = sales

    products = repository.get_products()
    if not products.empty:
        columns = [
            c for c in ("product_id", "category", "brand", "unit_cost") if c in products.columns
        ]
        panel = panel.merge(products[columns], on="product_id", how="left")

    stores = repository.get_stores()
    if not stores.empty:
        columns = [
            c
            for c in ("store_id", "region", "store_tier", "city")
            if c in stores.columns
        ]
        # `channel` already arrives on the sales fact; taking it from the store
        # table too would produce channel_x/channel_y and silently break every
        # downstream reference to it.
        columns = [c for c in columns if c not in panel.columns or c == "store_id"]
        panel = panel.merge(stores[columns], on="store_id", how="left")

    return panel


def _attach_event_spend(
    repository: DataRepository,
    panel: pd.DataFrame,
    products: list[str] | None,
    stores: list[str] | None,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Bring promotion type and realised spend onto the panel.

    Spend is joined at **event** level and placed on the event's first day only.
    Broadcasting a per-event total across every day of a twenty-day window would
    multiply the spend by twenty as soon as the event table is summed - and ROI
    would come back at a twentieth of its true value, which looks plausible
    enough to survive review.
    """
    promotions = repository.get_promotions(
        product_ids=products,
        store_ids=stores,
        start_date=start_date,
        end_date=end_date,
        max_rows=MAX_PANEL_ROWS,
    )
    if promotions.empty:
        panel["promotion_type"] = None
        panel["promotion_spend"] = 0.0
        return panel

    events = promotions.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])

    columns = ["promotion_id", "promotion_type", "start_date"]
    for optional in ("promotion_spend", "display_flag", "bundle_flag"):
        if optional in events.columns:
            columns.append(optional)

    merged = panel.merge(
        events[columns].rename(columns={"start_date": "_event_start"}),
        on="promotion_id",
        how="left",
    )

    if "promotion_spend" in merged.columns:
        first_day = merged["date"] == merged["_event_start"]
        merged["promotion_spend"] = merged["promotion_spend"].where(first_day, 0.0).fillna(0.0)
    else:
        merged["promotion_spend"] = 0.0

    return merged.drop(columns=["_event_start"])


def baseline_predictions(
    panel: pd.DataFrame,
    repository: DataRepository,
    model_dir: object,
) -> pd.Series | None:
    """Per-row baseline units from the Step 5 model, or ``None`` if untrained.

    Returns ``None`` rather than raising. The baseline counterfactual is one
    estimator among six, and an untrained baseline should cost that one row of
    the comparison table - not the whole analysis.
    """
    from pathlib import Path

    try:
        from features.datasets.builders import FeatureRequest
        from features.engineering.engineer import FeatureEngineer
        from ml.baseline.model import FittedBaselineModel
    except ImportError:
        return None

    directory = Path(str(model_dir))
    if not (directory / "model.joblib").is_file():
        return None

    try:
        model = FittedBaselineModel.load_from(directory, repository)
        as_of = pd.to_datetime(panel["date"]).max().date()
        engineer = FeatureEngineer(repository.as_of(as_of))
        features = engineer.build(
            FeatureRequest(
                start_date=pd.to_datetime(panel["date"]).min().date(),
                end_date=as_of,
                product_ids=sorted(panel["product_id"].astype(str).unique()),
                store_ids=sorted(panel["store_id"].astype(str).unique()),
                promotion=False,
                include_promotion_spend=False,
            )
        )
        prediction = model.predict_panel(features)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.info("promo_uplift.baseline_unavailable", error=str(exc))
        return None

    frame = prediction.frame
    keyed = frame.set_index(
        [
            pd.to_datetime(frame["date"]),
            frame["product_id"].astype(str),
            frame["store_id"].astype(str),
        ]
    )["baseline_units"]

    lookup = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(panel["date"]),
            panel["product_id"].astype(str),
            panel["store_id"].astype(str),
        ]
    )
    return pd.Series(keyed.reindex(lookup).to_numpy(), index=panel.index, name="baseline_units")


__all__ = ["HISTORY_PAD_DAYS", "baseline_predictions", "build_uplift_panel"]
