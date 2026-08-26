"""Series sampling that actually samples what it says it does.

This module exists because of a measured defect in Step 4 that must not be
inherited.

``ml/baseline/pipeline.py:190-201`` samples N product-store pairs, then throws
the pairing away::

    pairs = sample_product_store_pairs(repository, n_pairs=sample_pairs, ...)
    product_ids = sorted(pairs["product_id"].unique().tolist())
    store_ids = sorted(pairs["store_id"].unique().tolist())

Passing those two lists as independent filters asks for the **cross product** -
the exact thing the comment three lines above it warns against. Measured on this
dataset: 300 products are each listed in roughly 20 of 200 stores, so 6,128 real
series exist out of a possible 60,000. Drawing 400 pairs touches ~221 distinct
products and ~173 distinct stores, and the resulting box contains
``221 x 173 x 0.102 ~= 3,900`` real series. ``--sample-pairs 400`` therefore
loads about **ten times** what it claims to.

Two things fix it:

1. **Cluster the sample by store.** Restricting to a handful of stores *first*
   keeps the filter box tight around the pairs actually wanted. Sampling pairs
   uniformly across all 200 stores guarantees a wide box no matter how few pairs
   are requested, because the store list saturates almost immediately.
2. **Semi-join afterwards.** The repository filter can only ever be a superset -
   it takes two lists, not a set of tuples - so the exact pair set is enforced on
   the built panel. ``n_series=800`` then means 800.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataAccessError, DataRepository

logger = get_logger(__name__)

#: How much larger the filter box is allowed to be than the requested sample.
#: Some slack is unavoidable - stores carry different assortments - but an order
#: of magnitude is the Step 4 defect, and this keeps it near 1.2x.
_BOX_TOLERANCE = 1.6


@dataclass(frozen=True)
class SeriesSample:
    """An exact set of product-store series, plus the filters that reach them."""

    #: The authoritative pair set. One row per series.
    pairs: pd.DataFrame
    #: Superset filters for the repository, which cannot express tuples.
    product_ids: list[str]
    store_ids: list[str]

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def box_size(self) -> int:
        """Series the repository filter will actually return, before the semi-join."""
        return len(self.product_ids) * len(self.store_ids)

    def describe(self) -> str:
        return (
            f"{len(self.pairs):,} series across {len(self.product_ids)} products "
            f"and {len(self.store_ids)} stores"
        )

    def restrict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Cut a built panel down to exactly these series.

        The step that makes the sample exact. Without it the panel carries every
        real series inside the product x store box, which is strictly more than
        was asked for.
        """
        if panel.empty:
            return panel
        keys = pd.MultiIndex.from_frame(self.pairs[["product_id", "store_id"]])
        panel_keys = pd.MultiIndex.from_frame(panel[["product_id", "store_id"]])
        return panel[panel_keys.isin(keys)].reset_index(drop=True)


def _listings(
    repository: DataRepository,
    *,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Observed product-store listings with their observation counts."""
    window = ""
    parameters: dict[str, object] = {}
    if start_date is not None:
        window += " WHERE date >= ?"
        parameters["start"] = start_date
    if end_date is not None:
        window += " AND date <= ?" if window else " WHERE date <= ?"
        parameters["end"] = end_date

    # Safe: `window` is built from the two module-internal literals above and
    # contains only `?` placeholders; both dates bind as parameters.
    projection = "SELECT product_id, store_id, COUNT(*) AS observations, SUM(units) AS total_units"
    grouping = "GROUP BY product_id, store_id"
    sql = f"{projection} FROM sales_daily{window} {grouping}"  # nosec B608

    listings = repository.execute_query(sql, parameters, max_rows=500_000)
    if listings.empty:
        raise DataAccessError("no product-store listings found in the requested window")

    # Sort before anything samples from this. DuckDB parallelises the aggregate
    # and returns groups in arbitrary order, so without a total ordering the
    # same seed selects different rows between runs - which is a reproducibility
    # failure that looks like a modelling result. Caught by the determinism test
    # rather than by reasoning.
    return listings.sort_values(["product_id", "store_id"]).reset_index(drop=True)


def _stratified_draw(
    listings: pd.DataFrame, *, n_series: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw proportionally from each volume decile.

    A uniform draw over listings is dominated by the long tail of slow movers,
    because that is most of the catalogue by count. The resulting sample is
    unrepresentative in the way that matters: it under-weights the high-volume
    series that carry the revenue and that WMAPE weights most heavily, so the
    headline metric ends up measuring accuracy on series nobody asks about.
    """
    if len(listings) <= n_series:
        return listings.reset_index(drop=True)

    working = listings.copy()
    # `duplicates="drop"` because a dataset with many tied volumes produces
    # non-unique decile edges, which qcut rejects outright.
    working["_decile"] = pd.qcut(
        working["total_units"].rank(method="first"), q=10, labels=False, duplicates="drop"
    )

    chosen: list[pd.DataFrame] = []
    deciles = sorted(working["_decile"].unique())
    per_decile = max(n_series // len(deciles), 1)

    for decile in deciles:
        block = working[working["_decile"] == decile]
        take = min(per_decile, len(block))
        indices = rng.choice(len(block), size=take, replace=False)
        chosen.append(block.iloc[indices])

    sample = pd.concat(chosen, ignore_index=True)

    # Integer division across deciles leaves a remainder; top up from whatever
    # is left so the requested count is honoured exactly.
    if len(sample) < n_series:
        taken = set(zip(sample["product_id"], sample["store_id"], strict=True))
        remaining = working[
            ~pd.Series(
                list(zip(working["product_id"], working["store_id"], strict=True)),
                index=working.index,
            ).isin(taken)
        ]
        shortfall = min(n_series - len(sample), len(remaining))
        if shortfall > 0:
            indices = rng.choice(len(remaining), size=shortfall, replace=False)
            sample = pd.concat([sample, remaining.iloc[indices]], ignore_index=True)

    # Sorted, so the returned pair set is byte-identical across runs regardless
    # of the order the deciles happened to contribute in.
    return (
        sample.drop(columns="_decile")
        .sort_values(["product_id", "store_id"])
        .reset_index(drop=True)
    )


def sample_series(
    repository: DataRepository,
    *,
    n_series: int = 800,
    start_date: date | None = None,
    end_date: date | None = None,
    seed: int = 42,
    stratify_by_volume: bool = True,
) -> SeriesSample:
    """Sample exactly ``n_series`` real product-store series.

    Store-clustered so the repository filter box stays close to the sample size.
    The clustering is the whole efficiency argument: pairs drawn uniformly across
    every store saturate the store list immediately, and the box is then the full
    catalogue regardless of how few pairs were requested.
    """
    listings = _listings(repository, start_date=start_date, end_date=end_date)
    total_series = len(listings)

    if n_series >= total_series:
        pairs = listings.reset_index(drop=True)
        return SeriesSample(
            pairs=pairs,
            product_ids=sorted(pairs["product_id"].unique().tolist()),
            store_ids=sorted(pairs["store_id"].unique().tolist()),
        )

    rng = np.random.default_rng(seed)

    # How many stores are needed to hold n_series listings, with slack for the
    # fact that stores carry different assortment sizes. Sorted for the same
    # determinism reason as the listings themselves.
    all_stores = np.sort(listings["store_id"].unique())
    listings_per_store = total_series / len(all_stores)
    n_stores = min(
        len(all_stores),
        max(1, int(np.ceil(n_series / listings_per_store * _BOX_TOLERANCE))),
    )
    store_indices = rng.choice(len(all_stores), size=n_stores, replace=False)
    stores = set(all_stores[store_indices])

    clustered = listings[listings["store_id"].isin(stores)]
    if len(clustered) < n_series:
        # The cluster came up short - unusual assortments, or a small dataset.
        # Widen rather than silently return fewer series than requested.
        logger.debug(
            "forecast.sampling_widened",
            clustered=len(clustered), requested=n_series,
        )
        clustered = listings

    pairs = (
        _stratified_draw(clustered, n_series=n_series, rng=rng)
        if stratify_by_volume
        else clustered.iloc[
            rng.choice(len(clustered), size=min(n_series, len(clustered)), replace=False)
        ]
        .sort_values(["product_id", "store_id"])
        .reset_index(drop=True)
    )

    sample = SeriesSample(
        pairs=pairs,
        product_ids=sorted(pairs["product_id"].unique().tolist()),
        store_ids=sorted(pairs["store_id"].unique().tolist()),
    )

    logger.info(
        "forecast.series_sampled",
        requested=n_series,
        sampled=len(sample),
        products=len(sample.product_ids),
        stores=len(sample.store_ids),
        # The number Step 4 got wrong. Worth logging every run: if this drifts
        # far above the sample size, the filter box has widened again.
        box_series_estimate=round(sample.box_size * total_series / (
            len(listings["product_id"].unique()) * len(all_stores)
        )),
    )
    return sample
