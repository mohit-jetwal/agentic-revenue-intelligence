"""Sampling helpers (brief section 29).

Utilities for taking a workable slice of the panel without dragging the whole
thing into memory.

The design point section 29 is really asking about: **sample the keys, then
filter, rather than loading and then sampling**. Reading 6.7M rows to keep 5,000
is the pattern that works fine locally and falls over the moment the same code
meets a real warehouse. Every helper here returns *identifiers*, which the caller
then passes as a filter so the narrowing happens in the engine.

Sampling is seeded, because an unseeded sample makes a model comparison
meaningless - two runs would differ for reasons unrelated to the change being
tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataAccessError, DataRepository

logger = get_logger(__name__)

DEFAULT_SEED = 42


@dataclass(frozen=True)
class PanelSample:
    """A sampled slice of the panel, expressed as filters.

    Passed straight into repository ``get_*`` calls, so the narrowing is pushed
    down rather than applied after loading.
    """

    product_ids: list[str]
    store_ids: list[str]
    start_date: date
    end_date: date

    def describe(self) -> str:
        return (
            f"{len(self.product_ids)} products x {len(self.store_ids)} stores, "
            f"{self.start_date} -> {self.end_date}"
        )


def sample_products(
    repository: DataRepository,
    *,
    n: int = 20,
    category: str | None = None,
    seed: int = DEFAULT_SEED,
    prefer_well_observed: bool = True,
) -> list[str]:
    """Sample product identifiers.

    ``prefer_well_observed`` weights toward products with a longer sellable
    history. A uniform sample over the catalogue pulls in newly-launched and
    discontinued SKUs whose series are mostly empty, and a feature test on those
    fails for reasons of sample size rather than correctness - which sends you
    debugging the wrong thing.
    """
    products = repository.get_products(category=category)
    if products.empty:
        raise DataAccessError("no products available to sample")

    if prefer_well_observed and "product_status" in products.columns:
        active = products[products["product_status"] == "Active"]
        if len(active) >= n:
            products = active

    rng = np.random.default_rng(seed)
    n = min(n, len(products))
    chosen = rng.choice(products["product_id"].to_numpy(), size=n, replace=False)
    return [str(p) for p in chosen]


def sample_stores(
    repository: DataRepository,
    *,
    n: int = 10,
    region: str | None = None,
    channel: str | None = None,
    seed: int = DEFAULT_SEED,
    stratify_by_channel: bool = True,
) -> list[str]:
    """Sample store identifiers, stratified by channel by default.

    Channel drives demand scale by an order of magnitude - a hypermarket and a
    convenience store are not interchangeable. An unstratified sample of ten
    stores can easily draw nine convenience stores, and any feature statistic
    computed on it then describes convenience retail rather than the business.
    """
    stores = repository.get_stores(region=region, channel=channel)
    if stores.empty:
        raise DataAccessError("no stores available to sample")

    rng = np.random.default_rng(seed)
    n = min(n, len(stores))

    if not stratify_by_channel or channel is not None or "channel" not in stores.columns:
        chosen = rng.choice(stores["store_id"].to_numpy(), size=n, replace=False)
        return [str(s) for s in chosen]

    # Proportional allocation with at least one per channel, so small channels
    # are represented rather than rounded away.
    groups = list(stores.groupby("channel", observed=True))
    per_group = max(1, n // max(len(groups), 1))

    picked: list[str] = []
    for _, group in groups:
        take = min(per_group, len(group))
        drawn = rng.choice(group["store_id"].to_numpy(), size=take, replace=False)
        picked.extend(str(s) for s in drawn)

    # Top up to n from whatever remains.
    if len(picked) < n:
        remaining = stores[~stores["store_id"].isin(picked)]["store_id"].to_numpy()
        if len(remaining):
            extra = rng.choice(remaining, size=min(n - len(picked), len(remaining)), replace=False)
            picked.extend(str(s) for s in extra)

    return picked[:n]


def sample_date_range(
    repository: DataRepository,
    *,
    days: int = 180,
    end_date: date | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[date, date]:
    """A contiguous window from the available history.

    Contiguous, never a random scatter of dates: lags and rolling windows are
    meaningless over a discontinuous series, and a "sampled" panel with holes
    produces features that are quietly wrong rather than obviously missing.
    """
    calendar = repository.get_calendar()
    if calendar.empty:
        raise DataAccessError("calendar is empty; generate a dataset first")

    dates = pd.to_datetime(calendar["date"]).dt.date
    available_start, available_end = dates.min(), dates.max()

    if end_date is not None:
        window_end = min(end_date, available_end)
        return max(available_start, window_end - timedelta(days=days - 1)), window_end

    span = (available_end - available_start).days
    if span <= days:
        return available_start, available_end

    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, span - days))
    start = available_start + timedelta(days=offset)
    return start, start + timedelta(days=days - 1)


def sample_product_store_pairs(
    repository: DataRepository,
    *,
    n_pairs: int = 50,
    start_date: date | None = None,
    end_date: date | None = None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Sample listings that genuinely exist.

    Sampling products and stores independently and crossing them produces pairs
    that were never stocked together - most of the cross product, in fact, since
    a product is listed in a minority of stores. Those pairs come back empty and
    a caller sees an inexplicably sparse panel. This samples from observed
    listings instead.

    Uses a bounded aggregate query rather than reading the fact table, so the
    work stays in the engine.
    """
    window = ""
    parameters: dict[str, object] = {}
    if start_date is not None:
        window += " WHERE date >= ?"
        parameters["start"] = start_date
    if end_date is not None:
        window += " AND date <= ?" if window else " WHERE date <= ?"
        parameters["end"] = end_date

    # Safe: `window` is assembled from the two module-internal literals above and
    # contains only `?` placeholders - both dates bind as parameters.
    projection = "SELECT product_id, store_id, COUNT(*) AS observations"
    grouping = "GROUP BY product_id, store_id ORDER BY observations DESC"
    sql = f"{projection} FROM sales_daily{window} {grouping}"  # nosec B608

    listings = repository.execute_query(sql, parameters, max_rows=500_000)
    if listings.empty:
        raise DataAccessError("no product-store listings found in the requested window")

    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, len(listings))
    indices = rng.choice(len(listings), size=n_pairs, replace=False)
    sampled = listings.iloc[indices].reset_index(drop=True)

    logger.debug("sampling.pairs", requested=n_pairs, available=len(listings))
    return sampled


def sample_training_period(
    repository: DataRepository,
    *,
    train_days: int = 365,
    test_days: int = 28,
    end_date: date | None = None,
) -> tuple[tuple[date, date], tuple[date, date]]:
    """Contiguous train and test windows, test immediately after train.

    Chronological rather than random. A random split lets a model see the future
    while predicting the past, which inflates every metric and is the single most
    common way a forecasting evaluation ends up meaningless.

    Returns ``((train_start, train_end), (test_start, test_end))``.
    """
    calendar = repository.get_calendar()
    if calendar.empty:
        raise DataAccessError("calendar is empty; generate a dataset first")

    dates = pd.to_datetime(calendar["date"]).dt.date
    available_end = min(end_date, dates.max()) if end_date else dates.max()

    test_end = available_end
    test_start = test_end - timedelta(days=test_days - 1)
    train_end = test_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_days - 1)

    if train_start < dates.min():
        raise DataAccessError(
            f"not enough history: need {train_days + test_days} days ending "
            f"{available_end}, but data starts at {dates.min()}"
        )

    return (train_start, train_end), (test_start, test_end)


def build_panel_sample(
    repository: DataRepository,
    *,
    n_products: int = 20,
    n_stores: int = 10,
    days: int = 180,
    seed: int = DEFAULT_SEED,
    end_date: date | None = None,
    co_listed: bool = True,
) -> PanelSample:
    """A ready-to-use filter set covering products and stores that co-exist.

    ``co_listed=True`` samples from **observed listings**, so the resulting
    product and store lists actually intersect. Sampling the two independently
    and crossing them is the obvious approach and it is wrong: a product is
    stocked in a minority of stores, so most of the cross product was never a
    real listing and the panel comes back far emptier than requested - or
    entirely empty, which then looks like a bug somewhere else entirely.
    """
    start, end = sample_date_range(repository, days=days, end_date=end_date, seed=seed)

    if not co_listed:
        sample = PanelSample(
            product_ids=sample_products(repository, n=n_products, seed=seed),
            store_ids=sample_stores(repository, n=n_stores, seed=seed),
            start_date=start,
            end_date=end,
        )
        logger.info("sampling.panel", sample=sample.describe(), seed=seed, co_listed=False)
        return sample

    pairs = sample_product_store_pairs(
        repository,
        n_pairs=max(n_products * n_stores, 200),
        start_date=start,
        end_date=end,
        seed=seed,
    )

    # Take the products with the widest presence in the sampled listings, then
    # the stores that carry the most of *those* products. That intersection is
    # what makes the resulting panel dense rather than mostly holes.
    top_products = (
        pairs.groupby("product_id")["observations"].sum().nlargest(n_products).index.tolist()
    )
    within = pairs[pairs["product_id"].isin(top_products)]
    top_stores = (
        within.groupby("store_id")["product_id"].nunique().nlargest(n_stores).index.tolist()
    )

    sample = PanelSample(
        product_ids=[str(p) for p in top_products],
        store_ids=[str(s) for s in top_stores],
        start_date=start,
        end_date=end,
    )
    logger.info("sampling.panel", sample=sample.describe(), seed=seed, co_listed=True)
    return sample
