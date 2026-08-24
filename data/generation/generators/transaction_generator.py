"""Transaction-level sales, derived from the daily panel.

Brief section 8 specifies a ``sales`` table with ``transaction_id`` and
``customer_id``. That is basket grain, but basket grain is the wrong place to
*model* - real CPG pricing and promotion science runs on aggregated POS panels,
and generating a row per basket for three years would produce tens of millions of
rows carrying no information the daily panel does not already hold.

So transactions are **derived**: a configurable sample of daily panel rows is
disaggregated into individual baskets. That satisfies the schema, supports
customer-segment analysis and Text-to-SQL variety, and keeps the modelling grain
where it belongs.

Customers are drawn according to the *store's* segment mix rather than uniformly,
so a Value-heavy store really does show more Value shoppers. Uniform assignment
would leave segment uncorrelated with everything else and make segment analysis
meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream


def generate_transactions(
    sales: pd.DataFrame,
    customers: pd.DataFrame,
    stores: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
    chunk_index: int,
    id_offset: int,
) -> pd.DataFrame:
    """Disaggregate a sample of daily rows into individual transactions."""
    rng = rngs.fresh(Stream.TRANSACTION, chunk_index)

    sellable = sales[sales["units"] > 0]
    if sellable.empty:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "date",
                "product_id",
                "store_id",
                "customer_id",
                "units",
                "regular_price",
                "selling_price",
                "discount_percentage",
                "revenue",
                "cost",
                "gross_profit",
                "promotion_id",
                "channel",
            ]
        )

    rate = config.scale.transaction_sample_rate
    sample_size = max(int(len(sellable) * rate), 1)
    sampled = sellable.iloc[
        rng.choice(len(sellable), size=min(sample_size, len(sellable)), replace=False)
    ]

    # Split each day's units into baskets. Most baskets contain one or two units
    # of a given SKU; a few contain more.
    units = sampled["units"].to_numpy()
    basket_sizes = np.clip(rng.geometric(0.62, size=len(sampled)), 1, 12)
    n_baskets = np.maximum(1, np.ceil(units / basket_sizes).astype(int))
    # Cap the fan-out: one very high-volume store-day should not explode into
    # thousands of rows and distort the table.
    n_baskets = np.minimum(n_baskets, 25)

    repeated = sampled.loc[sampled.index.repeat(n_baskets)].reset_index(drop=True)
    total = len(repeated)

    # Distribute the day's units across its baskets, preserving the total so
    # transaction sums reconcile back to the panel for the sampled rows.
    expanded_units = np.repeat(units, n_baskets)
    expanded_counts = np.repeat(n_baskets, n_baskets)
    base_units = expanded_units // expanded_counts
    remainder = expanded_units % expanded_counts
    position = np.concatenate([np.arange(k) for k in n_baskets])
    basket_units = base_units + (position < remainder).astype(int)
    basket_units = np.maximum(basket_units, 1)

    # Customer assignment weighted by each store's segment mix.
    segment_names = list(config.customers.segments)
    store_mix = stores.set_index("store_id")[
        [f"_mix_{segment.lower()}" for segment in segment_names]
    ]
    customers_by_segment = {
        segment: customers.loc[customers["segment"] == segment, "customer_id"].to_numpy()
        for segment in segment_names
    }

    store_ids = repeated["store_id"].to_numpy()
    unique_stores, inverse = np.unique(store_ids, return_inverse=True)
    customer_ids = np.empty(total, dtype=object)

    for position_index, store_id in enumerate(unique_stores):
        rows = np.flatnonzero(inverse == position_index)
        if rows.size == 0:
            continue
        try:
            mix = store_mix.loc[store_id].to_numpy(dtype=float)
        except KeyError:
            mix = np.ones(len(segment_names)) / len(segment_names)
        mix = np.clip(mix, 1e-6, None)
        mix = mix / mix.sum()

        segments = rng.choice(len(segment_names), size=rows.size, p=mix)
        for segment_index, segment in enumerate(segment_names):
            selected = rows[segments == segment_index]
            if selected.size == 0:
                continue
            pool = customers_by_segment[segment]
            if pool.size == 0:
                pool = customers["customer_id"].to_numpy()
            customer_ids[selected] = rng.choice(pool, size=selected.size)

    selling_price = repeated["selling_price"].to_numpy()
    unit_cost = np.divide(
        repeated["cost"].to_numpy(),
        np.clip(repeated["units"].to_numpy(), 1, None),
    )
    revenue = np.round(basket_units * selling_price, 2)
    cost = np.round(basket_units * unit_cost, 2)

    return pd.DataFrame(
        {
            "transaction_id": [f"T{id_offset + i + 1:010d}" for i in range(total)],
            "date": repeated["date"].to_numpy(),
            "product_id": repeated["product_id"].to_numpy(),
            "store_id": store_ids,
            "customer_id": customer_ids,
            "units": basket_units.astype(np.int32),
            "regular_price": repeated["regular_price"].to_numpy(),
            "selling_price": selling_price,
            "discount_percentage": repeated["discount_percentage"].to_numpy(),
            "revenue": revenue,
            "cost": cost,
            "gross_profit": np.round(revenue - cost, 2),
            "promotion_id": repeated["promotion_id"].to_numpy(),
            "channel": repeated["channel"].to_numpy(),
        }
    )
