"""Store dimension.

Each store carries a demand scale (channel size), a regional multiplier, and -
importantly for brief section 20 - a customer segment mix.

The segment mix is what lets customer behaviour influence a panel that has no
customer dimension. A store in a Value-heavy catchment gets its products'
elasticities amplified; a Premium-heavy store gets them damped. The result is
genuine store-level heterogeneity in price response that a hierarchical model in
Step 8 can discover, rather than one global elasticity per product with noise
sprinkled on top.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

_CITY_SUFFIXES = [
    "Central",
    "North",
    "South",
    "East",
    "West",
    "Park",
    "Plaza",
    "Junction",
    "Market",
]
_STORE_NAME_PREFIXES = [
    "Metro",
    "FreshMart",
    "ValueBazaar",
    "QuickShop",
    "GreenGrocer",
    "UrbanKart",
]


def generate_stores(config: GenerationConfig, rngs: RngFactory) -> pd.DataFrame:
    """Build the store master with latent demand scaling and segment mix."""
    rng = rngs.get(Stream.STORE)
    n = config.scale.stores

    channel_names = list(config.stores.channels)
    channel_weights = np.array([config.stores.channels[c].weight for c in channel_names])
    channel_weights = channel_weights / channel_weights.sum()
    channels = rng.choice(channel_names, size=n, p=channel_weights)

    region_names = list(config.stores.regions)
    region_weights = np.array([config.stores.regions[r].weight for r in region_names])
    region_weights = region_weights / region_weights.sum()
    regions = rng.choice(region_names, size=n, p=region_weights)

    segment_names = list(config.customers.segments)
    segment_base = np.array([config.customers.segments[s].weight for s in segment_names])
    segment_base = segment_base / segment_base.sum()

    start = config.time.start_date
    horizon_days = (config.time.end_date - start).days

    rows: list[dict[str, object]] = []
    for i in range(n):
        channel_name = str(channels[i])
        region_name = str(regions[i])
        channel = config.stores.channels[channel_name]
        region = config.stores.regions[region_name]

        state = str(rng.choice(region.states))
        city = f"{state.split()[0]} {rng.choice(_CITY_SUFFIXES)}"

        size_sqft = int(rng.integers(channel.size_sqft[0], max(channel.size_sqft[1], 1) + 1))
        demand_scale = float(rng.uniform(*channel.demand_scale))
        # Within-channel spread: two supermarkets are not identical.
        demand_scale *= float(np.exp(rng.normal(0.0, 0.22)))

        opening_date = start
        if rng.random() < config.stores.opened_mid_history_pct:
            opening_date = start + timedelta(days=int(rng.integers(30, max(31, horizon_days // 2))))

        # Dirichlet around the national mix: each store's catchment differs, but
        # the population-level segment shares still hold.
        mix = rng.dirichlet(segment_base * 22.0)

        # Effective price-sensitivity multiplier for this store, from its mix.
        sensitivity = float(
            sum(
                mix[j] * config.customers.segments[segment_names[j]].price_sensitivity
                for j in range(len(segment_names))
            )
        )
        promo_response = float(
            sum(
                mix[j] * config.customers.segments[segment_names[j]].promo_responsiveness
                for j in range(len(segment_names))
            )
        )

        row: dict[str, object] = {
            "store_id": f"S{i + 1:05d}",
            "store_name": f"{rng.choice(_STORE_NAME_PREFIXES)} {city}",
            "store_type": "Flagship" if size_sqft > 20000 else "Standard",
            "channel": channel_name,
            "region": region_name,
            "state": state,
            "city": city,
            "store_size_sqft": size_sqft,
            "opening_date": opening_date,
            # --- latent simulation attributes ---
            "_demand_scale": demand_scale,
            "_region_multiplier": region.demand_multiplier,
            "_price_sensitivity": sensitivity,
            "_promo_responsiveness": promo_response,
        }
        for j, segment in enumerate(segment_names):
            row[f"_mix_{segment.lower()}"] = float(mix[j])
        rows.append(row)

    return pd.DataFrame(rows)


def analytical_columns() -> list[str]:
    """Columns published to gold. Latent ``_`` attributes stay hidden."""
    return [
        "store_id",
        "store_name",
        "store_type",
        "channel",
        "region",
        "state",
        "city",
        "store_size_sqft",
        "opening_date",
    ]


def generate_listings(
    products: pd.DataFrame,
    stores: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> pd.DataFrame:
    """Decide which products are stocked in which stores.

    This is the panel spine - its size determines the row count of every daily
    fact. Listings are not random: a large hypermarket carries far more of the
    catalogue than a convenience store, so listing probability scales with the
    store's demand scale. That produces realistic assortment depth and, usefully,
    concentrates observations in the stores where volume actually is.
    """
    rng = rngs.get(Stream.LISTING)

    n_products = len(products)
    n_stores = len(stores)
    target_mean = config.scale.stores_per_product_mean
    target_std = config.scale.stores_per_product_std

    # How many stores each product is listed in.
    counts = rng.normal(target_mean, target_std, size=n_products)
    counts = np.clip(np.round(counts), 1, n_stores).astype(int)

    # Store attractiveness: bigger stores carry more SKUs.
    scale = stores["_demand_scale"].to_numpy(dtype=float)
    store_weights = scale / scale.sum()

    store_ids = stores["store_id"].to_numpy()
    product_ids = products["product_id"].to_numpy()

    listed_products: list[np.ndarray] = []
    listed_stores: list[np.ndarray] = []

    for i in range(n_products):
        k = int(counts[i])
        chosen = rng.choice(n_stores, size=k, replace=False, p=store_weights)
        listed_products.append(np.full(k, product_ids[i]))
        listed_stores.append(store_ids[chosen])

    listings = pd.DataFrame(
        {
            "product_id": np.concatenate(listed_products),
            "store_id": np.concatenate(listed_stores),
        }
    )
    listings["pair_index"] = np.arange(len(listings), dtype=np.int32)
    return listings
