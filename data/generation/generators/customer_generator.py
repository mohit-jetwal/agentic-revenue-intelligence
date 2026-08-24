"""Customer dimension.

Non-sensitive attributes only (brief section 6): segment, region, loyalty tier,
acquisition channel, tenure. No names, addresses, contact details or anything
resembling PII - there is no analytical use for it here, and generating
realistic-looking personal data creates a handling problem for zero benefit.

Customers exist to give the transaction table a realistic distribution of
buyers, and to make segment-level analysis possible in Text-to-SQL and BI. The
demand simulation itself operates on the store's *segment mix*, not on
individual customers - see ``store_generator``.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream


def generate_customers(config: GenerationConfig, rngs: RngFactory) -> pd.DataFrame:
    """Build the customer master."""
    rng = rngs.get(Stream.CUSTOMER)
    n = config.scale.customers

    segment_names = list(config.customers.segments)
    segment_weights = np.array([config.customers.segments[s].weight for s in segment_names])
    segment_weights = segment_weights / segment_weights.sum()
    segments = rng.choice(segment_names, size=n, p=segment_weights)

    region_names = list(config.stores.regions)
    region_weights = np.array([config.stores.regions[r].weight for r in region_names])
    region_weights = region_weights / region_weights.sum()
    regions = rng.choice(region_names, size=n, p=region_weights)

    tiers = np.array(config.customers.loyalty_tiers, dtype=object)
    tier_weights = np.array(config.customers.loyalty_weights, dtype=float)
    tier_weights = tier_weights / tier_weights.sum()

    # Loyalty tier correlates with segment: Premium and Loyal shoppers skew
    # higher. Independent draws would leave the two columns uncorrelated, which
    # is both unrealistic and removes a relationship worth discovering in BI.
    tier_shift = {"Value": -0.7, "Regular": 0.0, "Premium": 1.0, "Loyal": 1.2, "Occasional": -0.5}
    base_scores = rng.normal(0.0, 1.0, size=n)
    shifts = np.array([tier_shift.get(str(s), 0.0) for s in segments])
    scores = base_scores + shifts
    # Map scores onto tiers by quantile so the configured tier shares still hold.
    cutoffs = np.quantile(scores, np.cumsum(tier_weights)[:-1])
    tier_index = np.searchsorted(cutoffs, scores)
    loyalty = tiers[np.clip(tier_index, 0, len(tiers) - 1)]

    # Tenure: most customers acquired before the window, some during it.
    max_tenure_days = 8 * 365
    tenure = rng.integers(30, max_tenure_days, size=n)
    customer_since = [config.time.start_date - timedelta(days=int(d)) for d in tenure]
    # A minority join during the observed period.
    joined_later = rng.random(n) < 0.18
    horizon = (config.time.end_date - config.time.start_date).days
    for i in np.flatnonzero(joined_later):
        customer_since[i] = config.time.start_date + timedelta(
            days=int(rng.integers(0, max(1, horizon)))
        )

    return pd.DataFrame(
        {
            "customer_id": [f"C{i + 1:07d}" for i in range(n)],
            "segment": pd.Series(segments, dtype="string"),
            "region": pd.Series(regions, dtype="string"),
            "loyalty_tier": pd.Series(loyalty, dtype="string"),
            "acquisition_channel": pd.Series(
                rng.choice(config.customers.acquisition_channels, size=n), dtype="string"
            ),
            "customer_since": customer_since,
        }
    )
