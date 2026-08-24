"""Product dimension.

Builds a realistic hierarchy (category -> subcategory -> brand -> SKU) and, more
importantly, gives each product the latent attributes the demand simulation
needs: base demand, cost, reference price, elasticity tier and lifecycle window.

Base demand is drawn log-normally rather than uniformly. Real CPG catalogues are
extremely skewed - a handful of hero SKUs carry most of the volume while a long
tail barely moves. That skew is what makes WMAPE a more honest forecasting metric
than MAPE, and a uniform draw would quietly remove the reason WMAPE matters.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

_BRAND_PREFIXES = [
    "Aura",
    "Nova",
    "Everest",
    "Sunrise",
    "Crest",
    "Verve",
    "Bloom",
    "Zenith",
    "Harvest",
    "Orchid",
    "Summit",
    "Lotus",
    "Pioneer",
    "Cascade",
    "Ember",
]
_BRAND_SUFFIXES = ["", " Gold", " Select", " Naturals", " Pro", " Classic", " Fresh"]


def _assign_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` across weighted keys, guaranteeing at least one each.

    Largest-remainder allocation so the parts sum exactly to the total; naive
    rounding would drift and silently produce a different product count than the
    profile asked for.
    """
    names = list(weights)
    raw = np.array([weights[n] for n in names], dtype=float)
    raw = raw / raw.sum()

    exact = raw * total
    counts = np.floor(exact).astype(int)
    counts = np.maximum(counts, 1)

    # Reconcile against the target after the minimum-one adjustment.
    while counts.sum() > total and counts.max() > 1:
        counts[int(np.argmax(counts))] -= 1
    remainder = total - counts.sum()
    if remainder > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    return dict(zip(names, (int(c) for c in counts), strict=True))


def generate_products(config: GenerationConfig, rngs: RngFactory) -> pd.DataFrame:
    """Build the product master with its hidden simulation attributes."""
    rng = rngs.get(Stream.PRODUCT)

    weights = {name: cat.weight for name, cat in config.categories.items()}
    per_category = _assign_counts(config.scale.products, weights)

    start = config.time.start_date
    end = config.time.end_date
    horizon_days = (end - start).days

    rows: list[dict[str, object]] = []
    index = 0

    for category_name, count in per_category.items():
        category = config.categories[category_name]
        n_brands = int(
            rng.integers(config.brands_per_category[0], config.brands_per_category[1] + 1)
        )
        brands = [
            f"{rng.choice(_BRAND_PREFIXES)}{rng.choice(_BRAND_SUFFIXES)}" for _ in range(n_brands)
        ]
        # Deduplicate while preserving order, so a collision does not silently
        # merge two brands into one.
        brands = list(dict.fromkeys(brands)) or [f"{category_name} Brand"]

        for _ in range(count):
            index += 1
            product_id = f"P{index:05d}"
            subcategory = str(rng.choice(category.subcategories))
            brand = str(rng.choice(brands))
            pack_size = str(rng.choice(config.pack_sizes))

            unit_cost = float(rng.uniform(*category.unit_cost))
            margin = float(rng.uniform(*category.margin))
            base_price = round(unit_cost / max(1e-6, 1.0 - margin), 2)

            # Log-normal: a few hero SKUs, a long slow tail.
            low, high = category.base_demand
            mu = np.log((low + high) / 2.0)
            base_demand = float(np.exp(rng.normal(mu, 0.55)))
            base_demand = float(np.clip(base_demand, low * 0.35, high * 2.5))

            launch_date = start
            discontinue_date: date | None = None
            status = "Active"

            if rng.random() < config.lifecycle.launched_mid_history_pct:
                # Launch somewhere in the first three quarters, leaving room for
                # the ramp and enough post-launch history to model.
                offset = int(rng.integers(60, max(61, int(horizon_days * 0.7))))
                launch_date = start + timedelta(days=offset)
                status = "Launched"
            elif rng.random() < config.lifecycle.discontinued_pct:
                offset = int(rng.integers(int(horizon_days * 0.6), horizon_days))
                discontinue_date = start + timedelta(days=offset)
                status = "Discontinued"

            rows.append(
                {
                    "product_id": product_id,
                    "product_name": f"{brand} {subcategory} {pack_size}",
                    "brand": brand,
                    "category": category_name,
                    "subcategory": subcategory,
                    "pack_size": pack_size,
                    "unit_cost": round(unit_cost, 2),
                    "base_price": base_price,
                    "launch_date": launch_date,
                    "discontinue_date": discontinue_date,
                    "product_status": status,
                    # --- latent simulation attributes (not analytical columns) ---
                    "_base_demand": base_demand,
                    "_margin": margin,
                }
            )

    frame = pd.DataFrame(rows)

    # Category-level annual trend, shared by every product in the category so
    # that category-level aggregates show a coherent direction rather than
    # cancelling noise.
    category_trend = {
        name: float(rng.uniform(*config.demand.trend_annual)) for name in config.categories
    }
    frame["_trend_annual"] = frame["category"].map(category_trend).astype(float)
    # Product-specific deviation around its category trend.
    frame["_trend_annual"] += rng.normal(0.0, 0.03, size=len(frame))

    frame["_dispersion"] = rng.uniform(*config.demand.negbinom_dispersion, size=len(frame))

    return frame


def analytical_columns() -> list[str]:
    """Columns published to the gold layer.

    Everything prefixed with ``_`` is a hidden simulation parameter and must not
    reach the analytical tables - exposing base demand or the true trend would
    hand future models the answer they are supposed to estimate.
    """
    return [
        "product_id",
        "product_name",
        "brand",
        "category",
        "subcategory",
        "pack_size",
        "unit_cost",
        "base_price",
        "launch_date",
        "discontinue_date",
        "product_status",
    ]
