"""Dataset generation pipeline.

Composes the generators in dependency order and streams the large facts out one
date chunk at a time.

Two orderings carry the design:

* **Ground truth is drawn before any driver series exists.** Elasticities and
  promotion response curves are parameters of the world, not summaries of the
  output, so they must precede it. Deriving them afterwards would be circular
  and would make the recovery tests meaningless.
* **Scenarios are injected into the driver matrices, not into the output.**
  Raising a price path and letting demand respond means the effect propagates to
  substitutes and complements automatically, and gets censored by inventory like
  anything else. Editing sales afterwards would produce a number without a cause.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config.settings import get_settings
from app.observability.logging import get_logger
from data.generation.config import GenerationConfig, load_config
from data.generation.generators.calendar_generator import generate_calendar
from data.generation.generators.competitor_generator import generate_competitor_pricing
from data.generation.generators.customer_generator import generate_customers
from data.generation.generators.pricing_generator import (
    PRICE_CHANGE_REASONS,
    generate_cost_index,
    generate_price_paths,
)
from data.generation.generators.product_generator import (
    analytical_columns as product_columns,
)
from data.generation.generators.product_generator import generate_products
from data.generation.generators.product_relationship_generator import (
    generate_product_relationships,
)
from data.generation.generators.promotion_generator import (
    generate_promotions,
    generate_trade_promotions,
)
from data.generation.generators.sales_generator import (
    build_panel_context,
    initial_inventory,
    simulate_chunk,
)
from data.generation.generators.store_generator import (
    analytical_columns as store_columns,
)
from data.generation.generators.store_generator import generate_listings, generate_stores
from data.generation.generators.transaction_generator import generate_transactions
from data.generation.ground_truth import draw_ground_truth
from data.generation.quality import (
    CorruptionReport,
    corrupt_inventory,
    corrupt_promotions,
    corrupt_sales,
)
from data.generation.rng import RngFactory
from data.generation.scenarios.injector import ScenarioInjector
from data.generation.writer import DatasetWriter

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Summary of a completed run."""

    root: Path
    config: GenerationConfig
    row_counts: dict[str, int]
    corruption: dict[str, dict[str, int]]
    scenarios: list[dict[str, Any]]
    duration_seconds: float
    gold_hash: str

    def summary(self) -> str:
        lines = [
            f"dataset_version : {self.config.dataset_version}",
            f"seed            : {self.config.seed}",
            f"config_hash     : {self.config.config_hash()}",
            f"output          : {self.root}",
            f"duration        : {self.duration_seconds:.1f}s",
            f"scenarios       : {len(self.scenarios)}",
            "",
            "rows:",
        ]
        for name, count in sorted(self.row_counts.items()):
            lines.append(f"  {name:<24} {count:>12,}")
        lines.append(f"  {'TOTAL':<24} {sum(self.row_counts.values()):>12,}")
        return "\n".join(lines)


def _chunk_bounds(
    n_days: int, calendar: pd.DataFrame, chunk_months: int
) -> list[tuple[int, int, str]]:
    """Split the horizon into ``(start, end, label)`` windows on month boundaries.

    Month-aligned so partitions correspond to a business period rather than an
    arbitrary row count - which makes partition pruning useful when a model asks
    for "last quarter" rather than merely making files smaller.
    """
    periods = pd.PeriodIndex(pd.to_datetime(calendar["date"]), freq="M")
    unique = list(dict.fromkeys(periods.astype(str)))

    bounds: list[tuple[int, int, str]] = []
    for i in range(0, len(unique), chunk_months):
        group = unique[i : i + chunk_months]
        mask = periods.astype(str).isin(group)
        indices = np.flatnonzero(mask)
        bounds.append((int(indices[0]), int(indices[-1]) + 1, group[0].replace("-", "")))
    return bounds


def generate_dataset(
    profile: str = "dev",
    *,
    seed: int | None = None,
    output_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> GenerationResult:
    """Generate a complete synthetic dataset."""
    started = time.perf_counter()

    merged: dict[str, Any] = dict(overrides or {})
    if seed is not None:
        merged["seed"] = seed
    config = load_config(profile, overrides=merged)

    settings = get_settings()
    root = output_root or settings.resolve(settings.data.parquet_root).parent
    root = Path(root)

    writer = DatasetWriter(root=root, compression=config.output.compression)
    writer.reset()

    rngs = RngFactory(config.seed)
    log = logger.bind(profile=profile, seed=config.seed)
    log.info("generation.started", output=str(root))

    # --- dimensions ---------------------------------------------------------
    calendar = generate_calendar(
        config.time.start_date, config.time.end_date, geography=config.time.geography
    )
    products = generate_products(config, rngs)
    relationships = generate_product_relationships(products, config, rngs)
    stores = generate_stores(config, rngs)
    customers = generate_customers(config, rngs)
    listings = generate_listings(products, stores, config, rngs)

    # Channel is carried on the listing so the sales fact can report it without
    # a join inside the hot loop.
    listings["_channel"] = listings["store_id"].map(stores.set_index("store_id")["channel"])

    log.info(
        "generation.dimensions_ready",
        products=len(products),
        stores=len(stores),
        customers=len(customers),
        listings=len(listings),
        days=len(calendar),
    )

    # --- ground truth: drawn before anything observable ---------------------
    ground_truth = draw_ground_truth(products, relationships, config, rngs)

    # --- drivers ------------------------------------------------------------
    cost_index = generate_cost_index(calendar, config, rngs)
    price_paths = generate_price_paths(
        listings, products, stores, calendar, cost_index, config, rngs
    )
    competitor_paths = generate_competitor_pricing(products, calendar, cost_index, config, rngs)
    promotion_paths = generate_promotions(
        listings, products, stores, calendar, ground_truth, config, rngs
    )

    # --- scenario injection, before demand is simulated ---------------------
    n_pairs, n_days = price_paths.regular_price.shape
    scenario_term = np.zeros((n_pairs, n_days), dtype=np.float32)
    supply_cap = np.ones((n_pairs, n_days), dtype=np.float32)

    injector = ScenarioInjector(listings, products, stores, calendar, relationships, config, rngs)
    if config.scenarios.enabled:
        price_products = set(injector.inject_price_increase(price_paths))
        injector.inject_promotions(promotion_paths, ground_truth.promo_response)
        injector.inject_stockouts(supply_cap, exclude=price_products)
        injector.inject_competitor_price_cut(competitor_paths)
        injector.inject_regional_shock(scenario_term, supply_cap)
        injector.register_seasonal_and_launch()
    ground_truth.scenarios = injector.registry()
    ground_truth.write(root)

    log.info("generation.drivers_ready", scenarios=len(ground_truth.scenarios))

    # --- panel context ------------------------------------------------------
    context = build_panel_context(
        listings,
        products,
        stores,
        calendar,
        price_paths,
        promotion_paths,
        competitor_paths,
        ground_truth,
        config,
        rngs,
    )
    context.scenario_term = scenario_term
    context.supply_cap = supply_cap

    # --- chunked simulation -------------------------------------------------
    inventory_state = initial_inventory(context.base_demand, config, rngs)
    corruption = CorruptionReport()
    bounds = _chunk_bounds(len(calendar), calendar, config.output.chunk_months)

    transaction_offset = 0
    promotion_units: dict[str, int] = {}
    weekly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    # Head rows from the first chunk of each partitioned fact. Without this the
    # committed CSV samples would cover only the small dimension tables and miss
    # sales_daily entirely - the one table a reader most wants to see.
    fact_samples: dict[str, pd.DataFrame] = {}

    for chunk_index, (day_start, day_end, label) in enumerate(bounds):
        result = simulate_chunk(
            context, day_start, day_end, chunk_index, config, rngs, inventory_state
        )

        # Price-change reason codes become labels only at write time; carrying
        # strings through the matrices would cost memory for no benefit.
        result.pricing["price_change_reason"] = PRICE_CHANGE_REASONS[
            result.pricing["price_change_reason"].to_numpy()
        ]

        writer.write_partition(result.sales, "sales_daily", label)
        writer.write_partition(result.inventory, "inventory", label)
        writer.write_partition(result.pricing, "pricing", label)
        writer.write_partition(result.latent, "latent_demand", label, layer="ground_truth")

        transactions = generate_transactions(
            result.sales, customers, stores, config, rngs, chunk_index, transaction_offset
        )
        transaction_offset += len(transactions)
        writer.write_partition(transactions, "sales_transactions", label)

        if chunk_index == 0 and config.output.write_samples:
            rows = config.output.sample_rows
            fact_samples = {
                "sales_daily": result.sales.head(rows),
                "inventory": result.inventory.head(rows),
                "pricing": result.pricing.head(rows),
                "sales_transactions": transactions.head(rows),
            }

        # Promotional volume, accumulated for spend reconciliation below.
        promoted = result.sales[result.sales["promotion_id"].notna()]
        if not promoted.empty:
            grouped = promoted.groupby("promotion_id")["units"].sum()
            for promotion_id, units in grouped.items():
                promotion_units[str(promotion_id)] = promotion_units.get(
                    str(promotion_id), 0
                ) + int(units)

        weekly_parts.append(_aggregate(result.sales, calendar, "W"))
        monthly_parts.append(_aggregate(result.sales, calendar, "M"))

        if config.output.write_bronze:
            writer.write_partition(
                corrupt_sales(result.sales, config.data_quality, rngs, corruption, chunk_index),
                "sales_daily",
                label,
                layer="bronze",
            )
            writer.write_partition(
                corrupt_inventory(
                    result.inventory, config.data_quality, rngs, corruption, chunk_index
                ),
                "inventory",
                label,
                layer="bronze",
            )

        log.debug("generation.chunk_written", chunk=label, rows=len(result.sales))

    # --- promotion spend, now that units are known --------------------------
    promotions = promotion_paths.frame.drop(
        columns=["_pair_index", "_start_day", "_end_day"], errors="ignore"
    )
    if not promotions.empty:
        units = promotions["promotion_id"].map(promotion_units).fillna(0).astype(float)
        promotions["promotion_units"] = units.astype(int)
        promotions["promotion_spend"] = (
            promotions["fixed_spend"] + promotions["spend_per_unit"] * units
        ).round(2)
        promotions = promotions.drop(columns=["fixed_spend", "spend_per_unit"])

    trade_promotions = generate_trade_promotions(promotion_paths.frame, products, config, rngs)

    # --- write remaining tables ---------------------------------------------
    gold_tables: dict[str, pd.DataFrame] = {
        "products": products[product_columns()],
        "stores": stores[store_columns()],
        "customers": customers,
        "calendar": calendar,
        "product_relationships": relationships,
        "promotions": promotions,
        "trade_promotions": trade_promotions,
        "competitor_pricing": competitor_paths.frame,
        "commodity_costs": cost_index,
        "sales_weekly": _combine(weekly_parts),
        "sales_monthly": _combine(monthly_parts),
    }
    for name, frame in gold_tables.items():
        writer.write_table(frame, name)

    if config.output.write_bronze:
        writer.write_table(
            corrupt_promotions(promotions, config.data_quality, rngs, corruption),
            "promotions",
            layer="bronze",
        )
        for name in ("products", "stores", "customers", "calendar", "product_relationships"):
            writer.write_table(gold_tables[name], name, layer="bronze")

    if config.output.write_samples:
        writer.write_samples(
            {**gold_tables, **fact_samples},
            get_settings().project_root / "data" / "sample",
            config.output.sample_rows,
        )

    # --- manifest -----------------------------------------------------------
    duration = time.perf_counter() - started
    manifest = {
        "dataset_version": config.dataset_version,
        "scenario_version": config.scenario_version,
        "profile": profile,
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "start_date": str(config.time.start_date),
        "end_date": str(config.time.end_date),
        "generation_seconds": round(duration, 2),
        "scenarios": len(ground_truth.scenarios),
        "data_quality_injected": corruption.to_dict(),
        "data_quality_total": corruption.total(),
    }
    writer.write_manifest(manifest)

    gold_hash = writer.content_hash("gold")
    log.info(
        "generation.completed",
        duration_seconds=round(duration, 1),
        total_rows=sum(writer.row_counts.values()),
        gold_hash=gold_hash[:12],
    )

    return GenerationResult(
        root=root,
        config=config,
        row_counts=dict(writer.row_counts),
        corruption=corruption.to_dict(),
        scenarios=ground_truth.scenarios,
        duration_seconds=duration,
        gold_hash=gold_hash,
    )


def _aggregate(sales: pd.DataFrame, calendar: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Roll daily sales up to a coarser period.

    Aggregates are precomputed rather than left to query time because BI and
    Text-to-SQL both want them constantly, and re-scanning millions of daily
    rows for "monthly revenue by region" is the single most common slow query in
    a lakehouse of this shape.
    """
    if sales.empty:
        return sales

    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["period"] = frame["date"].dt.to_period(freq).astype(str)

    grouped = (
        frame.groupby(["period", "product_id", "store_id"], observed=True)
        .agg(
            units=("units", "sum"),
            revenue=("revenue", "sum"),
            cost=("cost", "sum"),
            gross_profit=("gross_profit", "sum"),
            promo_days=("promotion_flag", "sum"),
            stockout_days=("stockout_flag", "sum"),
            days=("units", "size"),
            avg_selling_price=("selling_price", "mean"),
        )
        .reset_index()
    )
    grouped["avg_selling_price"] = grouped["avg_selling_price"].round(2)
    return grouped


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-chunk aggregates, re-summing periods split across chunks."""
    frames = [part for part in parts if not part.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["period", "product_id", "store_id"], observed=True)
        .agg(
            units=("units", "sum"),
            revenue=("revenue", "sum"),
            cost=("cost", "sum"),
            gross_profit=("gross_profit", "sum"),
            promo_days=("promo_days", "sum"),
            stockout_days=("stockout_days", "sum"),
            days=("days", "sum"),
            avg_selling_price=("avg_selling_price", "mean"),
        )
        .reset_index()
    )
