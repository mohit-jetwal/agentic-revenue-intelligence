"""The demand simulation - the structural causal model behind every fact table.

Everything else in ``data/generation`` exists to feed this module. For product
``p`` in store ``s`` on day ``t``:

    log lambda[p,s,t] = log base_demand[p] + log store_scale[s] + region + channel
                      + trend[p,t] + annual_season[cat,t] + dow[t] + holiday + festival
                      + beta_own[p,s]  * log(price[p,s,t] / ref_price[p,s])
                      + sum_j beta_cross[p,j] * log(price[j,t] / ref_price[j])
                      + promo_lift[p,s,t] + pull_forward[p,s,t]
                      + gamma[p] * log(comp_price[p,t] / comp_ref[p])
                      + launch_ramp + regional_shock + epsilon

    latent_units   ~ NegativeBinomial(mean = exp(log lambda), dispersion)
    observed_units = min(latent_units, inventory_available)

Four decisions worth defending:

**Log-additive.** This makes a log-log panel regression the *correctly specified*
estimator, so Step 8 can be shown to recover truth rather than merely producing
a plausible number. It also makes every effect compose multiplicatively on the
demand scale, which is how retail demand actually behaves.

**Competitor price enters through its own reference**, not as
``log(comp / own)``. The ratio form smuggles a second own-price term into the
equation and contaminates beta_own - the coefficient Step 8 is trying to
recover would no longer be the parameter that was drawn.

**Negative binomial, not Gaussian noise.** Sales are counts. Real POS data is
over-dispersed relative to Poisson, and slow-moving SKUs genuinely sell zero
units on many days. Gaussian noise on a float would erase the zero-inflation and
make forecasting look easier than it is.

**Censoring is the point.** ``latent_units`` is the demand that existed;
``observed_units`` is what the till recorded. Keeping both, and exposing only the
latter, is what lets Step 4 be tested on whether it can tell a demand collapse
from a supply failure - the distinction brief section 18 is asking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.generation.calendar_math import annual_seasonality_series, launch_ramp, linear_trend
from data.generation.config import GenerationConfig
from data.generation.generators.competitor_generator import CompetitorPaths
from data.generation.generators.pricing_generator import PricePaths
from data.generation.generators.promotion_generator import PromotionPaths
from data.generation.ground_truth import GroundTruth
from data.generation.rng import RngFactory, Stream


@dataclass
class PanelContext:
    """Pre-computed, pair-aligned inputs to the demand equation.

    Built once and reused across every date chunk. Assembling this inside the
    chunk loop would repeat the joins and dominate runtime.
    """

    listings: pd.DataFrame
    products: pd.DataFrame
    stores: pd.DataFrame
    calendar: pd.DataFrame

    #: (pairs,) latent per-pair attributes.
    base_demand: np.ndarray
    beta_own: np.ndarray
    gamma: np.ndarray
    dispersion: np.ndarray
    trend_annual: np.ndarray
    unit_cost: np.ndarray
    launch_day: np.ndarray
    discontinue_day: np.ndarray

    #: (pairs, days) driver matrices.
    price_paths: PricePaths
    promotion_paths: PromotionPaths

    #: (pairs, days) seasonal + calendar log-space terms.
    calendar_term: np.ndarray
    #: (pairs, days) competitor log-ratio term, already multiplied by gamma.
    competitor_term: np.ndarray
    #: (pairs, days) cross-price log-space term.
    cross_term: np.ndarray
    #: (pairs, days) additive log-space shock from injected scenarios.
    scenario_term: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    #: (pairs, days) multiplicative availability cap from scenario stockouts.
    supply_cap: np.ndarray = field(default_factory=lambda: np.ones((0, 0), dtype=np.float32))


def build_panel_context(
    listings: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    price_paths: PricePaths,
    promotion_paths: PromotionPaths,
    competitor_paths: CompetitorPaths,
    ground_truth: GroundTruth,
    config: GenerationConfig,
    rngs: RngFactory,
) -> PanelContext:
    """Assemble every pair-aligned input the demand equation needs."""
    n_pairs = len(listings)
    n_days = len(calendar)

    product_lookup = products.set_index("product_id")
    store_lookup = stores.set_index("store_id")

    pair_products = listings["product_id"].to_numpy()

    base_demand = listings["product_id"].map(product_lookup["_base_demand"]).to_numpy(dtype=float)
    trend_annual = listings["product_id"].map(product_lookup["_trend_annual"]).to_numpy(dtype=float)
    dispersion = listings["product_id"].map(product_lookup["_dispersion"]).to_numpy(dtype=float)
    unit_cost = listings["product_id"].map(product_lookup["unit_cost"]).to_numpy(dtype=float)
    categories = listings["product_id"].map(product_lookup["category"]).to_numpy()

    demand_scale = listings["store_id"].map(store_lookup["_demand_scale"]).to_numpy(dtype=float)
    region_multiplier = (
        listings["store_id"].map(store_lookup["_region_multiplier"]).to_numpy(dtype=float)
    )
    price_sensitivity = (
        listings["store_id"].map(store_lookup["_price_sensitivity"]).to_numpy(dtype=float)
    )

    # Own-price elasticity: the product's true value, modulated by the store's
    # segment mix. This is how customer behaviour (section 20) reaches a panel
    # that has no customer axis, and it gives Step 8 real heterogeneity to model.
    product_elasticity = np.array(
        [ground_truth.own_elasticity[str(pid)] for pid in pair_products], dtype=float
    )
    beta_own = product_elasticity * price_sensitivity

    gamma = np.array(
        [ground_truth.competitor_sensitivity[str(pid)] for pid in pair_products], dtype=float
    )

    # --- calendar term: seasonality, day of week, holidays, trend, launch ----
    rng = rngs.get(Stream.DEMAND)
    day_index = np.arange(n_days, dtype=float)
    day_of_year = calendar["day_of_year"].to_numpy()
    day_of_week = calendar["day_of_week"].to_numpy()
    holiday = calendar["holiday_flag"].to_numpy()
    festival = calendar["festival_flag"].to_numpy()

    dow_multipliers = np.array(config.demand.day_of_week, dtype=float)
    dow_term = np.log(dow_multipliers[day_of_week])

    holiday_term = np.zeros(n_days)
    holiday_term[holiday] = np.log(
        rng.uniform(*config.demand.holiday_multiplier, size=int(holiday.sum()))
    )
    festival_mask = festival & ~holiday
    holiday_term[festival_mask] = np.log(
        rng.uniform(*config.demand.festival_multiplier, size=int(festival_mask.sum()))
    )

    seasonal_by_category = {
        name: annual_seasonality_series(
            day_of_year, category.seasonal_amplitude, category.seasonal_peak_month
        )
        for name, category in config.categories.items()
    }
    seasonal = np.vstack([seasonal_by_category[str(c)] for c in categories]).astype(np.float32)

    trend = np.vstack([linear_trend(day_index, float(r)) for r in trend_annual]).astype(np.float32)

    calendar_term = seasonal + trend
    calendar_term += (dow_term + holiday_term).astype(np.float32)[np.newaxis, :]

    # --- lifecycle: launch ramp and discontinuation --------------------------
    start_date = calendar["date"].iloc[0]
    launch_dates = listings["product_id"].map(product_lookup["launch_date"])
    launch_day = np.array(
        [max((d - start_date).days, 0) if pd.notna(d) else 0 for d in launch_dates],
        dtype=np.int32,
    )
    discontinue_dates = listings["product_id"].map(product_lookup["discontinue_date"])
    discontinue_day = np.array(
        [(d - start_date).days if pd.notna(d) else n_days + 1 for d in discontinue_dates],
        dtype=np.int32,
    )

    ramp_days = config.lifecycle.launch_ramp_days
    for i in range(n_pairs):
        if launch_day[i] > 0:
            days_since = day_index - launch_day[i]
            ramp = launch_ramp(days_since, ramp_days)
            # Below the launch date there is no distribution at all.
            ramp[day_index < launch_day[i]] = 0.0
            calendar_term[i] += np.log(np.clip(ramp, 1e-4, None)).astype(np.float32)

    # --- competitor term -----------------------------------------------------
    product_positions = {str(pid): i for i, pid in enumerate(products["product_id"])}
    competitor_rows = np.array([product_positions[str(pid)] for pid in pair_products])
    comp_price = competitor_paths.mean_price[competitor_rows]
    comp_reference = competitor_paths.reference_price[competitor_rows][:, np.newaxis]
    competitor_term = (
        gamma[:, np.newaxis]
        * np.log(np.clip(comp_price / np.clip(comp_reference, 1e-6, None), 1e-6, None))
    ).astype(np.float32)

    # --- cross-price term ----------------------------------------------------
    cross_term = _build_cross_term(
        listings=listings,
        products=products,
        price_paths=price_paths,
        promotion_paths=promotion_paths,
        ground_truth=ground_truth,
    )

    return PanelContext(
        listings=listings,
        products=products,
        stores=stores,
        calendar=calendar,
        base_demand=base_demand * demand_scale * region_multiplier,
        beta_own=beta_own,
        gamma=gamma,
        dispersion=dispersion,
        trend_annual=trend_annual,
        unit_cost=unit_cost,
        launch_day=launch_day,
        discontinue_day=discontinue_day,
        price_paths=price_paths,
        promotion_paths=promotion_paths,
        calendar_term=calendar_term,
        competitor_term=competitor_term,
        cross_term=cross_term,
        scenario_term=np.zeros((n_pairs, n_days), dtype=np.float32),
        supply_cap=np.ones((n_pairs, n_days), dtype=np.float32),
    )


def _build_cross_term(
    listings: pd.DataFrame,
    products: pd.DataFrame,
    price_paths: PricePaths,
    promotion_paths: PromotionPaths,
    ground_truth: GroundTruth,
) -> np.ndarray:
    """Cross-price effects, computed at product level within each store.

    Substitution happens on a shelf: shoppers swap between products they can see
    side by side, so the effect is computed store by store, using the *selling*
    price of the related product in that same store. Using a national average
    price would blur exactly the store-level variation Step 9 needs.

    Iterates only over declared relationships rather than all product pairs -
    a dense scan would be O(products^2) per store for a matrix that is ~99% zero.
    """
    n_pairs = len(listings)
    n_days = price_paths.regular_price.shape[1]
    cross_term = np.zeros((n_pairs, n_days), dtype=np.float32)

    if not ground_truth.cross_elasticity:
        return cross_term

    # Locate each (product, store) listing so a relationship can find the
    # related product's row in the same store.
    position: dict[tuple[str, str], int] = {
        (str(p), str(s)): i
        for i, (p, s) in enumerate(zip(listings["product_id"], listings["store_id"], strict=True))
    }

    selling_price = price_paths.regular_price * (1.0 - promotion_paths.discount)
    reference = price_paths.reference_price[:, np.newaxis]
    log_ratio = np.log(np.clip(selling_price / np.clip(reference, 1e-6, None), 1e-6, None))

    for i in range(n_pairs):
        target_product = str(listings["product_id"].iloc[i])
        related = ground_truth.cross_elasticity.get(target_product)
        if not related:
            continue
        store_id = str(listings["store_id"].iloc[i])
        accumulator = np.zeros(n_days, dtype=np.float32)
        for source_product, coefficient in related.items():
            j = position.get((source_product, store_id))
            if j is None:
                # The related product is not stocked in this store, so there is
                # nothing to substitute to. Correct behaviour, not a gap.
                continue
            accumulator += (coefficient * log_ratio[j]).astype(np.float32)
        cross_term[i] = accumulator

    return cross_term


@dataclass
class ChunkResult:
    """Generated facts for one date window."""

    sales: pd.DataFrame
    inventory: pd.DataFrame
    pricing: pd.DataFrame
    latent: pd.DataFrame


def simulate_chunk(
    context: PanelContext,
    day_start: int,
    day_end: int,
    chunk_index: int,
    config: GenerationConfig,
    rngs: RngFactory,
    inventory_state: np.ndarray,
) -> ChunkResult:
    """Simulate demand, inventory and sales for ``[day_start, day_end)``.

    ``inventory_state`` is the closing position carried in from the previous
    chunk and is mutated in place, so stock levels remain continuous across
    partition boundaries rather than resetting each time.
    """
    rng = rngs.fresh(Stream.DEMAND, chunk_index)
    inventory_rng = rngs.fresh(Stream.INVENTORY, chunk_index)

    n_pairs = len(context.listings)
    window = slice(day_start, day_end)
    n_days = day_end - day_start
    day_index = np.arange(day_start, day_end)

    price = context.price_paths.regular_price[:, window].astype(np.float64)
    discount = context.promotion_paths.discount[:, window].astype(np.float64)
    selling_price = np.round(price * (1.0 - discount), 2)

    reference = context.price_paths.reference_price[:, np.newaxis].astype(np.float64)
    log_price_ratio = np.log(np.clip(selling_price / np.clip(reference, 1e-6, None), 1e-6, None))

    # --- assemble log-demand -------------------------------------------------
    log_lambda = np.log(np.clip(context.base_demand, 1e-6, None))[:, np.newaxis]
    log_lambda = log_lambda + context.calendar_term[:, window]
    log_lambda = log_lambda + context.beta_own[:, np.newaxis] * log_price_ratio
    log_lambda = log_lambda + context.cross_term[:, window]
    log_lambda = log_lambda + context.promotion_paths.lift[:, window]
    log_lambda = log_lambda + context.promotion_paths.pull_forward[:, window]
    log_lambda = log_lambda + context.competitor_term[:, window]
    log_lambda = log_lambda + context.scenario_term[:, window]

    # Residual noise, then occasional anomalies so the series is not too clean.
    log_lambda = log_lambda + rng.normal(0.0, config.demand.noise_sigma, size=(n_pairs, n_days))
    anomalies = rng.random((n_pairs, n_days)) < config.demand.anomaly_rate
    if anomalies.any():
        magnitude = rng.uniform(*config.demand.anomaly_magnitude, size=int(anomalies.sum()))
        log_lambda[anomalies] += np.log(magnitude)

    # Products outside their lifecycle window contribute nothing.
    active = (day_index[np.newaxis, :] >= context.launch_day[:, np.newaxis]) & (
        day_index[np.newaxis, :] < context.discontinue_day[:, np.newaxis]
    )

    mean_demand = np.exp(np.clip(log_lambda, -12.0, 12.0))
    mean_demand = np.where(active, mean_demand, 0.0)

    # --- count draw ----------------------------------------------------------
    # Negative binomial via its gamma-Poisson mixture: draw a gamma-distributed
    # rate then a Poisson count. Over-dispersed, integer, and zero-inflated for
    # slow movers - what POS data actually looks like.
    dispersion = context.dispersion[:, np.newaxis]
    safe_mean = np.clip(mean_demand, 1e-9, None)
    rate = rng.gamma(shape=dispersion, scale=safe_mean / dispersion)
    latent_units = rng.poisson(np.clip(rate, 0.0, 1e7)).astype(np.int32)
    latent_units = np.where(active, latent_units, 0)

    # --- inventory and censoring --------------------------------------------
    opening, received, sold, closing, stockout = _simulate_inventory(
        latent_units=latent_units,
        inventory_state=inventory_state,
        supply_cap=context.supply_cap[:, window],
        config=config,
        rng=inventory_rng,
    )

    observed_units = sold
    revenue = np.round(observed_units * selling_price, 2)
    cost = np.round(observed_units * context.unit_cost[:, np.newaxis], 2)
    gross_profit = np.round(revenue - cost, 2)

    dates = context.calendar["date"].to_numpy()[window]
    product_ids = context.listings["product_id"].to_numpy()
    store_ids = context.listings["store_id"].to_numpy()
    channels = context.listings["_channel"].to_numpy()

    date_column = np.tile(dates, n_pairs)
    product_column = np.repeat(product_ids, n_days)
    store_column = np.repeat(store_ids, n_days)

    promotion_ids = context.promotion_paths.promotion_index[:, window]
    on_promotion = promotion_ids >= 0
    # Only format labels for cells that actually carry a promotion. Building
    # them for the whole matrix and masking afterwards would do string work on
    # every row, and the vast majority are not promoted.
    promotion_labels = np.full(promotion_ids.shape, None, dtype=object)
    if on_promotion.any():
        promoted = (promotion_ids[on_promotion] + 1).astype(str)
        promotion_labels[on_promotion] = np.char.add("PR", np.char.zfill(promoted, 7))

    sales = pd.DataFrame(
        {
            "date": date_column,
            "product_id": product_column,
            "store_id": store_column,
            "channel": np.repeat(channels, n_days),
            "units": observed_units.reshape(-1),
            "regular_price": price.reshape(-1).round(2),
            "selling_price": selling_price.reshape(-1),
            "discount_percentage": (discount.reshape(-1) * 100.0).round(2),
            "revenue": revenue.reshape(-1),
            "cost": cost.reshape(-1),
            "gross_profit": gross_profit.reshape(-1),
            "promotion_id": promotion_labels.reshape(-1),
            "promotion_flag": (promotion_ids >= 0).reshape(-1),
            "inventory_available": (opening + received).reshape(-1),
            "stockout_flag": stockout.reshape(-1),
        }
    )

    inventory = pd.DataFrame(
        {
            "date": date_column,
            "product_id": product_column,
            "store_id": store_column,
            "opening_inventory": opening.reshape(-1),
            "received_units": received.reshape(-1),
            "sold_units": sold.reshape(-1),
            "closing_inventory": closing.reshape(-1),
            "stockout_flag": stockout.reshape(-1),
        }
    )
    # Days of cover at the current sales rate. Guard the divide: a day with zero
    # sales is not infinite cover, it is simply unknown.
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(sold > 0, closing / np.clip(sold, 1, None), np.nan)
    inventory["inventory_days"] = np.round(cover.reshape(-1), 2)

    pricing = pd.DataFrame(
        {
            "date": date_column,
            "product_id": product_column,
            "store_id": store_column,
            "regular_price": price.reshape(-1).round(2),
            "selling_price": selling_price.reshape(-1),
            "discount_percentage": (discount.reshape(-1) * 100.0).round(2),
            "price_change_flag": context.price_paths.change_flag[:, window].reshape(-1),
            "price_change_reason": context.price_paths.change_reason[:, window].reshape(-1),
        }
    )

    # Ground truth: the demand that existed before supply constrained it.
    latent = pd.DataFrame(
        {
            "date": date_column,
            "product_id": product_column,
            "store_id": store_column,
            "latent_units": latent_units.reshape(-1),
            "observed_units": observed_units.reshape(-1),
            "lost_units": (latent_units - observed_units).reshape(-1),
            "mean_demand": mean_demand.reshape(-1).round(4),
        }
    )

    return ChunkResult(sales=sales, inventory=inventory, pricing=pricing, latent=latent)


def _simulate_inventory(
    latent_units: np.ndarray,
    inventory_state: np.ndarray,
    supply_cap: np.ndarray,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order-up-to replenishment with censoring.

    Sequential over days because inventory is a recurrence - today's opening is
    yesterday's closing - but vectorised across all pairs within each day, so
    the loop runs once per day rather than once per row.

    Reconciliation ``opening + received - sold = closing`` holds exactly by
    construction; it is asserted in the validation suite rather than assumed.

    Stockouts arise two ways, both wanted: naturally when demand outruns the
    reorder policy (which correlates stockouts with high demand - a real source
    of selection bias), and by injected scenario via ``supply_cap``.
    """
    n_pairs, n_days = latent_units.shape

    settings = config.inventory
    cover_days = rng.uniform(*settings.target_cover_days, size=n_pairs)
    reorder_days = rng.uniform(*settings.reorder_point_days, size=n_pairs)
    lead_time = rng.integers(
        settings.lead_time_days[0], settings.lead_time_days[1] + 1, size=n_pairs
    )

    opening = np.zeros((n_pairs, n_days), dtype=np.int32)
    received = np.zeros((n_pairs, n_days), dtype=np.int32)
    sold = np.zeros((n_pairs, n_days), dtype=np.int32)
    closing = np.zeros((n_pairs, n_days), dtype=np.int32)
    stockout = np.zeros((n_pairs, n_days), dtype=bool)

    # Rolling demand estimate driving the reorder policy. Seeded from the first
    # few days so the policy is not blind on day one.
    demand_estimate = np.maximum(latent_units[:, : min(14, n_days)].mean(axis=1), 1.0)

    # In-flight replenishment orders, indexed by arrival day.
    pipeline = np.zeros((n_pairs, n_days + int(lead_time.max()) + 1), dtype=np.int32)
    # Units ordered but not yet delivered. Tracked explicitly so the reorder
    # decision can use *inventory position* (on-hand + on-order) rather than
    # on-hand alone. Ordering against on-hand re-triggers every day of the lead
    # time and stacks duplicate orders - the classic double-ordering bug, which
    # inflates stock to many times the target cover and makes stockouts
    # essentially impossible.
    outstanding = np.zeros(n_pairs, dtype=np.int64)

    stock = inventory_state.astype(np.int32).copy()

    for t in range(n_days):
        arriving = pipeline[:, t]
        # The order is closed out on its due date whether or not the goods
        # actually land. Leaving a failed delivery "in flight" would suppress
        # reordering for as long as the disruption lasts, which is the opposite
        # of how a replenishment system behaves.
        outstanding -= arriving

        # Supply failures suppress *deliveries*, not stock already on the shelf.
        # Modelling them any other way would make inventory vanish and break the
        # reconciliation identity - and it would be physically wrong: a
        # distribution failure means goods stop arriving, so the store sells down
        # what it has and only then runs dry. That delay is realistic and is what
        # makes Scenario D a gradual decline rather than a cliff.
        cap = supply_cap[:, t]
        if not np.all(cap >= 1.0):
            arriving = np.floor(arriving * cap).astype(np.int32)

        # Baseline random delivery failures, independent of the scenarios.
        if settings.random_stockout_rate > 0:
            failed = rng.random(n_pairs) < settings.random_stockout_rate
            arriving = np.where(failed, (arriving * 0.2).astype(np.int32), arriving)

        opening[:, t] = stock
        received[:, t] = arriving
        available = stock + arriving

        demand = latent_units[:, t]
        units_sold = np.minimum(demand, available)
        sold[:, t] = units_sold
        stockout[:, t] = demand > available

        stock = available - units_sold
        closing[:, t] = stock

        # Exponentially weighted demand estimate for the reorder policy.
        demand_estimate = 0.9 * demand_estimate + 0.1 * np.maximum(demand, 0)

        reorder_point = demand_estimate * reorder_days
        target_level = demand_estimate * cover_days
        # Order-up-to against inventory position, not on-hand.
        position = stock + outstanding
        needs_order = position < reorder_point
        order_quantity = np.where(needs_order, np.maximum(target_level - position, 0), 0)
        arrival = np.asarray(lead_time + t, dtype=int)
        valid = arrival < pipeline.shape[1]
        rows = np.flatnonzero(needs_order & valid)
        if rows.size:
            quantities = order_quantity[rows].astype(np.int32)
            pipeline[rows, arrival[rows]] += quantities
            outstanding[rows] += quantities

    inventory_state[:] = stock
    return opening, received, sold, closing, stockout


def initial_inventory(
    context_base_demand: np.ndarray,
    config: GenerationConfig,
    rngs: RngFactory,
) -> np.ndarray:
    """Opening stock on day one: a configurable number of days of cover."""
    rng = rngs.get(Stream.INVENTORY)
    cover = config.inventory.initial_cover_days
    jitter = rng.uniform(0.7, 1.3, size=len(context_base_demand))
    return np.maximum((context_base_demand * cover * jitter).astype(np.int32), 1)
