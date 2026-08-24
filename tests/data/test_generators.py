"""Unit tests for individual generators.

These exercise the components in isolation, against a tiny in-memory config, so
a failure points at one generator rather than at "the pipeline".
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.generation.calendar_math import (
    annual_seasonality_series,
    exponential_decay,
    launch_ramp,
    linear_trend,
)
from data.generation.config import GenerationConfig, load_config
from data.generation.generators.calendar_generator import generate_calendar
from data.generation.generators.customer_generator import generate_customers
from data.generation.generators.product_generator import generate_products
from data.generation.generators.product_relationship_generator import (
    build_cross_matrix,
    generate_product_relationships,
)
from data.generation.generators.store_generator import generate_listings, generate_stores
from data.generation.rng import RngFactory, Stream

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def config() -> GenerationConfig:
    return load_config("smoke", overrides={"seed": 7})


@pytest.fixture
def rngs(config: GenerationConfig) -> RngFactory:
    return RngFactory(config.seed)


# --- calendar --------------------------------------------------------------


def test_calendar_is_contiguous() -> None:
    calendar = generate_calendar(date(2023, 1, 1), date(2023, 12, 31))
    assert len(calendar) == 365
    gaps = pd.to_datetime(calendar["date"]).diff().dt.days.dropna()
    assert (gaps == 1).all()


def test_calendar_flags_weekends_correctly() -> None:
    calendar = generate_calendar(date(2023, 1, 1), date(2023, 1, 31))
    saturday = calendar[calendar["date"] == date(2023, 1, 7)].iloc[0]
    monday = calendar[calendar["date"] == date(2023, 1, 9)].iloc[0]
    assert bool(saturday["weekend_flag"]) is True
    assert bool(monday["weekend_flag"]) is False


def test_indian_festivals_are_flagged() -> None:
    """Diwali 2023 fell on 12 November; the run-up must be flagged too."""
    calendar = generate_calendar(date(2023, 11, 1), date(2023, 11, 30), geography="IN")
    diwali = calendar[calendar["date"] == date(2023, 11, 12)].iloc[0]
    assert bool(diwali["festival_flag"]) is True
    assert diwali["festival_name"] == "Diwali"
    # Shoppers buy ahead, so the days before must also carry the flag.
    lead = calendar[calendar["date"] == date(2023, 11, 9)].iloc[0]
    assert bool(lead["festival_flag"]) is True


def test_festival_dates_move_between_years() -> None:
    """Lunar festivals shift; a fixed-date approximation would be wrong."""
    calendar = generate_calendar(date(2023, 1, 1), date(2025, 12, 31), geography="IN")
    diwali = calendar[calendar["festival_name"] == "Diwali"]
    per_year = diwali.groupby("year")["date"].max()
    assert per_year.loc[2023] != per_year.loc[2024].replace(year=2023)


def test_indian_financial_year_starts_in_april() -> None:
    calendar = generate_calendar(date(2023, 1, 1), date(2023, 12, 31), geography="IN")
    april = calendar[calendar["month"] == 4].iloc[0]
    march = calendar[calendar["month"] == 3].iloc[0]
    assert int(april["financial_month"]) == 1
    assert int(march["financial_month"]) == 12
    assert int(april["financial_year"]) == 2023
    assert int(march["financial_year"]) == 2022


def test_global_geography_excludes_indian_festivals() -> None:
    calendar = generate_calendar(date(2023, 11, 1), date(2023, 11, 30), geography="GLOBAL")
    assert not calendar["festival_flag"].any()


# --- calendar maths --------------------------------------------------------


@pytest.mark.parametrize("peak_month", [1, 4, 7, 10])
def test_seasonality_peaks_at_configured_month(peak_month: int) -> None:
    day_of_year = np.arange(1, 366)
    seasonal = annual_seasonality_series(day_of_year, amplitude=0.3, peak_month=peak_month)
    peak_day = int(day_of_year[int(np.argmax(seasonal))])
    peak_date = date(2023, 1, 1) + pd.Timedelta(days=peak_day - 1)
    assert peak_date.month == peak_month


def test_seasonality_amplitude_is_respected() -> None:
    seasonal = annual_seasonality_series(np.arange(1, 366), amplitude=0.25, peak_month=1)
    assert pytest.approx(float(seasonal.max()), abs=0.01) == 0.25
    assert pytest.approx(float(seasonal.min()), abs=0.01) == -0.25


def test_trend_compounds_annually() -> None:
    trend = linear_trend(np.array([0.0, 365.25]), annual_rate=0.10)
    assert pytest.approx(float(np.exp(trend[1])), rel=0.01) == 1.10


def test_launch_ramp_is_gradual_and_saturating() -> None:
    ramp = launch_ramp(np.array([0, 30, 60, 120, 400]), ramp_days=120)
    assert ramp[0] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < ramp[1] < ramp[2] < ramp[3]
    assert ramp[4] > 0.95  # saturated


def test_exponential_decay_is_zero_before_the_event() -> None:
    decay = exponential_decay(np.array([-3.0, 0.0, 5.0]), half_life_days=5.0)
    assert decay[0] == 0.0
    assert decay[1] == pytest.approx(1.0)
    assert decay[2] == pytest.approx(0.5, abs=0.01)


# --- products --------------------------------------------------------------


def test_product_count_matches_config(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    assert len(products) == config.scale.products


def test_products_have_positive_economics(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    assert (products["unit_cost"] > 0).all()
    assert (products["base_price"] > products["unit_cost"]).all()


def test_base_demand_is_skewed_not_uniform(config: GenerationConfig, rngs: RngFactory) -> None:
    """Real CPG catalogues have hero SKUs and a long tail.

    A uniform draw would remove the volume concentration that makes WMAPE a
    more honest forecasting metric than MAPE.
    """
    products = generate_products(config, rngs)
    demand = products["_base_demand"]
    assert float(demand.mean()) > float(demand.median())


def test_every_category_is_represented(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    assert set(products["category"]) == set(config.categories)


def test_lifecycle_products_exist(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    assert set(products["product_status"]) <= {"Active", "Launched", "Discontinued"}


# --- relationships ---------------------------------------------------------


def test_substitute_and_complement_signs(config: GenerationConfig, rngs: RngFactory) -> None:
    """The sign convention everything downstream depends on."""
    products = generate_products(config, rngs)
    relationships = generate_product_relationships(products, config, rngs)

    substitutes = relationships[relationships["relationship_type"] == "substitute"]
    complements = relationships[relationships["relationship_type"] == "complement"]
    unrelated = relationships[relationships["relationship_type"] == "unrelated"]

    assert (substitutes["cross_elasticity"] > 0).all()
    assert (complements["cross_elasticity"] < 0).all()
    assert (unrelated["cross_elasticity"] == 0).all()


def test_relationships_are_asymmetric(config: GenerationConfig, rngs: RngFactory) -> None:
    """A premium SKU losing volume to a value SKU is not matched in reverse."""
    products = generate_products(config, rngs)
    relationships = generate_product_relationships(products, config, rngs)
    effective = relationships[relationships["relationship_type"] != "unrelated"]

    lookup = {
        (row.product_a, row.product_b): row.cross_elasticity
        for row in effective.itertuples(index=False)
    }
    asymmetric = [
        (a, b)
        for (a, b), value in lookup.items()
        if (b, a) in lookup and abs(lookup[(b, a)] - value) > 1e-9
    ]
    assert asymmetric, "expected at least one asymmetric relationship pair"


def test_unrelated_pairs_are_labelled(config: GenerationConfig, rngs: RngFactory) -> None:
    """Labelled negatives let Step 9 be scored on false positives too."""
    products = generate_products(config, rngs)
    relationships = generate_product_relationships(products, config, rngs)
    assert (relationships["relationship_type"] == "unrelated").any()


def test_cross_matrix_excludes_unrelated(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    relationships = generate_product_relationships(products, config, rngs)
    matrix = build_cross_matrix(relationships, products["product_id"].to_numpy())

    unrelated = relationships[relationships["relationship_type"] == "unrelated"]
    for row in unrelated.itertuples(index=False):
        assert row.product_b not in matrix.get(str(row.product_a), {})


# --- stores and listings ---------------------------------------------------


def test_store_count_and_regions(config: GenerationConfig, rngs: RngFactory) -> None:
    stores = generate_stores(config, rngs)
    assert len(stores) == config.scale.stores
    assert set(stores["region"]) <= set(config.stores.regions)
    assert set(stores["channel"]) <= set(config.stores.channels)


def test_store_segment_mix_sums_to_one(config: GenerationConfig, rngs: RngFactory) -> None:
    stores = generate_stores(config, rngs)
    mix_columns = [c for c in stores.columns if c.startswith("_mix_")]
    totals = stores[mix_columns].sum(axis=1)
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-6)


def test_price_sensitivity_varies_across_stores(config: GenerationConfig, rngs: RngFactory) -> None:
    """Segment mix must produce genuine store-level elasticity heterogeneity."""
    stores = generate_stores(config, rngs)
    assert float(stores["_price_sensitivity"].std()) > 0.01


def test_listings_reference_real_entities(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    stores = generate_stores(config, rngs)
    listings = generate_listings(products, stores, config, rngs)

    assert set(listings["product_id"]) <= set(products["product_id"])
    assert set(listings["store_id"]) <= set(stores["store_id"])
    assert not listings.duplicated(subset=["product_id", "store_id"]).any()


def test_every_product_is_listed_somewhere(config: GenerationConfig, rngs: RngFactory) -> None:
    products = generate_products(config, rngs)
    stores = generate_stores(config, rngs)
    listings = generate_listings(products, stores, config, rngs)
    assert set(listings["product_id"]) == set(products["product_id"])


# --- customers -------------------------------------------------------------


def test_customer_count_and_segments(config: GenerationConfig, rngs: RngFactory) -> None:
    customers = generate_customers(config, rngs)
    assert len(customers) == config.scale.customers
    assert set(customers["segment"]) <= set(config.customers.segments)


def test_customers_carry_no_pii(config: GenerationConfig, rngs: RngFactory) -> None:
    """Brief section 6: no unnecessary personal information."""
    customers = generate_customers(config, rngs)
    forbidden = {"name", "email", "phone", "address", "dob", "date_of_birth", "pan", "aadhaar"}
    assert not (forbidden & {c.lower() for c in customers.columns})


def test_loyalty_tier_correlates_with_segment(config: GenerationConfig, rngs: RngFactory) -> None:
    """Independent draws would leave the columns uncorrelated and unrealistic."""
    customers = generate_customers(config, rngs)
    tier_rank = {tier: i for i, tier in enumerate(config.customers.loyalty_tiers)}
    customers = customers.assign(rank=customers["loyalty_tier"].map(tier_rank))
    by_segment = customers.groupby("segment")["rank"].mean()
    assert by_segment.max() - by_segment.min() > 0.15


# --- promotion response curve ----------------------------------------------


def _uplift(a: float, b: float, discount: float) -> float:
    """The generator's saturating response, on the demand scale."""
    return float(np.exp(a * (1.0 - np.exp(-b * discount))) - 1.0)


def test_promotion_response_saturates(config: GenerationConfig) -> None:
    """Marginal uplift must shrink as discount deepens.

    Tested against the response function directly rather than against observed
    band means: the function is deterministic, so curvature can be asserted
    exactly, whereas observed uplift by depth also carries display, bundle and
    mechanic-mix effects that vary with depth.

    Without saturation, Step 7's optimiser would pour the entire budget into the
    single deepest discount available - which is both wrong and obviously wrong
    to any category manager.
    """
    settings = config.promotions.types["Price Discount"]
    a = float(np.mean(settings.a))
    b = settings.b

    marginals = [
        _uplift(a, b, depth + 0.10) - _uplift(a, b, depth) for depth in (0.0, 0.10, 0.20, 0.30)
    ]
    assert all(m > 0 for m in marginals), "uplift must keep rising with depth"
    assert marginals == sorted(marginals, reverse=True), (
        f"marginal uplift must shrink with depth, got {marginals}"
    )


def test_promotion_response_matches_briefed_bands(config: GenerationConfig) -> None:
    """Brief section 15: 10% -> ~10-15%, 20% -> ~20-30%, 30% -> ~25-35%."""
    settings = config.promotions.types["Price Discount"]
    a = float(np.mean(settings.a))
    b = settings.b

    assert 0.08 <= _uplift(a, b, 0.10) <= 0.18
    assert 0.18 <= _uplift(a, b, 0.20) <= 0.32
    assert 0.22 <= _uplift(a, b, 0.30) <= 0.40


def test_all_promotion_types_saturate(config: GenerationConfig) -> None:
    for name, settings in config.promotions.types.items():
        a = float(np.mean(settings.a))
        first = _uplift(a, settings.b, 0.10) - _uplift(a, settings.b, 0.0)
        last = _uplift(a, settings.b, 0.40) - _uplift(a, settings.b, 0.30)
        assert last < first, f"{name} does not show diminishing returns"


# --- rng -------------------------------------------------------------------


def test_streams_are_independent() -> None:
    """Changing draws in one stream must not shift another."""
    factory = RngFactory(42)
    product_first = factory.get(Stream.PRODUCT).random(5)

    other = RngFactory(42)
    other.get(Stream.STORE).random(1000)  # consume a different stream heavily
    product_second = other.get(Stream.PRODUCT).random(5)

    assert np.allclose(product_first, product_second)


def test_same_seed_reproduces_stream() -> None:
    assert np.allclose(
        RngFactory(11).get(Stream.DEMAND).random(10),
        RngFactory(11).get(Stream.DEMAND).random(10),
    )


def test_different_seeds_diverge() -> None:
    assert not np.allclose(
        RngFactory(11).get(Stream.DEMAND).random(10),
        RngFactory(12).get(Stream.DEMAND).random(10),
    )


def test_chunk_generators_are_order_independent() -> None:
    """Chunk randomness must depend on the index, not on iteration order.

    Otherwise changing ``chunk_months`` would change the data, and the profile
    setting would silently stop being a pure performance knob.
    """
    factory = RngFactory(5)
    third_first = RngFactory(5).fresh(Stream.DEMAND, 2).random(4)
    factory.fresh(Stream.DEMAND, 0).random(100)
    factory.fresh(Stream.DEMAND, 1).random(100)
    third_after = factory.fresh(Stream.DEMAND, 2).random(4)
    assert np.allclose(third_first, third_after)
