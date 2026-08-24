"""Dataset generation configuration.

Values live in ``configs/data/*.yaml``; shape and bounds are enforced here.

Why YAML validated by Pydantic, rather than pydantic-settings alone (brief
section 26): these are ~100 nested *business* parameters - per-category
seasonality, elasticity bands, promotion response constants. Three things follow
from that:

* Environment variables cannot express nested structure usably. ``CATEGORIES__
  BEVERAGES__ELASTICITY_RANGE`` is not a serious interface.
* A change to elasticity bands must be visible in review. A YAML diff shows it;
  an env var buried in a deployment does not.
* Validation must fail fast with a field path, not a ``KeyError`` thrown a
  thousand rows into generation.

Step 1's ``app.config.settings`` remains the home of *environment* config -
paths, secrets, budgets. Deliberately not merged: dataset shape and deployment
environment change for different reasons and on different cadences.

Profiles compose via ``extends:``, so ``smoke`` and ``stress`` state only their
differences from ``dev`` and cannot silently drift from it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import PROJECT_ROOT

CONFIG_ROOT = PROJECT_ROOT / "configs" / "data"

#: Inclusive [low, high] band that generators sample from.
Range = tuple[float, float]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_range(value: Range, name: str) -> Range:
    if value[0] > value[1]:
        raise ValueError(f"{name}: low ({value[0]}) must not exceed high ({value[1]})")
    return value


class ScaleConfig(_Base):
    products: int = Field(gt=0)
    stores: int = Field(gt=0)
    customers: int = Field(gt=0)
    competitors: int = Field(gt=0)
    stores_per_product_mean: float = Field(gt=0)
    stores_per_product_std: float = Field(ge=0)
    transaction_sample_rate: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _listings_fit(self) -> Self:
        if self.stores_per_product_mean > self.stores:
            raise ValueError(
                f"stores_per_product_mean ({self.stores_per_product_mean}) exceeds "
                f"stores ({self.stores}); a product cannot be listed in more stores "
                f"than exist"
            )
        return self

    @property
    def expected_pairs(self) -> int:
        """Approximate product-store listings; drives total row count."""
        return int(self.products * self.stores_per_product_mean)


class TimeConfig(_Base):
    start_date: date
    end_date: date
    holdout_start_date: date | None = None
    holdout_end_date: date | None = None
    generate_holdout: bool = False
    #: Calendar geography. "IN" adds Diwali, Holi, Eid; "US"/"GLOBAL" do not.
    geography: str = "IN"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must precede end_date")
        if self.generate_holdout:
            if self.holdout_start_date is None or self.holdout_end_date is None:
                raise ValueError("generate_holdout requires holdout_start_date/end_date")
            if self.holdout_start_date <= self.end_date:
                raise ValueError("holdout_start_date must fall after end_date")
        return self

    @property
    def n_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


class CategoryConfig(_Base):
    weight: float = Field(gt=0)
    subcategories: list[str] = Field(min_length=1)
    base_demand: Range
    unit_cost: Range
    margin: Range
    #: Own-price elasticity band. Both ends must be negative for a normal good.
    elasticity_range: Range
    seasonal_amplitude: float = Field(ge=0, le=1)
    seasonal_peak_month: int = Field(ge=1, le=12)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        for name in ("base_demand", "unit_cost", "margin", "elasticity_range"):
            _check_range(getattr(self, name), name)
        if self.elasticity_range[1] >= 0:
            raise ValueError(
                f"elasticity_range must be entirely negative for a normal good, "
                f"got {self.elasticity_range}"
            )
        return self


class LifecycleConfig(_Base):
    launched_mid_history_pct: float = Field(ge=0, le=1)
    discontinued_pct: float = Field(ge=0, le=1)
    launch_ramp_days: int = Field(gt=0)


class RelationshipStrength(_Base):
    strong_substitute: Range
    weak_substitute: Range
    strong_complement: Range
    weak_complement: Range

    @model_validator(mode="after")
    def _signs(self) -> Self:
        # Positive cross-elasticity => substitutes; negative => complements.
        # Getting these backwards inverts every downstream conclusion, so it is
        # checked here rather than discovered in Step 9.
        if self.strong_substitute[0] <= 0 or self.weak_substitute[0] <= 0:
            raise ValueError("substitute cross-elasticities must be positive")
        if self.strong_complement[1] >= 0 or self.weak_complement[1] >= 0:
            raise ValueError("complement cross-elasticities must be negative")
        return self


class RelationshipConfig(_Base):
    strong_substitute_pairs_per_category: int = Field(ge=0)
    weak_substitute_pairs_per_category: int = Field(ge=0)
    complement_pairs_per_category: int = Field(ge=0)
    cross_category_complement_pairs: int = Field(ge=0)
    strength: RelationshipStrength


class ChannelConfig(_Base):
    weight: float = Field(gt=0)
    size_sqft: tuple[int, int]
    demand_scale: Range


class RegionConfig(_Base):
    weight: float = Field(gt=0)
    demand_multiplier: float = Field(gt=0)
    states: list[str] = Field(min_length=1)


class StoreConfig(_Base):
    channels: dict[str, ChannelConfig]
    regions: dict[str, RegionConfig]
    opened_mid_history_pct: float = Field(ge=0, le=1)


class SegmentConfig(_Base):
    weight: float = Field(gt=0)
    #: Multiplies the product's own-price elasticity for this segment.
    price_sensitivity: float = Field(gt=0)
    promo_responsiveness: float = Field(gt=0)


class CustomerConfig(_Base):
    segments: dict[str, SegmentConfig]
    loyalty_tiers: list[str] = Field(min_length=1)
    loyalty_weights: list[float] = Field(min_length=1)
    acquisition_channels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _tiers_match(self) -> Self:
        if len(self.loyalty_tiers) != len(self.loyalty_weights):
            raise ValueError("loyalty_tiers and loyalty_weights must be the same length")
        return self


class DemandConfig(_Base):
    trend_annual: Range
    day_of_week: list[float] = Field(min_length=7, max_length=7)
    holiday_multiplier: Range
    festival_multiplier: Range
    negbinom_dispersion: Range
    noise_sigma: float = Field(gt=0)
    anomaly_rate: float = Field(ge=0, le=1)
    anomaly_magnitude: Range


class PricingConfig(_Base):
    price_changes_per_year: tuple[int, int]
    price_change_magnitude: Range
    regional_price_spread: float = Field(ge=0, lt=1)
    #: Confounder 1: price responds to anticipated demand. Biases naive OLS
    #: toward zero; the cure is fixed effects or the cost instrument.
    endogeneity_strength: float = Field(ge=0, le=1)
    #: Confounder 3 (the antidote): randomised price tests give exogenous variation.
    randomised_test_fraction: float = Field(ge=0, le=1)
    #: Confounder 2 (the instrument): commodity cost shifts price, not demand.
    cost_index_volatility: float = Field(ge=0)
    cost_passthrough: float = Field(ge=0, le=1)


class CompetitorConfig(_Base):
    price_index_vs_ours: Range
    price_changes_per_year: tuple[int, int]
    promotion_frequency: float = Field(ge=0, le=1)
    #: gamma: competitor price up => our demand up. Must be positive.
    cross_sensitivity: Range
    cost_index_correlation: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _positive_gamma(self) -> Self:
        if self.cross_sensitivity[0] <= 0:
            raise ValueError("competitor cross_sensitivity must be positive")
        return self


class PromotionTypeConfig(_Base):
    weight: float = Field(gt=0)
    #: Saturation ceiling of the response curve, in log space.
    a: Range
    #: Curvature. Higher => saturates sooner.
    b: float = Field(gt=0)
    spend_per_unit: float = Field(ge=0)
    display: float = Field(ge=0, le=1)
    bundle: float = Field(ge=0, le=1)


class PromotionConfig(_Base):
    events_per_product_per_year: tuple[int, int]
    duration_days: tuple[int, int]
    discount_depth: Range
    types: dict[str, PromotionTypeConfig]
    display_lift: float = Field(ge=0)
    bundle_lift: float = Field(ge=0)
    #: Pantry loading. Why naive during-vs-before uplift overstates incrementality.
    pull_forward_fraction: float = Field(ge=0, le=1)
    pull_forward_decay_days: int = Field(gt=0)
    #: Confounder 4: promos target already-softening products.
    targeting_strength: float = Field(ge=0, le=1)
    fixed_spend: Range


class TradePromotionConfig(_Base):
    annual_budget: float = Field(gt=0)
    retailers: list[str] = Field(min_length=1)
    events_per_retailer_per_year: tuple[int, int]
    planned_vs_actual_spend_variance: float = Field(ge=0)
    expected_vs_actual_uplift_variance: float = Field(ge=0)


class InventoryConfig(_Base):
    target_cover_days: tuple[int, int]
    reorder_point_days: tuple[int, int]
    lead_time_days: tuple[int, int]
    random_stockout_rate: float = Field(ge=0, le=1)
    initial_cover_days: int = Field(gt=0)

    @model_validator(mode="after")
    def _reorder_below_target(self) -> Self:
        if self.reorder_point_days[1] >= self.target_cover_days[0]:
            raise ValueError(
                "reorder_point_days must stay below target_cover_days, otherwise "
                "the replenishment policy reorders continuously"
            )
        return self


class ScenarioSpec(_Base):
    products: int = Field(default=0, ge=0)
    magnitude: float = 0.0
    discount: float = 0.0
    duration_days: int = Field(default=30, gt=0)
    stores_fraction: float = Field(default=1.0, gt=0, le=1)
    region: str | None = None


class ScenarioConfig(_Base):
    enabled: bool = True
    price_increase: ScenarioSpec
    successful_promo: ScenarioSpec
    bad_promo: ScenarioSpec
    stockout: ScenarioSpec
    competitor_price_cut: ScenarioSpec
    regional_shock: ScenarioSpec


class DataQualityConfig(_Base):
    """Corruption rates. Applied to the bronze layer only - gold stays clean."""

    enabled: bool = True
    missing_product_id: float = Field(ge=0, le=1)
    missing_store_id: float = Field(ge=0, le=1)
    duplicate_transactions: float = Field(ge=0, le=1)
    invalid_price: float = Field(ge=0, le=1)
    negative_quantity: float = Field(ge=0, le=1)
    invalid_discount: float = Field(ge=0, le=1)
    orphan_promotion_id: float = Field(ge=0, le=1)
    malformed_date: float = Field(ge=0, le=1)
    duplicate_promotions: float = Field(ge=0, le=1)

    def rates(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if k != "enabled"}


class OutputConfig(_Base):
    chunk_months: int = Field(gt=0)
    write_bronze: bool = True
    write_samples: bool = True
    sample_rows: int = Field(default=1000, gt=0)
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd"


class GenerationConfig(_Base):
    """Fully validated dataset profile."""

    dataset_version: str
    scenario_version: str
    seed: int

    scale: ScaleConfig
    time: TimeConfig
    categories: dict[str, CategoryConfig]
    brands_per_category: tuple[int, int]
    pack_sizes: list[str] = Field(min_length=1)
    lifecycle: LifecycleConfig
    relationships: RelationshipConfig
    stores: StoreConfig
    customers: CustomerConfig
    demand: DemandConfig
    pricing: PricingConfig
    competitor: CompetitorConfig
    promotions: PromotionConfig
    trade_promotions: TradePromotionConfig
    inventory: InventoryConfig
    scenarios: ScenarioConfig
    data_quality: DataQualityConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _enough_products_for_relationships(self) -> Self:
        # With few products per category there may be too few to form the
        # requested pairs. Generators clamp, but a wildly impossible ask is a
        # config error worth surfacing now.
        r = self.relationships
        per_category = self.scale.products / max(len(self.categories), 1)
        needed = (
            r.strong_substitute_pairs_per_category
            + r.weak_substitute_pairs_per_category
            + r.complement_pairs_per_category
        )
        if needed > 0 and per_category < 2:
            raise ValueError(
                f"only ~{per_category:.1f} products per category; at least 2 are "
                f"required to form relationship pairs"
            )
        return self

    def config_hash(self) -> str:
        """Stable hash of the resolved config, recorded in the manifest.

        Lets a dataset be tied to the exact parameters that produced it, so a
        surprising validation result can be traced to a config change rather
        than guessed at.
        """
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Nested dicts merge key-by-key; scalars and lists replace wholesale. That
    asymmetry is intended: a profile overriding ``scale.products`` should not
    have to restate the rest of ``scale``, but overriding ``day_of_week`` should
    replace all seven values rather than splice them.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_raw(profile: str, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    if profile in _seen:
        raise ValueError(f"circular 'extends' chain involving profile {profile!r}")

    path = CONFIG_ROOT / f"{profile}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_ROOT.glob("*.yaml"))
        raise FileNotFoundError(f"No data profile {profile!r} at {path}. Available: {available}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_raw(str(parent), _seen | {profile}), raw)


def load_config(
    profile: str = "dev",
    *,
    overrides: dict[str, Any] | None = None,
) -> GenerationConfig:
    """Load and validate a profile, applying optional CLI overrides.

    ``overrides`` uses dotted paths, e.g. ``{"scale.products": 500, "seed": 7}``,
    which is what the CLI flags in brief section 21 map onto.
    """
    raw = _load_raw(profile)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = raw
        *parents, leaf = dotted.split(".")
        for part in parents:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[leaf] = value

    return GenerationConfig.model_validate(raw)


def available_profiles() -> list[str]:
    return sorted(p.stem for p in CONFIG_ROOT.glob("*.yaml"))
