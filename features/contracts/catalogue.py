"""The feature catalogue and per-model requirements (brief sections 22, 24).

Every feature the platform produces is declared here with its temporality. That
declaration is not documentation - the leakage tests read it, so a feature that
starts reaching forward without its spec being updated fails the suite.

The catalogue is the source of truth; ``configs/features/features.yaml`` selects
*which* of these a given feature set uses. Definitions in code, selection in
config: a definition is behaviour and belongs under review, whereas a selection
is a knob.
"""

from __future__ import annotations

from features.contracts.specs import (
    FeatureGroup,
    FeatureGroupName,
    FeatureSpec,
    ModelFeatureRequirement,
    Temporality,
)

_B = Temporality.BACKWARD
_C = Temporality.CONTEMPORANEOUS
_F = Temporality.FORWARD_PLANNED


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------

DEMAND_GROUP = FeatureGroup(
    name=FeatureGroupName.DEMAND,
    description="Historical demand: lags, rolling statistics and dynamics.",
    features=(
        *(
            FeatureSpec(
                name=f"lag_{n}_units",
                group=FeatureGroupName.DEMAND,
                description=f"Units sold {n} day(s) earlier for this product-store.",
                source_columns=("units",),
                source_tables=("sales_daily",),
                transformation=f"groupby(product,store).shift({n})",
                temporality=_B,
            )
            for n in (1, 7, 14, 28, 56, 364)
        ),
        *(
            FeatureSpec(
                name=f"rolling_{w}_units",
                group=FeatureGroupName.DEMAND,
                description=(
                    f"Mean units over the {w} days ending yesterday. Excludes the "
                    f"current day by construction."
                ),
                source_columns=("units",),
                source_tables=("sales_daily",),
                transformation=f"shift(1).rolling({w}).mean()",
                temporality=_B,
            )
            for w in (7, 14, 28, 56)
        ),
        *(
            FeatureSpec(
                name=f"rolling_{w}_units_std",
                group=FeatureGroupName.DEMAND,
                description=f"Volatility of units over the {w} days ending yesterday.",
                source_columns=("units",),
                source_tables=("sales_daily",),
                transformation=f"shift(1).rolling({w}).std()",
                temporality=_B,
            )
            for w in (7, 14, 28, 56)
        ),
        FeatureSpec(
            name="demand_momentum",
            group=FeatureGroupName.DEMAND,
            description="7-day mean over 28-day mean. Above 1 means accelerating.",
            source_columns=("units",),
            source_tables=("sales_daily",),
            transformation="rolling_7 / rolling_28, both shifted",
            temporality=_B,
        ),
        FeatureSpec(
            name="demand_volatility",
            group=FeatureGroupName.DEMAND,
            description="Coefficient of variation over 28 days; widens forecast intervals.",
            source_columns=("units",),
            source_tables=("sales_daily",),
            transformation="rolling_28_std / rolling_28_mean, both shifted",
            temporality=_B,
        ),
        FeatureSpec(
            name="demand_trend_28",
            group=FeatureGroupName.DEMAND,
            description="Growth of the recent week against four weeks ago.",
            source_columns=("units",),
            source_tables=("sales_daily",),
            transformation="(rolling_7 - lag_28) / lag_28",
            temporality=_B,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

PRICE_GROUP = FeatureGroup(
    name=FeatureGroupName.PRICE,
    description="Own price level, movement and competitive position.",
    features=(
        FeatureSpec(
            name="selling_price",
            group=FeatureGroupName.PRICE,
            description="Price actually paid after any promotional discount.",
            source_columns=("selling_price",),
            source_tables=("sales_daily", "pricing"),
            transformation="passthrough",
            # Today's price is set by the business, so it is known at prediction
            # time. This is why `pricing` is KNOWN_IN_ADVANCE.
            temporality=_C,
        ),
        FeatureSpec(
            name="regular_price",
            group=FeatureGroupName.PRICE,
            description="Undiscounted shelf price.",
            source_columns=("regular_price",),
            source_tables=("sales_daily", "pricing"),
            transformation="passthrough",
            temporality=_C,
        ),
        FeatureSpec(
            name="discount_depth",
            group=FeatureGroupName.PRICE,
            description="1 - selling/regular. Recomputed rather than trusted.",
            source_columns=("selling_price", "regular_price"),
            source_tables=("pricing",),
            transformation="1 - selling_price / regular_price",
            temporality=_C,
        ),
        FeatureSpec(
            name="price_index",
            group=FeatureGroupName.PRICE,
            description=(
                "Own price over the same-day category mean. Above 1 means priced "
                "above the competitive set on the shelf."
            ),
            source_columns=("selling_price", "category"),
            source_tables=("sales_daily", "products"),
            transformation="selling_price / mean(selling_price) by (date, category)",
            temporality=_C,
        ),
        FeatureSpec(
            name="price_change_pct_1",
            group=FeatureGroupName.PRICE,
            description="Price change against yesterday.",
            source_columns=("selling_price",),
            source_tables=("pricing",),
            transformation="(price - shift(1)) / shift(1)",
            temporality=_C,
        ),
        FeatureSpec(
            name="price_change_pct_7",
            group=FeatureGroupName.PRICE,
            description="Price change against a week ago.",
            source_columns=("selling_price",),
            source_tables=("pricing",),
            transformation="(price - shift(7)) / shift(7)",
            temporality=_C,
        ),
        FeatureSpec(
            name="price_vs_rolling_average",
            group=FeatureGroupName.PRICE,
            description="Today's price over this item's own trailing 28-day mean.",
            source_columns=("selling_price",),
            source_tables=("pricing",),
            transformation="price / shift(1).rolling(28).mean()",
            temporality=_C,
        ),
        FeatureSpec(
            name="historical_average_price",
            group=FeatureGroupName.PRICE,
            description="Expanding mean of price over all prior days.",
            source_columns=("selling_price",),
            source_tables=("pricing",),
            transformation="shift(1).expanding().mean()",
            temporality=_B,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Competitor
# ---------------------------------------------------------------------------

COMPETITOR_GROUP = FeatureGroup(
    name=FeatureGroupName.COMPETITOR,
    description="Rival pricing. Observed, so never available beyond the as-of date.",
    features=(
        FeatureSpec(
            name="competitor_price",
            group=FeatureGroupName.COMPETITOR,
            description="Mean effective competitor price for this product on this date.",
            source_columns=("competitor_effective_price",),
            source_tables=("competitor_pricing",),
            transformation="mean by (date, product)",
            temporality=_B,
        ),
        FeatureSpec(
            name="price_gap",
            group=FeatureGroupName.COMPETITOR,
            description="own_price - competitor_price, in currency.",
            source_columns=("selling_price", "competitor_effective_price"),
            source_tables=("sales_daily", "competitor_pricing"),
            transformation="own - competitor",
            temporality=_B,
        ),
        FeatureSpec(
            name="price_ratio",
            group=FeatureGroupName.COMPETITOR,
            description="own_price / competitor_price. Scale-free; linear in a log model.",
            source_columns=("selling_price", "competitor_effective_price"),
            source_tables=("sales_daily", "competitor_pricing"),
            transformation="own / competitor",
            temporality=_B,
        ),
        FeatureSpec(
            name="competitor_price_index",
            group=FeatureGroupName.COMPETITOR,
            description="competitor_price / own_price.",
            source_columns=("selling_price", "competitor_effective_price"),
            source_tables=("competitor_pricing",),
            transformation="competitor / own",
            temporality=_B,
        ),
        FeatureSpec(
            name="competitor_discount",
            group=FeatureGroupName.COMPETITOR,
            description="Mean competitor discount depth.",
            source_columns=("competitor_discount",),
            source_tables=("competitor_pricing",),
            transformation="mean by (date, product)",
            temporality=_B,
        ),
        FeatureSpec(
            name="competitor_promotion_flag",
            group=FeatureGroupName.COMPETITOR,
            description="Any competitor promoting this product on this date.",
            source_columns=("competitor_promotion_flag",),
            source_tables=("competitor_pricing",),
            transformation="max by (date, product)",
            temporality=_B,
            dtype="bool",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

PROMOTION_GROUP = FeatureGroup(
    name=FeatureGroupName.PROMOTION,
    description=(
        "Promotion schedule and history. Schedule features look forward "
        "legitimately; realised spend does not."
    ),
    features=(
        FeatureSpec(
            name="promotion_flag",
            group=FeatureGroupName.PROMOTION,
            description="A promotion is active on this product-store-date.",
            source_columns=("promotion_id",),
            source_tables=("promotions",),
            transformation="expanded promotion calendar, joined",
            temporality=_C,
            dtype="bool",
        ),
        FeatureSpec(
            name="promotion_discount",
            group=FeatureGroupName.PROMOTION,
            description="Depth of the active promotion, as a fraction.",
            source_columns=("discount_percentage",),
            source_tables=("promotions",),
            transformation="discount_percentage / 100",
            temporality=_C,
        ),
        FeatureSpec(
            name="promotion_duration",
            group=FeatureGroupName.PROMOTION,
            description="Total length of the active promotion in days.",
            source_columns=("duration_days",),
            source_tables=("promotions",),
            transformation="passthrough from the event",
            temporality=_C,
        ),
        FeatureSpec(
            name="days_into_promotion",
            group=FeatureGroupName.PROMOTION,
            description="Days elapsed since this promotion started.",
            source_columns=("start_date",),
            source_tables=("promotions",),
            transformation="date - start_date",
            temporality=_C,
        ),
        FeatureSpec(
            name="days_until_promotion_end",
            group=FeatureGroupName.PROMOTION,
            description="Days remaining in the active promotion.",
            source_columns=("end_date",),
            source_tables=("promotions",),
            transformation="end_date - date",
            temporality=_F,
            forward_justification=(
                "Promotion mechanics are agreed with retailers weeks ahead, so a "
                "planner on date D genuinely knows the end date of a promotion "
                "running on D. Brief section 18 calls this out explicitly. It is "
                "the end *date* that is known - the realised spend and units are "
                "not, and those are masked beyond the as-of date."
            ),
        ),
        FeatureSpec(
            name="days_to_next_promotion",
            group=FeatureGroupName.PROMOTION,
            description="Days until the next scheduled promotion begins.",
            source_columns=("start_date",),
            source_tables=("promotions",),
            transformation="merge_asof forward on start_date",
            temporality=_F,
            forward_justification=(
                "The forward promotion calendar is committed in advance and is "
                "genuinely visible to a planner. Predictive because trade demand "
                "typically softens just before a known promotion as buyers hold "
                "off - an effect a model cannot learn without this feature."
            ),
        ),
        FeatureSpec(
            name="days_since_promotion",
            group=FeatureGroupName.PROMOTION,
            description="Days since the previous promotion ended.",
            source_columns=("promotion_id",),
            source_tables=("promotions",),
            transformation="days since last true, shifted by 1",
            temporality=_B,
        ),
        FeatureSpec(
            name="promotions_last_28d",
            group=FeatureGroupName.PROMOTION,
            description="Promoted days in the 28 days ending yesterday.",
            source_columns=("promotion_id",),
            source_tables=("promotions",),
            transformation="shift(1).rolling(28).sum()",
            temporality=_B,
        ),
        FeatureSpec(
            name="promotions_last_90d",
            group=FeatureGroupName.PROMOTION,
            description="Promoted days in the 90 days ending yesterday.",
            source_columns=("promotion_id",),
            source_tables=("promotions",),
            transformation="shift(1).rolling(90).sum()",
            temporality=_B,
        ),
        FeatureSpec(
            name="promotion_spend",
            group=FeatureGroupName.PROMOTION,
            description=(
                "Realised spend on the active promotion. Historical analysis only "
                "- masked beyond the as-of date because it is an actual."
            ),
            source_columns=("promotion_spend",),
            source_tables=("promotions",),
            transformation="passthrough from the event",
            temporality=_B,
        ),
        FeatureSpec(
            name="promotion_intensity",
            group=FeatureGroupName.PROMOTION,
            description="Spend per day of the event, so long and short promos compare.",
            source_columns=("promotion_spend", "duration_days"),
            source_tables=("promotions",),
            transformation="promotion_spend / promotion_duration",
            temporality=_B,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

INVENTORY_GROUP = FeatureGroup(
    name=FeatureGroupName.INVENTORY,
    description=(
        "Availability. Closing inventory and sold units are excluded - both are "
        "functions of the day's sales."
    ),
    features=(
        FeatureSpec(
            name="inventory_available",
            group=FeatureGroupName.INVENTORY,
            description="Opening stock plus deliveries; known before any sale.",
            source_columns=("opening_inventory", "received_units"),
            source_tables=("inventory",),
            transformation="opening + received",
            temporality=_C,
        ),
        FeatureSpec(
            name="inventory_ratio",
            group=FeatureGroupName.INVENTORY,
            description="Availability over trailing 28-day mean demand.",
            source_columns=("opening_inventory", "received_units", "units"),
            source_tables=("inventory", "sales_daily"),
            transformation="inventory_available / shifted rolling_28 demand",
            temporality=_B,
        ),
        FeatureSpec(
            name="inventory_days_cover",
            group=FeatureGroupName.INVENTORY,
            description="Days of cover at trailing demand rate.",
            source_columns=("opening_inventory", "received_units", "units"),
            source_tables=("inventory", "sales_daily"),
            transformation="inventory_available / shifted rolling_28 demand",
            temporality=_B,
        ),
        FeatureSpec(
            name="stockout_yesterday",
            group=FeatureGroupName.INVENTORY,
            description="Whether yesterday was a stockout. Today's is not knowable.",
            source_columns=("stockout_flag",),
            source_tables=("inventory",),
            transformation="shift(1)",
            temporality=_B,
            dtype="bool",
        ),
        FeatureSpec(
            name="days_since_stockout",
            group=FeatureGroupName.INVENTORY,
            description="Days since the last stockout, counting prior days only.",
            source_columns=("stockout_flag",),
            source_tables=("inventory",),
            transformation="days since last true, shifted by 1",
            temporality=_B,
        ),
        FeatureSpec(
            name="stockouts_last_28d",
            group=FeatureGroupName.INVENTORY,
            description="Stockout days in the 28 days ending yesterday.",
            source_columns=("stockout_flag",),
            source_tables=("inventory",),
            transformation="shift(1).rolling(28).sum()",
            temporality=_B,
        ),
        FeatureSpec(
            name="stockouts_last_90d",
            group=FeatureGroupName.INVENTORY,
            description="Stockout days in the 90 days ending yesterday.",
            source_columns=("stockout_flag",),
            source_tables=("inventory",),
            transformation="shift(1).rolling(90).sum()",
            temporality=_B,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Temporal, product, store
# ---------------------------------------------------------------------------

TEMPORAL_GROUP = FeatureGroup(
    name=FeatureGroupName.TEMPORAL,
    description="Calendar attributes. Always knowable at any horizon.",
    features=(
        *(
            FeatureSpec(
                name=name,
                group=FeatureGroupName.TEMPORAL,
                description=description,
                source_columns=("date",),
                source_tables=("calendar",),
                transformation="derived from the date",
                temporality=_C,
            )
            for name, description in (
                ("day_of_week", "Monday = 0."),
                ("week_of_year", "ISO week number."),
                ("month", "Calendar month."),
                ("quarter", "Calendar quarter."),
                ("year", "Calendar year."),
                ("weekend_flag", "Saturday or Sunday."),
                ("holiday_flag", "Public holiday."),
                ("festival_flag", "Festival window including the run-up."),
                ("season", "Season label."),
                ("financial_month", "Indian financial year, April = 1."),
                ("financial_quarter", "Financial quarter."),
                ("dow_sin", "Cyclical encoding of day of week."),
                ("dow_cos", "Cyclical encoding of day of week."),
                ("month_sin", "Cyclical encoding of month."),
                ("month_cos", "Cyclical encoding of month."),
                ("doy_sin", "Cyclical encoding of day of year."),
                ("doy_cos", "Cyclical encoding of day of year."),
                ("time_index", "Days since the window start; linear trend term."),
            )
        ),
        FeatureSpec(
            name="days_to_festival",
            group=FeatureGroupName.TEMPORAL,
            description="Days until the next festival.",
            source_columns=("festival_flag",),
            source_tables=("calendar",),
            transformation="searchsorted over festival dates",
            temporality=_F,
            forward_justification=(
                "Festival dates are published years ahead. A planner on any date "
                "knows when the next one falls, so using it is not leakage - "
                "withholding it would remove information the business has."
            ),
        ),
        FeatureSpec(
            name="days_since_festival",
            group=FeatureGroupName.TEMPORAL,
            description="Days since the most recent festival.",
            source_columns=("festival_flag",),
            source_tables=("calendar",),
            transformation="searchsorted over festival dates",
            temporality=_B,
        ),
    ),
)


PRODUCT_GROUP = FeatureGroup(
    name=FeatureGroupName.PRODUCT,
    description="Product attributes and lifecycle position.",
    features=(
        *(
            FeatureSpec(
                name=name,
                group=FeatureGroupName.PRODUCT,
                description=description,
                source_columns=(name,),
                source_tables=("products",),
                transformation="joined; categorical dtype, not one-hot",
                temporality=_C,
                dtype="category",
            )
            for name, description in (
                ("category", "Product category."),
                ("subcategory", "Product subcategory."),
                ("brand", "Brand."),
                ("pack_size", "Pack size label."),
                ("product_status", "Active / Launched / Discontinued."),
            )
        ),
        FeatureSpec(
            name="unit_cost",
            group=FeatureGroupName.PRODUCT,
            description="Cost per unit; needed for margin and optimisation.",
            source_columns=("unit_cost",),
            source_tables=("products",),
            transformation="joined",
            temporality=_C,
        ),
        FeatureSpec(
            name="product_age_days",
            group=FeatureGroupName.PRODUCT,
            description="Days since launch, relative to the row's own date.",
            source_columns=("launch_date",),
            source_tables=("products",),
            transformation="row date - launch_date",
            temporality=_C,
        ),
        FeatureSpec(
            name="is_new_product",
            group=FeatureGroupName.PRODUCT,
            description="Within 90 days of launch; distribution still building.",
            source_columns=("launch_date",),
            source_tables=("products",),
            transformation="product_age_days < 90",
            temporality=_C,
            dtype="bool",
        ),
    ),
)


STORE_GROUP = FeatureGroup(
    name=FeatureGroupName.STORE,
    description="Store attributes and scale.",
    features=(
        *(
            FeatureSpec(
                name=name,
                group=FeatureGroupName.STORE,
                description=description,
                source_columns=(name,),
                source_tables=("stores",),
                transformation="joined; categorical dtype, not one-hot",
                temporality=_C,
                dtype="category",
            )
            for name, description in (
                ("store_type", "Flagship or Standard."),
                ("channel", "Hypermarket, Supermarket, Convenience, E-commerce, Wholesale."),
                ("region", "Region."),
                ("state", "State."),
            )
        ),
        FeatureSpec(
            name="store_size_sqft",
            group=FeatureGroupName.STORE,
            description="Floor area; 0 for e-commerce.",
            source_columns=("store_size_sqft",),
            source_tables=("stores",),
            transformation="joined",
            temporality=_C,
        ),
        FeatureSpec(
            name="store_age_days",
            group=FeatureGroupName.STORE,
            description="Days since opening, relative to the row's own date.",
            source_columns=("opening_date",),
            source_tables=("stores",),
            transformation="row date - opening_date",
            temporality=_C,
        ),
    ),
)


#: Every group, keyed by name.
FEATURE_GROUPS: dict[FeatureGroupName, FeatureGroup] = {
    group.name: group
    for group in (
        DEMAND_GROUP,
        PRICE_GROUP,
        COMPETITOR_GROUP,
        PROMOTION_GROUP,
        INVENTORY_GROUP,
        TEMPORAL_GROUP,
        PRODUCT_GROUP,
        STORE_GROUP,
    )
}

#: Flat lookup by feature name.
FEATURE_SPECS: dict[str, FeatureSpec] = {
    spec.name: spec for group in FEATURE_GROUPS.values() for spec in group.features
}


def spec_for(name: str) -> FeatureSpec | None:
    return FEATURE_SPECS.get(name)


def features_in_group(group: FeatureGroupName) -> list[str]:
    return FEATURE_GROUPS[group].names()


def forward_looking_features() -> list[FeatureSpec]:
    """Every feature that reads beyond its row date.

    The list a reviewer should scrutinise, and the one the leakage tests assert
    is exactly as expected - an unexpected addition means something started
    looking forward without anyone deciding it should.
    """
    return [
        spec for spec in FEATURE_SPECS.values() if spec.temporality is Temporality.FORWARD_PLANNED
    ]


# ---------------------------------------------------------------------------
# Per-model requirements (brief section 24)
# ---------------------------------------------------------------------------

MODEL_REQUIREMENTS: dict[str, ModelFeatureRequirement] = {
    requirement.model_name: requirement
    for requirement in (
        ModelFeatureRequirement(
            model_name="baseline_sales",
            description="Expected sales absent promotions and abnormal events.",
            required_groups=(
                FeatureGroupName.DEMAND,
                FeatureGroupName.TEMPORAL,
                FeatureGroupName.PRICE,
                FeatureGroupName.PROMOTION,
                FeatureGroupName.INVENTORY,
                FeatureGroupName.PRODUCT,
                FeatureGroupName.STORE,
            ),
            target="units",
            caveats=(
                "Lagging observed units propagates stockout censoring forward. Use "
                "demand.mask_censored so a supply failure does not read as weak demand.",
                "Baseline must represent demand *without* promotion, so the promotion "
                "features are controls to be zeroed at prediction, not drivers to keep.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="demand_forecast",
            description="7/14/30/90-day demand forecast.",
            required_groups=(
                FeatureGroupName.DEMAND,
                FeatureGroupName.TEMPORAL,
                FeatureGroupName.PRICE,
                FeatureGroupName.PROMOTION,
                FeatureGroupName.INVENTORY,
                FeatureGroupName.COMPETITOR,
                FeatureGroupName.PRODUCT,
                FeatureGroupName.STORE,
            ),
            required_features=("lag_7_units", "lag_28_units", "rolling_28_units"),
            target="units",
            caveats=(
                "Competitor price is observed, so it does not exist over the forecast "
                "horizon. Carry the last observed value forward and say so, rather "
                "than training on a value that will be absent at inference.",
                "Realised promotion spend is unavailable for future dates; build with "
                "include_promotion_spend=False.",
                "Validate with expanding-window splits. Random k-fold lets the model "
                "see the future while predicting the past.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="promo_uplift",
            description="Incremental sales caused by a promotion.",
            required_groups=(
                FeatureGroupName.DEMAND,
                FeatureGroupName.PROMOTION,
                FeatureGroupName.PRICE,
                FeatureGroupName.TEMPORAL,
                FeatureGroupName.PRODUCT,
                FeatureGroupName.STORE,
            ),
            required_features=("promotion_flag", "promotion_discount", "rolling_28_units"),
            target="units",
            caveats=(
                "Pre-period features must come from before the promotion started, not "
                "from the window itself.",
                "Post-promotion pull-forward means a naive during-vs-before comparison "
                "overstates incrementality. A control group is required, not optional.",
                "Promotions are targeted at softening products, so treatment is not "
                "random and a raw difference is biased.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="trade_promo_optimization",
            description="Budget allocation across products, regions and retailers.",
            required_groups=(
                FeatureGroupName.PROMOTION,
                FeatureGroupName.PRODUCT,
                FeatureGroupName.STORE,
            ),
            required_features=("promotion_spend", "unit_cost"),
            caveats=(
                "Consumes uplift and forecast *outputs*, not raw features - it runs "
                "after Steps 5 and 6.",
                "Promotional response saturates. A linear ROI-per-rupee sends the whole "
                "budget to one cell, which is wrong and obviously so.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="price_elasticity",
            description="Own-price elasticity of demand.",
            required_groups=(
                FeatureGroupName.PRICE,
                FeatureGroupName.DEMAND,
                FeatureGroupName.PROMOTION,
                FeatureGroupName.COMPETITOR,
                FeatureGroupName.TEMPORAL,
                FeatureGroupName.PRODUCT,
                FeatureGroupName.STORE,
            ),
            required_features=("selling_price", "price_index", "competitor_price"),
            target="units",
            caveats=(
                "Price is endogenous - it responds to anticipated demand. Fixed effects, "
                "the commodity cost instrument, or the randomised_test price subset are "
                "the three available identification strategies.",
                "Estimate on non-promotional, in-stock rows. Promotional rows carry a "
                "price cut and an additive uplift at once, which inflates the coefficient.",
                "Stockout rows report supply, not demand, and bias the estimate toward zero.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="cross_price_elasticity",
            description="How one product's price moves another's demand.",
            required_groups=(
                FeatureGroupName.PRICE,
                FeatureGroupName.DEMAND,
                FeatureGroupName.PROMOTION,
                FeatureGroupName.TEMPORAL,
                FeatureGroupName.PRODUCT,
            ),
            required_features=("selling_price", "price_index"),
            target="units",
            caveats=(
                "Control for the target's own price. Same-category products share a cost "
                "index, so without it the target's own elasticity swamps the cross effect "
                "and flips its sign.",
                "Estimate at store-date grain: substitution happens on a shelf, and "
                "aggregating to product-date averages away the identifying variation.",
                "Restrict candidates using product_relationships and correct for multiple "
                "comparisons - N products give N(N-1) ordered pairs.",
            ),
        ),
        ModelFeatureRequirement(
            model_name="price_optimization",
            description="Profit- or revenue-maximising price under constraints.",
            required_groups=(
                FeatureGroupName.PRICE,
                FeatureGroupName.COMPETITOR,
                FeatureGroupName.INVENTORY,
                FeatureGroupName.PRODUCT,
            ),
            required_features=("selling_price", "unit_cost", "competitor_price"),
            caveats=(
                "Consumes elasticity and forecast outputs; it does not fit a demand model "
                "of its own.",
                "Optimising a product in isolation recommends a rise that merely shifts "
                "volume to its own category neighbour. Cross-price effects are required "
                "input, not a refinement.",
                "A wide elasticity confidence interval must widen the recommended price "
                "range, not be discarded for a crisp-looking optimum.",
            ),
        ),
    )
}


def requirement_for(model_name: str) -> ModelFeatureRequirement | None:
    return MODEL_REQUIREMENTS.get(model_name)
