# Feature Catalogue & Point-in-Time Correctness

Reference for the feature layer: what exists, what each model needs, and how
leakage is prevented.

Definitions live in code — [features/contracts/catalogue.py](../../features/contracts/catalogue.py)
— because a definition is behaviour and belongs under review. Selection lives in
[configs/features/features.yaml](../../configs/features/features.yaml), because a
selection is a knob.

---

## Point-in-time correctness

The property everything else here serves. Every model in Steps 4–11 is temporal,
and a single leaked future value produces a model that backtests beautifully and
fails in production — silently, because the frame is well-formed and the metrics
look excellent.

### Availability is per-table, not global

The naive reading of "as-of date" is *clamp everything to D*. That is wrong and
would cripple forecasting. A planner on 1 June genuinely knows the promotion
calendar for 10–24 June, and genuinely knows next Diwali's date. Clamping those
deletes information the business actually has.

So each table is classified in
[data/repositories/availability.py](../../data/repositories/availability.py):

| Class | Tables | Visible past as-of? |
|---|---|---|
| `OBSERVED` | `sales_daily`, `inventory`, `competitor_pricing`, `trade_promotions`, `commodity_costs` | **No** |
| `KNOWN_IN_ADVANCE` | `calendar`, `promotions`, `pricing` | **Yes** |
| `STATIC` | `products`, `stores`, `customers`, `product_relationships` | N/A |

An unclassified table defaults to `OBSERVED`. Forgetting therefore costs signal,
never correctness.

### The dangerous class, and its safeguard

`KNOWN_IN_ADVANCE` is the one that lets data through. Membership is small and
each entry is justified — but a planned table can still carry *actuals*. Next
month's promotion schedule is knowable; the spend it will eventually book is not.

`PointInTimeView` nulls those columns beyond the as-of date:

```python
ACTUALS_COLUMNS = {"promotions": ("promotion_spend", "promotion_units")}
```

Without this, a model could read a future promotion's realised spend — which is a
function of the demand it is trying to predict.

### Structural, not procedural

An `as_of_date` keyword exists on every method for the brief's §11, but the
*view* is the mechanism:

```python
view = repository.as_of(date(2024, 6, 30))
engineer = FeatureEngineer(view)     # a bare repository raises TypeError
```

A keyword can be forgotten — one omission in one builder and the model trains on
the future. A view has no method that returns future observed data, so the
mistake is unavailable rather than merely discouraged.

### The shift discipline

Every temporal feature routes through
[features/engineering/panel.py](../../features/engineering/panel.py):

```python
shifted_group(panel, "units", periods=7)          # rejects periods < 1
rolling_on_shifted(panel, "units", window=7)      # shift(1) then roll
```

`df.groupby(k).rolling(7)` reads perfectly well and silently includes the current
row — so `rolling_7_sales` would contain one seventh of the number being
predicted. Doing the shift in one shared helper means it is right everywhere or
wrong everywhere, and the tests pin it as right.

### Target-derived columns

`revenue = units × selling_price`. A model given revenue and price recovers units
exactly. These are dropped from every feature panel by default:

```
revenue, cost, gross_profit, sold_units, closing_inventory,
inventory_days, promotion_units
```

Centralised rather than left to each consumer, on the same reasoning: expecting
seven future models to each remember that revenue is the target in disguise is
how one of them forgets.

---

## Feature groups

| Group | Count | Temporality |
|---|---|---|
| `demand` | 17 | all backward |
| `price` | 8 | contemporaneous (price is set by us) |
| `competitor` | 6 | backward (their price is observed) |
| `promotion` | 12 | mixed — schedule contemporaneous/forward, spend backward |
| `inventory` | 7 | availability contemporaneous, history backward |
| `temporal` | 20 | contemporaneous, plus `days_to_festival` forward |
| `product` | 8 | contemporaneous |
| `store` | 6 | contemporaneous |

### Key definitions

`price_index`
: Own price ÷ same-day category mean. Category rather than a fixed basket,
  because the comparison a shopper makes is against the alternatives on the shelf
  — and a fixed reference goes stale as the assortment changes.

`price_gap` / `price_ratio`
: `own − competitor` and `own ÷ competitor`. Both, because they answer different
  questions: the gap is what a category manager negotiates in, the ratio is
  scale-free and enters a log-log demand model linearly.

`inventory_available`
: `opening + received`. Knowable before a single unit sells, unlike
  `closing_inventory`, which is a function of the day's sales.

`demand_momentum`
: 7-day mean ÷ 28-day mean, both shifted. Above 1 means accelerating.

---

## The three forward-looking features

Only these may read beyond their row date. The list is pinned in
`features.yaml` and asserted by the leakage tests, so adding a fourth is a
deliberate, reviewable act.

| Feature | Why it is knowable |
|---|---|
| `days_until_promotion_end` | Promotion mechanics are agreed with retailers weeks ahead (brief §18 says so explicitly). The end *date* is known; the realised spend is not. |
| `days_to_next_promotion` | Same. Predictive because trade demand softens just before a known promotion as buyers hold off. |
| `days_to_festival` | Festival dates are published years ahead. Withholding this removes information the business has. |

Constructing a `FORWARD_PLANNED` spec without a justification raises at import.

---

## Model requirements (brief §24)

Recorded in the catalogue before any model exists, so Steps 4–11 inherit a
stated contract. Each carries `caveats` — the modelling hazards specific to it.
Selected ones:

**`price_elasticity`**
> Price is endogenous — it responds to anticipated demand. Fixed effects, the
> commodity cost instrument, or the `randomised_test` price subset are the three
> available identification strategies.
>
> Estimate on non-promotional, in-stock rows. Promotional rows carry a price cut
> and an additive uplift at once, which inflates the coefficient.

**`cross_price_elasticity`**
> Control for the target's own price. Same-category products share a cost index,
> so without it the target's own elasticity swamps the cross effect and flips its
> sign.
>
> Estimate at store-date grain: substitution happens on a shelf.

**`demand_forecast`**
> Competitor price is observed, so it does not exist over the forecast horizon.
> Carry the last observed value forward and say so, rather than training on a
> value that will be absent at inference.

These are not decoration. Step 2's validation demonstrated each of them
empirically — the cross-price control took sign agreement from 5/8 to 8/8.

---

## The five dataset builders

| Builder | Target | Framing decision |
|---|---|---|
| `create_forecasting_dataset` | `units` | Excludes realised promo spend — absent at inference |
| `create_price_elasticity_dataset` | `log_units` | Excludes promotional and stockout rows |
| `create_promo_uplift_dataset` | `units` | Labels treatment/control and pre/post periods |
| `create_cross_price_dataset` | `log_demand_a` | Scoped by `product_relationships`; store grain |
| `create_promo_optimization_dataset` | — | Leaves `forecast_sales`/`uplift` null for Steps 5–6 |

Each returns a `FeatureSet` with `X` and `y` separated. Returning one combined
frame and trusting the caller to drop the target is how the target ends up in the
feature matrix.

---

## Lineage (brief §31)

Every feature set carries `FeatureSetMetadata`:

```
feature_set_name, feature_version, dataset_version, as_of_date,
start_date, end_date, source_tables, feature_names, target_name,
row_count, generated_at, code_version (git sha), request_hash
```

`dataset_version` is as-of qualified — `v1.0+3ec67bbf@2024-06-30` — because two
feature sets built from the same data at different as-of dates are different
artifacts, and MLflow in Step 12 must not record them as identical inputs.

`cache_key()` includes `feature_version`, so bumping it invalidates every cached
set. That is the behaviour you want when the definitions changed underneath.

---

## Running it

```python
from datetime import date
from app.services.container import Container
from features.datasets import create_forecasting_dataset

repo = Container().data_repository
view = repo.as_of(date(2025, 12, 3))          # freeze the world

dataset = create_forecasting_dataset(
    view, train_start=date(2025, 2, 7), train_end=date(2025, 12, 3),
    product_ids=[...], store_ids=[...],
)
X, y = dataset.X, dataset.y
print(dataset.describe())
```

```powershell
uv run pytest tests/leakage -v     # the property that matters
```
