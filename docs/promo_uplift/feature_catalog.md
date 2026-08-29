# Feature catalogue

Every covariate in the adjustment set, and every column deliberately kept out.

The rule: **a covariate must have been determined before the promotion started.**
A variable measured after treatment closes no back-door path — it either blocks
the causal path being measured, or opens a new one.

## Anchoring

| Row type | Covariates measured as of |
|---|---|
| Control | its own date |
| **Treated** | **the event start** |

Not the row's own date. On day five of a promotion, a trailing 7-day mean
anchored at the row contains four days of the effect being estimated: the
covariate carries the treatment effect, the outcome model explains the outcome
using it, and the estimate shrinks toward zero.

There is **no subtraction of a day** at the anchor. The trailing statistics are
computed on `shift(1)`, so the value indexed at date `D` already covers `D−1` and
earlier. Subtracting again would shift the window twice and discard the most
recent — and most informative — day of history. The exclusion of the current day
lives in one place.

Asserted by test: covariates are constant within an event, and `demand_mean_7`
matches a trailing mean reconstructed by hand from the source panel.

## In the adjustment set

### Demand history (anchored)

| Feature | Definition | Causal role |
|---|---|---|
| `demand_lag_1/7/14/28` | Units 1, 7, 14, 28 days before the anchor | Confounder — level drives both promotion choice and sales |
| `demand_mean_7/14/28/56` | Trailing means | Confounder |
| `demand_std_14/28` | Trailing standard deviations | Confounder — volatile listings are promoted differently |
| `demand_momentum_7_28` | `mean_7 / mean_28` | Confounder — running hot or cold is what a merchandiser reacts to |
| `demand_volatility` | `std_28 / mean_28` | Confounder |
| `demand_log_level` | `log1p(mean_28)` | Confounder — the level on the scale the outcome model works on |

### Prior promotion intensity (anchored)

| Feature | Definition | Causal role |
|---|---|---|
| `promo_share_28` | Share of the last 28 days promoted | **Confounder** — the strongest single predictor of being promoted again, and heavily promoted listings differ in demand |
| `promo_share_90` | Share of the last 90 days promoted | Confounder |
| `days_since_promotion` | Days since the last promoted day | Confounder |

### Calendar and season (row date — not affected by treatment)

| Feature | Definition | Causal role |
|---|---|---|
| `season_sin_1`, `season_cos_1` | Annual harmonic | **THE confounder.** Promotion timing is drawn with weights `exp(targeting × 2 × seasonal)` |
| `season_sin_2`, `season_cos_2` | Half-year harmonic | Confounder — asymmetric seasonal shape |
| `day_of_week`, `is_weekend`, `month` | Calendar position | Confounder |
| `time_index` | Days since the panel start | Confounder — promotions are not uniform across the window, so an untreated drift would load onto treatment |
| `holiday_flag`, `festival_flag` | When present | Confounder |

Harmonics rather than an empirical seasonal index estimated from the data. An
empirical index fitted on control rows would be contaminated by exactly the
selection it corrects for: promotions cluster at seasonal peaks, so the control
rows under-represent those peaks and the estimated curve is flattened there.

The propensity model constructs **season × category interactions** explicitly.
The relationship between date and treatment differs *by category*, and a model
with additive season and additive category cannot represent that — the back-door
path stays open however well the model fits.

### Price level (anchored)

| Feature | Definition | Causal role |
|---|---|---|
| `regular_price_lag_1` | Shelf price before the anchor | Confounder — the *regular* price, not the promotional one |
| `price_vs_trailing_mean` | Regular price ÷ its 56-day mean | Confounder |

### Static

`category`, `region`, `channel`, `brand`, `store_segment`, `store_tier` —
stratify the comparison and carry the seasonal interaction.

---

## Deliberately excluded

### Mediators — consequences of treatment

| Column | Why excluded |
|---|---|
| **`selling_price`** | Determined by the promotion. Conditioning holds the price cut fixed across arms |
| **`discount_percentage`** | Same |
| `promotion_flag`, `promotion_type`, `promotion_spend`, `promotion_units` | Definitionally post-treatment |
| `days_into_promotion`, `days_until_promotion_end` | Only defined for treated rows |
| `display_flag`, `bundle_flag` | Part of the treatment |

**This is the highest-risk exclusion in the package.** Discount looks like an
obviously relevant covariate, and including it is the natural thing to do. It
would reduce the estimate from **+71.3% to +17.7%** on this data — a plausible
number, wrong by a factor of four.

### Colliders

| Column | Why excluded |
|---|---|
| **`stockout_flag`** | Caused by the promotion *and* correlated with demand. Conditioning **opens** a closed path |
| `inventory_available`, `opening_inventory`, `closing_inventory`, `inventory_days` | Supply-side, same argument. Step 5 measured that inventory features teach a model to read low stock as low demand |

`stockout_flag` is used to *filter rows* and never as a covariate. The filtering
is itself a compromise — see [`assumptions.md`](assumptions.md#4-stockouts-and-the-estimand-they-change).

### Outcome and its arithmetic

`units`, `revenue`, `cost`, `gross_profit`, `sold_units`.

### Identifiers

`date`, `product_id`, `store_id`, `promotion_id` — carried through the frame,
never fitted on. A model given raw ids memorises listings instead of learning
what makes them promotable, and cannot generalise to a listing it has not seen.

### Simulation truth

`latent_units`, `mean_demand`, `lost_units`, and everything in
`GROUND_TRUTH_COLUMNS`. Stripped by `SyntheticPanel.observable()` rather than
left for callers to ignore.

---

## The two guards

**Allow-list (the real protection).** `feature_names` is built only from the
three constructed groups. A column cannot become a covariate by appearing in the
panel — it has to be built by one of the feature builders. That is what stops a
new post-treatment column nobody thought to exclude from silently entering the
set.

**Deny-list (belt and braces).** `_assert_no_post_treatment` raises at runtime if
an excluded name reaches the feature list. Note it cannot fire from the normal
path, because the same set filters the list — so it guards against a future
change that adds names after filtering, not against today's code. The tests
exercise it directly rather than pretending otherwise.

## Handling missing values

Rows without a complete covariate set are **dropped, not imputed**. A listing
three weeks old has no 56-day trailing mean, and filling it with the panel
average asserts that this listing runs at the average rate — a claim nobody made
and one the estimator would then treat as evidence.

The count and share of dropped rows are logged and reported. If every treated row
is dropped, `InsufficientPrePeriodError` is raised rather than an estimate
produced from whichever rows happened to survive.
