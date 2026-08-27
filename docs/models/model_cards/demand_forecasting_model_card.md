# Model Card — Demand Forecasting Model

| | |
|---|---|
| **Name** | `demand_forecast` |
| **Version** | v1.0 |
| **Stage** | Stage 1, Step 5 |
| **Type** | Direct multi-step demand forecasting (global pooled model) |
| **Status** | Local MVP — not production |
| **Horizons** | 7, 14, 30, 90 days |
| **Grain** | product × store × day, aggregated on request |
| **Tracking** | MLflow experiment `revenue_intelligence_forecasting` |

---

## Intended use

### What it is for

Predicting **units that will sell** over the next 7/14/30/90 days, given the
planned price and promotion calendar. Consumed by demand planning, inventory
sizing, and — from Step 16 — the `forecast_demand` agent tool.

### What it is *not* for

- **Estimating what a promotion caused.** This is predictive. It says what is
  likely given the plan; incremental effect is the uplift model's question. An
  agent conflating the two will credit a seasonal peak to whatever campaign
  happened to be live.
- **Price elasticity.** Price is a feature, not a lever here. Asking "what if we
  cut price 10%?" requires the elasticity model; this model would answer with
  whatever correlation it learned, confounded by why prices moved historically.
- **Shipment or replenishment planning.** It forecasts *demand*, deliberately
  excluding all inventory signal. A planner asking "how much will we ship if we
  stock out" wants a different model.
- **Individual days at long horizons.** Error grows with horizon by construction.
  Aggregate before acting.
- **Dates past 2025-12-31.** The promotion and price calendars end there; the
  service refuses rather than assuming.
- **Products or stores not in the trained series set.** Reported as
  `insufficient_data`, never as an empty forecast — "no demand expected" and
  "not in the model" are very different claims.

---

## Training data

| | |
|---|---|
| Source | Step 2 synthetic CPG/Retail dataset |
| Real product-store series | 6,128 |
| Series sampled (default) | 800, volume-stratified, store-clustered |
| Origins | every 7th day, after a 400-day warmup |
| Horizon draws per origin | 8, drawn at random from U{1..90} |
| Period | 2023-01-01 → 2025-12-31 |

### Rows deliberately excluded

| Exclusion | Why |
|---|---|
| Targets falling on a stockout | The target is censored — it records availability, not demand |
| Origins without a full year of history | The 364-day lag is undefined; the model would train on mostly-NaN rows |
| Origins inside the embargo band | Their targets would land in the evaluation window |

**Stockout origins are kept.** A stockout at the origin is a legitimate knowable
state, and dropping those origins biases the feature distribution for no gain.

### Not used as features

Everything Step 4 excludes, plus:

- **All supply-side columns** (`SUPPLY_FEATURES`, plus `received_units` and
  `sold_units_lag_1`). Step 4 measured the consequence of including them:
  LightGBM recovered only 0.30 of true stockout demand because it had learned
  that low stock predicts low sales.
- **`time_index`** — anchored to the frame's own minimum, so the same calendar
  date gets a different value at training and serving time. No leakage test on
  the training frame would ever reveal that.
- **`year` / `financial_year`** — a year is either one the model has seen, and it
  overfits, or one it has not, and it cannot place it.
- **`part`** — the hive partition key. A storage artifact a tree will happily
  split on as a coarse date proxy, and one that does not exist for a future date.

**No personal or customer-level data is used.**

---

## Model

| | |
|---|---|
| Architecture | Direct multi-step, one global model, `horizon_step` as a feature |
| Candidates | horizon naive, horizon seasonal naive, LightGBM, XGBoost |
| Objective | Poisson (both gradient-boosted candidates) |
| Early stopping | Against a temporally later validation fold |
| Intervals | Split conformal, per horizon bucket + a separate aggregate quantile |
| Statistical comparison | Weekly ETS at total/region/category/category×region |

### Why one global model rather than one per series

6,128 series would mean 6,128 models to fit, store, version and monitor, each on
~1,000 observations. A pooled model shares strength across series — a slow-moving
SKU borrows the seasonal shape its category exhibits — and product/store identity
enters as features rather than as a partition key. This is also why per-series
SARIMA is not attempted: the measured per-fit cost, extrapolated across the
catalogue, is hours per backtest fold.

### Selection rules

Accuracy on the test fold, with a **simplicity preference**: if the seasonal naive
lands within two percentage points of the best model, it wins. §9 is explicit that
the most complex model does not win by default, and a benchmark that holds its own
is telling you the signal is simple.

A warning is emitted — not suppressed — if error does **not** grow with horizon
for the selected model, because that is the signature of a leaking join.

---

## Evaluation

### Splits

Origin-based, chronological, with a **90-day embargo** between every fold:
train → calibration (60d) → validation (90d) → test (120d). The embargo exists
because a training origin near a boundary otherwise has its target inside the
evaluation window.

Walk-forward validation expands over origins, and applies the same embargo — a
backtest without it would be systematically more optimistic than the test number.

### Metrics

WMAPE headline (volume-weighted). MAPE only over non-zero actuals, with the
exclusion count reported, because zero and near-zero demand is common here and a
silent `inf` is worse than an absent number.

**Everything is reported per horizon bucket and never blended.** A single WMAPE
averaged over 1–90 days describes no decision anyone makes.

### Forecast Value Added

Measured in WMAPE percentage points against a horizon-aware seasonal naive, broken
out by bucket. **A bucket where FVA ≤ 0 is reported as such**: at those horizons
the benchmark is the honest choice, and §45 asks for that to be visible rather
than smoothed over.

### Measured results (800 series, 548,754 rows, seed 42)

| | |
|---|---|
| Selected model | **XGBoost**, Poisson objective |
| Test WMAPE | **43.8%** |
| Irreducible noise floor (Step 4) | 35.0% |
| **Ratio to floor** | **1.25×** |
| Bias | +8.4% |
| FVA vs seasonal naive | **+11 to +15 points at every horizon** |
| Backtest stability | 0.3–1.8pp std across folds, all buckets stable |
| WMAPE at total level | 9.6% |
| Training time | 475s for the full pipeline |

XGBoost beat LightGBM by 3.7 points — a larger gap than expected, and one that
only appeared after the `min_child_weight` scaling fix described below. LightGBM's
FVA decays with horizon (+13.1 → +5.7) while XGBoost's holds (+15.3 → +11.0).

**The horizon gradient is shallow** — 43.6% at h1-3 versus 44.7% at h57-90.
Reported rather than smoothed because it is flatter than a forecasting curve
usually looks. The cause is the noise floor: at 1.25× the irreducible error there
are only ~9 points of learnable signal in total, so a steep curve is
arithmetically unavailable. The gradient is consequently weak evidence about join
correctness here, and the leakage tests carry that argument instead.

### The interpretability floor

Step 4 measured the irreducible noise floor on this data at **35% WMAPE** — the
score a model knowing the true conditional mean would still achieve, because
demand is drawn from an over-dispersed negative binomial. Forecast WMAPE should be
read against that, not against zero. A forecast scoring below it has seen the
answer.

---

## Limitations and risks

| Limitation | Consequence |
|---|---|
| **Excluding stockout targets may bias low** | Removes the high-demand tail. Stockouts are endogenous — latent demand during one runs ~1.57× normal — so this is a real cost, measured and reported |
| **No inventory signal** | Forecasts demand, not shipments. Correct for this model's purpose, wrong for replenishment |
| Competitor prices frozen at the origin | `competitor_pricing` is OBSERVED; a competitor move inside the horizon is invisible |
| No cannibalisation or halo | A promotion on one SKU distorts its substitutes' forecasts, unattributed |
| Conformal assumes exchangeability | A trend violates it; coverage is measured per bucket so a shortfall surfaces rather than hiding |
| Cannot forecast past the calendar | Refused explicitly. A data limitation, not a modelling one |
| Default trains on 800 of 6,128 series | The sampled artifact writes to its own directory so it cannot be mistaken for the full model |
| Synthetic data | Real retail has structure this simulator does not reproduce |

### Failure modes to watch

- **A leaking join** would make long-horizon accuracy match short-horizon
  accuracy. Guarded behaviourally: bucket error must grow with horizon, and the
  pipeline warns when it does not.
- **Silent train/serve skew.** Training reads target features from history,
  serving from a future scaffold. Both go through one function, and a test asserts
  they agree. This already caught two real defects — missing festival columns over
  short windows, and per-frame categorical inference.
- **Long-horizon degradation** is where a stale model shows first, which is why
  monitoring reports per bucket.

---

## Ethical and safety considerations

- **No personal data.** Aggregate product/store/day throughout.
- **No fabricated confidence.** The `confidence` field is measured interval
  coverage or `None`. It is never a plausible-looking default.
- **No fabricated forecasts.** A horizon reaching past known planning data is
  refused, not filled with an assumption that no promotions are scheduled.
- **Uncertainty travels with the number.** Every response carries per-bucket
  accuracy, the interval, and the assumptions behind it.
- **Fallbacks are labelled.** `fallback_used` and `fallback_reason` are part of the
  response, because a caller must be able to tell an estimate from a guess.
- **Commercial impact.** Outputs influence inventory and promotion decisions.
  Aggregate before acting; treat individual long-horizon days as directional.

---

## Maintenance

| | |
|---|---|
| Retrain trigger | New dataset version, feature drift, or backtest instability |
| Monitor | Per-bucket WMAPE and bias, interval coverage, feature drift (PSI), prediction spread |
| Artifacts | `data/local/models/forecasting[_sampled]/` (joblib + JSON sidecar) |
| Reproduce | `uv run python scripts/train_forecast.py --seed 42` |

Every run records a **config fingerprint** — one hash over the whole
configuration — in MLflow params. Two runs sharing it used the same setup; two
that differ are not comparable, and the difference is discoverable rather than
argued about.
