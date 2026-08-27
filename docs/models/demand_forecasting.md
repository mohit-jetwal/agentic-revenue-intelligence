# Demand Forecasting Model

> Stage 1, Step 5. The first genuinely *predictive* capability: what demand
> **will** be over the next 7, 14, 30 or 90 days.

## The problem Step 4 left behind

Step 4's baseline predicts units at date *D* using features at *D* — including
`lag_1_units`, yesterday's sales. That is legitimate for a historical
counterfactual, where yesterday is known.

It is invalid for forecasting. Standing at as-of *T* predicting *T+30*, you do
not know *T+29*'s sales. **Every design decision in this step follows from that
one gap.**

| | Baseline (Step 4) | Forecast (Step 5) |
|---|---|---|
| Question | What *would* have happened? | What *will* happen? |
| Features and target | Same date | Features at *t*, target at *t+h* |
| Includes planned promotions | No | **Yes** |
| Uses yesterday's sales | Yes, legitimately | Only relative to the **origin** |
| Evaluated against | Latent demand | Held-out future outcomes |

## The design decision that carries the step

**Direct multi-step, one global model, with horizon step `h` as a feature.**

A training row is `(origin t, horizon step h, target = units at t+h)`. Each
feature is placed by asking one question — *is this knowable at t?* — and Step 3's
availability classes already answer it:

| Feature family | Sourced at | Why that is legitimate |
|---|---|---|
| lags, rollings, demand dynamics | **origin `t`** | `sales_daily` is `OBSERVED`, clamped to as-of |
| competitor price and gap | **origin `t`** | `competitor_pricing` is `OBSERVED` — *not* knowable forward |
| price position and promo history | **origin `t`** | derived from observed sales |
| calendar, festival, season | **target `t+h`** | `calendar` is `KNOWN_IN_ADVANCE` |
| planned promotion | **target `t+h`** | `promotions` is `KNOWN_IN_ADVANCE` |
| planned price | **target `t+h`** | `pricing` is `KNOWN_IN_ADVANCE` |
| product/store attributes | either | `STATIC` |
| `horizon_step` | — | lets one model span every horizon |

Target-side columns carry an **`h_` prefix**, so the origin/target split is
visible in the feature names and in the importance table rather than being a fact
you have to remember.

### Why not the alternatives

- **Four per-horizon models on a cumulative target** — dispositive: it cannot
  produce `ForecastResult.points`, which the interface requires. It also makes
  D7/D30/D90 mutually incoherent, since the nested totals are fitted
  independently.
- **Recursive one-step-ahead** — the specific killer is not error compounding but
  *feature-distribution collapse*. Feeding conditional-mean predictions forward
  drives `rolling_28_units_std`, `demand_volatility` and `demand_momentum` toward
  zero, so by h=30 the model sees inputs it never saw in training. Also 90
  sequential predict calls per as-of, and conformal has no valid construction
  under recursion.
- **Four direct models on daily targets** — 4× the compute, less data per model
  (the D7 model cannot learn from h=60 rows), and no benefit once `h` is already
  a feature.

### Horizon steps are drawn at random

Not from a fixed grid. With a grid, the model's splits on `horizon_step` are
piecewise-constant, which appears as a visible **staircase in the daily forecast
path** — and that path is a deliverable, not an internal detail. Random draws
cover 1..90 at the same row count.

---

## Stockouts and censoring

Approach A from §5: **exclude rows whose *target* fell on a stockout.** The target
is censored there — it records what was available to sell, not what customers
wanted.

Three qualifications matter:

1. **Stockout *origins* are kept.** A stockout at the origin is a legitimate
   knowable state. Dropping those origins would bias the feature distribution for
   no gain, since the target is not corrupted there.
2. **`mask_censored` is applied before lags are computed.** Step 3 shipped this
   function and nothing used it — its docstring defers the decision to the model,
   and this is the model that should take it. Without it a stockout depresses the
   next eight weeks of lag features, so the model learns the supply failure
   through the back door even though the stockout rows were excluded.
3. **The resulting bias is measured, not assumed.** Stockouts are endogenous —
   latent demand during one runs ~1.57× normal — so dropping them removes the
   high-demand tail. In Step 4 that bias was *intended*, because a baseline is
   defined as normal-conditions demand. **For a forecast the number is supposed
   to include peaks, so the same bias is now a defect rather than a definition.**

**No supply-side feature is used.** Step 4 measured what happens when they are:
LightGBM recovered only 0.30 of true demand during stockouts because it had
learned that low stock predicts low sales. This is a *demand* forecast; a
sales/shipment forecast admitting inventory would be a different product, and is
named in the limitations rather than built.

---

## Validation

### The split needs something Step 4's did not

An **embargo of `max_horizon` days** between folds. Without it, a training origin
sitting just before the boundary has its **target inside the evaluation window** —
the model is fitted on the very outcomes it is about to be scored on, the test
metric improves, and nothing raises.

```
train ──────► [embargo 90d] ──► calibration ──► [90d] ──► validation ──► [90d] ──► test
                                     60d                      90d                   120d
```

It costs real training data — 90 days of origins per boundary — and that cost is
the point.

Everything splits on the **origin date**, never the row. Rows sharing an origin
see identical history and differ only in `horizon_step`, so splitting by row
would scatter one origin's rows across train and test.

### Metrics, always per horizon bucket

WMAPE headline, plus MAE/RMSE/Bias and MAPE over non-zero actuals with the
exclusion count — all reused from Step 4. Reported per bucket
`{1-3, 4-7, 8-14, 15-28, 29-56, 57-90}` and **never blended**, because forecast
error grows with horizon by nature: one number averaged over 1..90 days answers a
question nobody asks.

### Forecast Value Added

`FVA = WMAPE(horizon seasonal naive) − WMAPE(model)`, in **percentage points**,
broken out by bucket. Positive means the model beat what a planner gets unaided.

Percentage points rather than a ratio: against a ~35% irreducible noise floor a
ratio compresses everything into a narrow band and a genuine four-point
improvement reads as a rounding error.

**Step 4's `SeasonalNaiveBaseline` cannot serve as the benchmark**, and reusing it
would have been the easy mistake. Its `lag_364_units` is measured at the *origin*,
so for a row at horizon `h` it reads sales from `364 + h` days before the target —
wrong weekday, wrong season, and progressively more wrong as `h` grows. Its
fallback chain also includes `lag_1_units`, the illegal nowcast feature.
`HorizonSeasonalNaive` reads units at `target_date − 364` instead.

### Prediction intervals

Two things Step 4's single scalar quantile could not do.

**Width must grow with horizon.** Mondrian conformal — one calibration per horizon
bucket, composing the existing `calibrate` — so each bucket keeps its own
finite-sample guarantee. A single global quantile over-covers short horizons,
under-covers long ones, and reports one blended figure that looks healthy while
being wrong at both ends.

**The horizon total gets its own calibration.** `total_predicted_units` is the
number an agent acts on, and its interval cannot be obtained by summing the daily
bounds: that assumes the daily errors move together perfectly, and if they were
independent the sum would be too wide by roughly √90 ≈ 9.5×. The truth is in
between and is not knowable a priori, so the aggregate is calibrated directly on
aggregate residuals.

Calibration origins are thinned to ≥7 days apart. Rows sharing an origin have
strongly correlated residuals, so counting them as independent inflates the
effective sample size and makes the finite-sample correction optimistic.

### Statistical models, fitted where they are correct

The cost was measured rather than assumed, and **the measurement is more modest
than the usual warning implies**. Weekly ETS fits in ~0.25s per series, so one
pass across all 6,128 product-store series is roughly **25 minutes** — about an
hour across three backtest folds. Expensive next to the global model's ~8 minutes
for the entire pipeline, but not infeasible, and saying otherwise would inflate a
real cost into a fictional impossibility.

**So the argument for aggregate grain is appropriateness, not affordability**, and
the numbers bear that out:

| Level | Series | Mean WMAPE |
|---|---|---|
| total | 1 | 15.8% |
| region | 5 | 16.0% |
| category | 7 | 16.4% |
| category × region | 35 | 17.9% |
| **product × store (25-series sample)** | 25 | **48.2%** |

At aggregate grain ETS is genuinely good — the series are smooth and high-volume,
and a level/trend/seasonal decomposition describes them well. At product-store
grain it scores 48.2% against the global model's 43.8%, on sparse counts sitting
against a 35% noise floor where it is largely fitting noise.

Fitting is weekly, not daily: daily annual seasonality needs 365 seasonal states
and will not converge on three years of history.

The bottom-grain sample is 25 series, so the claim is sized to it — enough to say
a classical univariate model is not dramatically better where the global model
operates, nowhere near enough to rank two models a few points apart.

### Hierarchy

Bottom-up product×store → category/region/total, **exactly coherent by
construction**. No reconciliation is applied.

The hierarchy table reports the *price* of that choice: independent errors average
out as you aggregate, so WMAPE falls sharply moving up. That answers a question a
planner actually asks — "should I trust the regional figure more than the SKU
figure?" — with a magnitude.

The aggregate ETS fits double as the independent top-level forecasts, so the §13
coherence check costs nothing extra. A systematic gap in one direction is evidence
of a level bias in the bottom model, which is plausible given the excluded
stockout targets.

**MinT reconciliation is rejected, not overlooked.** It needs a 6128×6128 error
covariance, and it buys accuracy only when you have genuinely independent
multi-level forecasts — which bottom-up deliberately does not produce.

---

## Results (800 series, 548,754 rows, seed 42)

| Model | WMAPE | MAE | RMSE | Bias | Mean FVA (pp) | Train | Predict |
|---|---|---|---|---|---|---|---|
| **xgboost** | **43.8%** | 24.86 | 53.13 | +8.4% | **+12.6** | 34.4s | 0.42s |
| lightgbm | 47.5% | 26.94 | 53.78 | +19.9% | +9.5 | 11.3s | 0.63s |
| horizon_seasonal_naive | 56.0% | 31.78 | 67.69 | +7.2% | — | 0.4s | 0.11s |
| horizon_naive | 78.1% | 44.33 | 96.93 | +36.1% | −19.8 | 0.4s | 0.09s |

**Selected: `xgboost`**, beating the seasonal naive by 12.2 points. Full run:
**475 seconds**.

### Forecast Value Added, per bucket (WMAPE percentage points)

| Model | h1-3 | h4-7 | h8-14 | h15-28 | h29-56 | h57-90 |
|---|---|---|---|---|---|---|
| xgboost | +15.3 | +11.3 | +12.5 | +13.2 | +12.0 | +11.0 |
| lightgbm | +13.1 | +9.4 | +10.4 | +10.4 | +7.8 | +5.7 |

**Positive at every horizon**, so there is no bucket where the seasonal naive
would be the better choice. LightGBM's advantage decays with horizon while
XGBoost's holds — the clearest difference between the two.

### Accuracy by aggregation level

| Level | Series | WMAPE |
|---|---|---|
| product × store | 45,295 | 43.6% |
| product | 24,752 | 35.7% |
| store | 4,857 | 18.7% |
| category | 854 | 15.0% |
| region | 610 | 11.8% |
| total | 122 | **9.6%** |

Error falls by a factor of 4.5 from SKU to total. That is the quantified answer
to "should I trust the regional number more than the SKU number?"

Backtest stability is strong: standard deviation across folds is 0.3–1.8
percentage points in every bucket.

### The horizon gradient is shallow, and why

Bucket WMAPE runs 43.6% (h1-3) → 44.7% (h57-90) — only **1.1 points over three
months**. That is flatter than a forecasting curve usually looks, so it is worth
stating plainly rather than presenting as a clean gradient.

The explanation is the noise floor. Step 4 measured the irreducible error on this
data at **35.0% WMAPE**, and the selected model sits at 43.8% — **1.25×** the
floor. That leaves only ~8.9 points of learnable signal *in total*, so the
degradation attributable to losing recent demand history is necessarily a
fraction of that. A steep curve is arithmetically unavailable here.

The gradient is therefore **weak evidence** about join correctness on this
dataset, and it is not what the design rests on. The leakage tests are: the
mutation test, the planted-bug test that proves it can fail, the per-row
arithmetic reconstruction, and train/serve equivalence.

## Behaviour beyond the end of the data

The calendar, promotion schedule and price plan **all end 2025-12-31**. A 90-day
forecast is therefore only fully informed from `as_of ≤ 2025-10-02`.

Requests past that are **refused** with a recoverable error naming the latest
workable as-of:

```
insufficient_data: a 90-day horizon from 2025-12-01 reaches 2026-03-01, but the
calendar, promotion schedule and price plan end 2025-12-31. Forecasting past that
would mean assuming no promotions are planned, which biases those days low. The
latest as-of that supports a 90-day horizon is 2025-10-02.
```

The alternative — assume no promotion runs and carry the last price forward —
produces a number that is systematically low and completely indistinguishable
from a real forecast. §45 asks for a credible system over an impressive one.

---

## Two parameter traps worth recording

**XGBoost's `min_child_weight` is not LightGBM's `min_child_samples`.** The first
sums *Hessians*; under `count:poisson` the Hessian is ≈ μ, so the parameter scales
with the target level. The second counts *rows*. Setting both to 50 gave XGBoost
roughly **38× less regularisation** on this data, and it scored 82.9% WMAPE at
+58% bias while fitting its own training fold perfectly. Scaled to `50 × mean(y)`
it scores 46.3% at −1.8%.

Reporting the unscaled run would have produced a confident and completely false
"XGBoost is much worse than LightGBM" line in the comparison table.

**Categorical dtypes must be frozen once, not inferred per frame.** `astype("category")`
takes its levels from whatever frame it is handed, so a season appearing only in
the calibration fold is either an unseen category (XGBoost raises) or a silently
different integer code (LightGBM does not, which is worse). The whole feature
schema is now pinned at fit time and coerced at predict time.

---

## Known limitations

| Limitation | Consequence |
|---|---|
| **Excluding stockout targets may bias the forecast low** | Removes the high-demand tail; measured against `latent_units` and reported rather than assumed away |
| **No inventory features** | This forecasts demand, not shipments. A replenishment planner wants the other model, which is not built |
| Competitor prices are frozen at the origin | `competitor_pricing` is OBSERVED, so a competitor move inside the horizon is invisible |
| No cannibalisation or halo | A promotion on one SKU distorts its substitutes' forecasts, unattributed |
| Conformal assumes exchangeability | A trend violates it mildly; coverage is measured per bucket so a shortfall surfaces |
| Cannot forecast past 2025-12-31 | A data limitation, not a design one — refused explicitly rather than faked |
| `--full` is a multi-hour run | The default samples 800 of 6,128 series; the sampled artifact is written to a separate directory so it cannot masquerade as the full model |

---

## Usage

```powershell
# Correctness check, under a minute
uv run python scripts/train_forecast.py --smoke --no-track

# The default run: 800 series, ~20 minutes
uv run python scripts/train_forecast.py --seed 42

# Every series. Hours, and declared as such.
uv run python scripts/train_forecast.py --full

uv run pytest tests/forecasting -v
```

```python
from app.schemas.domain import ForecastHorizon
from app.schemas.forecast import ForecastRequest
from app.services.container import Container

service = Container().forecasting_service
response = service.forecast(
    ForecastRequest(horizon=ForecastHorizon.D30, product_ids=["P00003"])
)
```

The service returns a structured error rather than raising for expected failures —
a missing model, an unknown product, a horizon past the calendar — each with a
code and a `recoverable` flag, because by Step 16 a supervisor agent has to
re-plan around them.

## Files

| Path | Contents |
|---|---|
| `ml/forecasting/config.py` | `ForecastConfig`, loaded from `configs/models/forecasting.yaml` |
| `ml/forecasting/sampling.py` | True pair sampling — see the Step 4 defect it exists to avoid |
| `ml/forecasting/dataset.py` | The horizon self-join and the inference scaffold |
| `ml/forecasting/split.py` | Origin-based split with the embargo |
| `ml/forecasting/baselines.py` | `HorizonNaive`, `HorizonSeasonalNaive` |
| `ml/forecasting/xgboost_model.py` | XGBoost candidate, with the Hessian-scaling fix |
| `ml/forecasting/conformal.py` | Per-bucket and aggregate calibration |
| `ml/forecasting/train.py` | Fit, calibrate, score; the frozen feature schema |
| `ml/forecasting/evaluate.py` | Bucket metrics, FVA, hierarchy, revenue impact |
| `ml/forecasting/backtest.py` | Expanding-origin walk-forward |
| `ml/forecasting/predict.py` | As-of validation and the fallback chain |
| `ml/forecasting/model.py` | `FittedForecastModel`, implementing the Step 1 interface |
| `ml/forecasting/statistical.py` | Weekly ETS at aggregate grain |
| `ml/forecasting/explain.py` | Permutation importance by horizon bucket |
| `ml/forecasting/monitoring.py` | Drift, forecast-vs-actual, prediction distribution |
| `app/services/forecast_service.py` | Service seam, assumptions and warnings |
| `app/tools/forecasting_tool.py` | The agent-facing contract |
| `tests/forecasting/` | Dataset, split, leakage, service and tool-contract tests |
