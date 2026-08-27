# Forecasting evaluation

How the model is measured, why those measures and not others, and how the winner
is chosen.

---

## 1. The number that makes every other number readable

Step 4 measured the **irreducible noise floor** on this dataset at **35.0%
WMAPE**. That is the score a model knowing the *true conditional mean* would
still achieve, because demand is drawn from an over-dispersed negative binomial
and that variance is not learnable by anything.

Every accuracy figure here should be read against it:

| | |
|---|---|
| Selected model (XGBoost) | **43.8%** WMAPE |
| Irreducible floor | 35.0% |
| **Ratio** | **1.25×** |
| Learnable signal remaining | ~8.9 points |

Without that context, 43.8% reads as poor. With it, the model has captured most
of what is capturable — and a model scoring 20% would be **impossible** on honest
features and should be treated as evidence of leakage rather than skill.

This is also why the hyperparameter search found nothing (§6): there is very
little left for hyperparameters to win.

## 2. Metrics, and why each one is here

| Metric | Definition | Why |
|---|---|---|
| **WMAPE** | `Σ\|y − ŷ\| / Σy` | **The headline.** Volume-weighted, so a 50% error on a SKU selling 10,000 units matters more than a 50% error on one selling three. Plain MAPE calls those identical |
| **MAE** | mean absolute error | Units, directly interpretable — "we are out by 24 units a day on average" |
| **RMSE** | root mean squared error | Punishes large misses; the gap between RMSE and MAE says how fat the error tail is |
| **Bias** | `Σ(ŷ − y) / Σy`, signed | **Matters more than dispersion.** Random error averages out over a planning period; a consistent skew does not — it compounds into every inventory decision, always in the same direction |
| **MASE** | `MAE(model) / MAE(seasonal naive, in-sample)` | Scale-free, so it is comparable across series with wildly different volumes. Below 1 means the model beats doing nothing |
| **MAPE** | mean absolute percentage error | Reported **only over non-zero actuals**, with the excluded count stated |
| **FVA** | `WMAPE(benchmark) − WMAPE(model)`, in points | Is the model worth having at all? |

### Why not MAPE as the headline

MAPE is undefined at zero and unstable near it. Roughly 8.5% of rows in this
panel are zero-unit days, and many more are single digits — exactly the rows where
the denominator is smallest and the ratio explodes. A MAPE computed over
everything would be dominated by the least commercially important rows.

So MAPE is computed only where it is defined, and **the number of excluded rows
is reported alongside it**. Silently dropping them would overstate accuracy; a
silent `inf` would be worse.

### Why MASE's denominator comes from the training fold

MASE scales the model's error by the error a naive forecaster makes *on data the
model was fitted on*. Taking the denominator from the evaluation fold instead
makes the metric partly self-referential — both numerator and denominator would
move with the same held-out noise, and a model could improve its MASE by getting
worse in a period where the naive got worse faster.

The seasonal period is **7 days**, not 364. Demand here is far more weekly than
annual, so one-week-ago is the natural "no effort" comparison. A yearly scale
would make almost any model look excellent, which is the flattering-benchmark
failure MASE exists to avoid.

## 3. Reported per horizon bucket, never blended

Buckets: `h1-3`, `h4-7`, `h8-14`, `h15-28`, `h29-56`, `h57-90`.

Forecast error grows with horizon by nature, so a single WMAPE averaged over 1–90
days answers a question nobody asks — *"how wrong are we, somewhere between
tomorrow and three months?"* Nobody forecasts that.

Measured for the selected model:

| Bucket | WMAPE | Bias |
|---|---|---|
| h1-3 | 43.6% | +9.8% |
| h4-7 | 42.2% | +6.9% |
| h8-14 | 43.4% | +7.2% |
| h15-28 | 43.0% | +7.6% |
| h29-56 | 44.2% | +8.6% |
| h57-90 | 44.7% | +9.5% |

### The gradient is shallow, and that is reported rather than smoothed

1.1 points across three months. Flatter than a forecasting curve usually looks,
and the explanation is §1: at 1.25× the noise floor there are only ~9 points of
learnable signal in total, so the degradation attributable to losing recent
demand history is a fraction of that. A steep curve is **arithmetically
unavailable** here.

The consequence is that the gradient is **weak evidence** about join correctness
on this dataset. The leakage tests carry that argument instead (§7), and the
behavioural test asserts only that long-horizon error has not *collapsed* — which
is what a real leak produces and which the data can reliably support.

## 4. Forecast Value Added

`FVA = WMAPE(horizon seasonal naive) − WMAPE(model)`, in percentage points.

| Model | h1-3 | h4-7 | h8-14 | h15-28 | h29-56 | h57-90 |
|---|---|---|---|---|---|---|
| **xgboost** | +15.3 | +11.3 | +12.5 | +13.2 | +12.0 | +11.0 |
| lightgbm | +13.1 | +9.4 | +10.4 | +10.4 | +7.8 | +5.7 |

**Positive at every horizon**, so there is no bucket where the naive benchmark
would be the better choice. Note the difference in shape: LightGBM's advantage
decays with horizon (+13.1 → +5.7) while XGBoost's holds (+15.3 → +11.0). That is
the clearest distinction between the two, and it is invisible in a blended figure.

Reported in **points, not as a ratio**: against a 35% floor a ratio compresses
every result into a narrow band and makes a genuine four-point improvement read as
a rounding error.

### The benchmark had to be corrected first

Step 4's `SeasonalNaiveBaseline` **cannot** serve here, and reusing it would have
been the easy mistake. Its `lag_364_units` is measured at the *origin*, so for a
row at horizon *h* it reads sales from `364 + h` days before the target — wrong
weekday, wrong point in the season, progressively worse as *h* grows. Its fallback
chain also includes `lag_1_units`, the illegal nowcast feature.

`HorizonSeasonalNaive` reads units at `target_date − 364` instead. 364 not 365,
because it is a multiple of 7 and therefore the same weekday.

A weak benchmark would make every model look good and turn this whole section into
flattery.

## 5. Accuracy by aggregation level

| Level | Series | WMAPE |
|---|---|---|
| product × store | 45,295 | 43.6% |
| product | 24,752 | 35.7% |
| store | 4,857 | 18.7% |
| category | 854 | 15.0% |
| region | 610 | 11.8% |
| **total** | 122 | **9.6%** |

Error falls by a factor of **4.5** from SKU to total. Bottom-up aggregation is
*exactly* coherent by construction — the regional number is always the sum of its
store numbers — so nothing is reconciled.

What this table quantifies is the price of that choice: independent errors average
out as you aggregate. It answers the question a planner actually asks — *should I
trust the regional figure more than the SKU figure?* — with a magnitude rather
than a shrug.

## 6. Model selection

**Criterion, in order:**

1. **Accuracy** on the test fold, by WMAPE against held-out outcomes.
2. **Simplicity on a near-tie.** If the seasonal naive lands within two
   percentage points of the best model, it wins. A benchmark that holds its own
   is telling you the signal is simple, and two points do not justify the
   training cost and opacity.
3. **A leakage warning is emitted, not suppressed**, if error does not grow with
   horizon for the selected model.

Measured comparison (800 series, 548,754 rows, seed 42):

| Model | WMAPE | MAE | RMSE | MASE | Bias | Mean FVA | Train | Predict |
|---|---|---|---|---|---|---|---|---|
| **xgboost** | **43.8%** | 24.86 | 53.13 | 0.68 | +8.4% | **+12.6 pp** | 34.4s | 0.42s |
| lightgbm | 47.5% | 26.94 | 53.78 | 0.74 | +19.9% | +9.5 pp | 11.3s | 0.63s |
| horizon_seasonal_naive | 56.0% | 31.78 | 67.69 | 0.80 | +7.2% | — | 0.4s | 0.11s |
| horizon_naive | 78.1% | 44.33 | 96.93 | 1.05 | +36.1% | −19.8 pp | 0.4s | 0.09s |

**XGBoost is selected**, beating the seasonal naive by 12.2 points — comfortably
outside the simplicity tolerance.

Note `horizon_naive` at MASE **1.05**: worse than a weekly seasonal naive, which
is the correct reading of a model that carries the last value forward regardless
of weekday.

### The XGBoost result had to be earned twice

The first run scored **82.9% WMAPE at +58% bias** and looked far worse than
LightGBM. The cause was parameter semantics, not modelling: `min_child_weight`
sums *Hessians*, and under `count:poisson` the Hessian is approximately μ, so the
parameter scales with the target level. LightGBM's `min_child_samples` counts
*rows*. Setting both to 50 gave XGBoost roughly **38× less regularisation**.

Scaled to `50 × mean(y)` it scores 43.8%. **Reporting the unscaled run would have
put a confident and completely false line in this table.**

## 7. Hyperparameter search

Twenty seeded trials, random search, scored on the **validation fold only** —
never test, or every number above would be a self-report.

Measured outcome:

> 19 trials in 207s. Best WMAPE 42.34% vs default 42.22% (−0.12 points) —
> **within fold-to-fold noise; keep the defaults.**

**The search found nothing, and the defaults were retained.** That is the expected
result at 1.25× the noise floor, and it is reported rather than re-run until a
number moved. Backtest standard deviation on this data runs 0.3–1.8 points, so
anything under half a point is indistinguishable from which fold you happened to
look at — which is why `best_params()` returns `{}` below that threshold.

## 8. Validation design

**Walk-forward over origins**, expanding, with an **embargo of `max_horizon`
days** between every fold.

The embargo is the part Step 4's split did not need. Here a training origin
sitting just before a fold boundary has its *target* inside the evaluation window
— the model is fitted on the very outcomes it is about to be scored on, the test
metric improves, and nothing raises. The gap costs 90 days of origins at every
boundary, and that cost is the point.

Everything splits on the **origin date**, never the row: rows sharing an origin
see identical history and differ only in horizon step, so splitting by row would
scatter one origin's rows across train and test.

Backtest stability across folds: **0.3–1.8 percentage points** standard deviation
in every bucket — steady enough that a single headline number is not misleading.

## 9. Prediction intervals

Split conformal, calibrated **per horizon bucket** (Mondrian), plus a
**separately calibrated quantile for the horizon total**.

Two things a single scalar quantile could not do:

- **Width must grow with horizon.** One global quantile produces a width
  proportional to the prediction alone, which barely widens with *h*. It
  over-covers short horizons, under-covers long ones, and reports one blended
  coverage figure that looks healthy while being wrong at both ends.
- **The total needs its own calibration.** Summing daily bounds assumes the daily
  errors move together perfectly; if they were independent the sum would be too
  wide by roughly √90 ≈ 9.5×. The truth is in between and is not knowable a
  priori, so the aggregate is calibrated directly on aggregate residuals.

**Coverage is measured on held-out data and reported whatever it is.** A 90%
interval covering 71% is a finding, not a detail to smooth over — and that
discipline is what separates this from a fabricated `confidence: 0.92`.

Calibration origins are thinned to ≥7 days apart: adjacent horizon steps from one
origin share an information set, so their residuals are correlated and the
effective sample size is far below the row count.

## 10. Segment and bias analysis

Reported by category, brand, region, channel, store type, season, promotion state
and holiday state — minimum 30 rows per group.

Bias gets particular attention because of how it propagates:

| Where bias lands | Consequence |
|---|---|
| Inventory | Over-forecast → overstock and markdown; under-forecast → lost sales |
| Revenue planning | A consistent skew compounds across a quarter |
| Promotion optimisation | A low baseline manufactures uplift on every promotion |
| Price optimisation | Biased demand at a given price biases the optimal price |

**One bias mechanism is known and structural**: rows whose target fell on a
stockout are excluded from training, and stockouts here are *endogenous* — they
happen because demand spiked, with latent demand during one running ~1.57× normal.
Dropping them removes part of the high-demand tail.

Section 25 of the brief says not to correct bias without investigating the cause.
Applying a flat correction factor would paper over that mechanism with a number
nobody can justify. The honest response, taken here, is to state the mechanism and
its expected direction.

## 11. Reproducing these numbers

```powershell
uv run python scripts/train_forecast.py --seed 42          # ~8 min, the deliverable
uv run python scripts/train_forecast.py --seed 42 --tune   # + ~4 min for the search
uv run ari evaluate-forecast                               # print the stored report
uv run ari forecast-quality                                # the input-side checks
```

Every run records a **config fingerprint** — one hash over the entire
configuration — in MLflow params. Two runs sharing it used the same setup; two that
differ are not comparable, and the difference is discoverable rather than argued
about.
