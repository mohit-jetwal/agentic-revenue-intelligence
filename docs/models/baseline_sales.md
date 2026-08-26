# Baseline Sales Model

> Stage 1, Step 4. The first analytical capability in the platform, and the one
> every later step measures against.

## What question this answers

**"What would sales have been under normal conditions?"**

Normal means: no promotion running, and stock available. That is a
*counterfactual*, not a forecast, and the distinction carries the whole step.

| | Forecast | Baseline |
|---|---|---|
| Question | What *will* happen? | What *would* have happened? |
| Includes planned promotions | Yes | **No** |
| Includes known stockouts | Yes | **No** |
| Evaluated against | What actually happened | Nothing directly observable |

Conflating the two is the most common way uplift analysis goes wrong. A forecast
that includes the promotion, used as a baseline, measures the promotion against
itself and reports approximately zero uplift.

The last row is why this step is unusually hard to validate: the quantity being
estimated is, by construction, one that never occurred. On promotional days the
true baseline was never observed - the promotion ran. This project can check it
anyway, for reasons covered under [Validation against ground truth](#validation-against-ground-truth).

## Why it matters more than its accuracy suggests

Three later steps consume this model:

- **Promotion uplift (Step 5)** - uplift *is* actual minus baseline.
- **Root cause (Step 17)** - "sales are down" is only meaningful relative to it.
- **Scenario simulation** - every projection starts from it.

So a bias here does not fail loudly. It propagates as a consistent, plausible
distortion into every downstream number, in the same direction, and each
consumer amplifies it. A baseline biased 5% low manufactures 5% uplift on every
promotion ever measured - and because the news is good, nobody investigates.

This is why **bias is treated as more important than error** throughout. Random
error averages out when a campaign is aggregated; bias does not.

## The two decisions that carry the step

### 1. Promotional contamination

Training data contains promotional periods. Their sales include promotional lift,
which is precisely what the baseline must exclude. Two defensible approaches
exist, and both have a real bias:

**Approach C - train on non-promotional rows only.**
Clean counterfactual semantics: the model has never seen a promotion, so it
cannot embed one. But it inherits a **selection bias**. Step 2 generated
promotions with `targeting_strength: 0.40`, weighting them toward seasonal
peaks. Dropping promotional rows therefore under-represents high season, the
baseline underestimates peaks, and **uplift is overstated**.

**Approach B - train on everything, with promotion features as controls, then
predict with those features zeroed.**
Uses all the data and avoids that selection bias. But it asks a gradient-boosted
model to extrapolate to a feature combination it may rarely have seen -
predicting `promotion_flag = 0` for a product-store that was almost always
promoted. Tree models are weakest exactly there, since they cannot extrapolate
beyond observed leaf regions.

**Neither argument settles it.** Which bias dominates is an empirical fact about
a given dataset, so the pipeline **builds both and selects on measured evidence**
against true demand. The comparison is a deliverable of this step in its own
right, not scaffolding.

### 2. Stockout censoring

Observed sales during a stockout measure **availability, not demand**. A model
trained on them learns that a supply failure predicts low demand - exactly
backwards, and the inversion that would make Step 17 recommend a price cut to
fix a warehouse problem.

Three mechanisms, addressing different halves of the problem:

- **Stockout rows are excluded from training.** The target is corrupted on those
  rows; it records what was available to sell.
- **Censored values are not lagged forward.** Step 3's
  `features.engineering.demand.mask_censored` exists for this. Without it, a
  stockout depresses the next four weeks of lag features and the model learns
  the supply failure indirectly, through the back door.
- **No supply-side column is a feature** (`SUPPLY_FEATURES`). See below - this
  one was not in the original design and was added after the ground truth caught
  it.

#### Why excluding stockout rows is not enough

The first training run on the full panel disqualified **both** LightGBM
candidates. The feature importances showed why: `closing_inventory_lag_1` was
the single most important feature in the model, with `opening_inventory` and
`inventory_available` also in the top five.

With inventory available as a predictor, the model learns *"low stock predicts
low sales"* - which is true, and is exactly the censoring relationship the step
exists to avoid. Excluding stockout rows does not prevent it: the relationship
is learned from the many partially-depleted rows just below the stockout
threshold and then extrapolated to zero stock.

The measured effect, from the same panel before and after the fix:

| | stockout lift (÷ observed) | ÷ latent (correct ≈ 0.64) | outcome |
|---|---|---|---|
| LightGBM, inventory features present | 1.12 | 0.30 | **disqualified** |
| LightGBM, supply features excluded | 2.48 | **0.68** | selected |

Accuracy barely moved - 40.4% to 40.2% WMAPE. That is the important part: the
inventory columns were contributing censoring signal rather than demand signal,
so removing them cost essentially nothing and fixed the diagnostic outright.

There is also a conceptual argument that should have caught this without the
experiment. The baseline is *defined* as demand with stock available.
Conditioning it on inventory answers a different question - "what would sell
given this stock level" - and that question is circular for every use the model
has.

**Accepted cost:** a recent stockout can genuinely depress future demand, as
customers switch brand or store. The model can no longer see that effect. It is
given up deliberately, because here it is inseparable from the censoring
artefact and far smaller than it.

## Target and objective

Target is observed `units` on clean rows. LightGBM uses
**`objective="poisson"`**, deliberately, rather than fitting on `log1p(units)`
and back-transforming with `expm1`.

The log-transform route introduces **retransformation bias**. By Jensen's
inequality, `E[exp(X)] != exp(E[X])`, so the back-transformed mean is
systematically *low* - the exact failure mode described above, arrived at
through a routine-looking preprocessing choice. Poisson models the count
directly on its natural scale, and its log link handles the skew without the
bias.

## Candidate models

| Model | Role |
|---|---|
| `SeasonalNaiveBaseline` | The benchmark. `lag_364` (same weekday last year) blended with `rolling_28`. If the others cannot beat it, complexity is not earning its place. |
| `RidgeBaseline` | Linear and interpretable. Its gap to LightGBM tells you whether the relationship is mostly linear. |
| `LightGBMBaseline` | Poisson objective, native categoricals, early stopping on a temporal validation fold. |

Each is trained under **both** promotion approaches, giving six candidates.

### Selection rules, in order

1. **Correctness before accuracy.** A candidate that failed the stockout check is
   disqualified regardless of headline accuracy. An accurate model measuring the
   wrong quantity is worse than a less accurate one measuring the right quantity
   - it is wrong *and* trusted.
2. **Accuracy** against true demand where available, observed sales otherwise.
3. **Simplicity on a near-tie.** If the seasonal naive is within 2 percentage
   points of the best model, it wins. Two points do not justify fifty times the
   training cost and a model nobody can explain in a planning meeting.
4. **Stability** breaks remaining ties. A model whose accuracy swings between
   quarters cannot sit behind a recommendation.

## Validation

### Temporal splits

Four chronological folds, never random. A random split lets the model see the
future while predicting the past, which inflates every metric.

```
train ──────────────► calibration ──► validation ──► test
                          60d            90d          120d
```

Calibration sits *before* validation on purpose: validation drives early
stopping, so it must be the fold closest to test for the stopping point to
reflect the most recent regime. Calibration is a separate fold because conformal
intervals calibrated on the test set would be reporting the test set's own
quantile back to itself.

### Metrics

**WMAPE is the headline** because it is volume-weighted. A 50% error on a hero
SKU selling 10,000 units matters more than a 50% error on a tail SKU selling
three; plain MAPE calls them identical.

MAPE is still reported, but **only over non-zero actuals, with the excluded
count alongside**. A zero actual makes the ratio infinite, and an `inf` quietly
poisoning a mean is worse than an honestly absent number.

Bias is reported signed, for the reasons above.

### The noise floor

A bare WMAPE is uninterpretable. Step 2 stores `mean_demand` - the *true*
conditional mean - alongside the realised count drawn from it, so the error
between those two is pure negative-binomial sampling noise that no model can
reduce.

On the dev dataset that floor is **35.0% WMAPE**. A model scoring 40% is
therefore at ~1.15x the theoretical best, not "inaccurate". A model scoring 20%
would be *impossible* and should be read as evidence of leakage.

Every reported accuracy figure is accompanied by this ratio. Without it, a
near-optimal model reads as poor and a leaking one reads as excellent.

### Validation against ground truth

Step 2 stored `data/local/ground_truth/latent_demand/`, holding `latent_units`
(true uncensored demand) alongside `observed_units`. Almost no real project has
this. It permits three checks that are normally impossible:

| Row type | Ground-truth relationship | What it proves |
|---|---|---|
| Clean | `latent ≈ observed` = true baseline | Baseline accuracy, directly |
| **Stockout** | `latent >> observed` | The model learned *demand*, not censored sales |
| Promotional | `latent = baseline x e^(promo_lift)` | Gap direction scales with discount depth |

**The stockout row is the headline.** Normally a model that learned censoring is
undetectable - it scores *better* against observed sales than the correct model,
because it reproduces the censoring too.

#### Reading the stockout diagnostic correctly

The obvious test is wrong, and it is worth stating why.

Comparing the baseline to *latent* demand during stockouts looks like the
natural check, but stockouts in this data are **endogenous**: they happen
*because* demand spiked. Measured on the generated dataset, latent demand during
a stockout runs **1.57x** the normal level. A baseline correctly predicting
normal demand therefore lands near 0.64 of stockout-period latent demand, and a
threshold expecting ~1.0 would disqualify a perfectly good model. An earlier
revision of this step did exactly that and rejected all six candidates.

The criterion used instead is **predicted / observed** on stockout rows, which
has no such confound. Observed sales are censored, so a model that learned
demand must sit clearly above them; one that learned the censoring lands near
1.0 by construction. The floor is set at **1.20**
(`comparison.STOCKOUT_LIFT_FLOOR`) - loose enough that a mediocre model still
passes, tight enough that a censoring-learner cannot.

The ratio to latent demand is still reported, for transparency, but is not used
for selection.

#### An honest limit

Exact baseline reconstruction on promotional rows is **not** possible from the
stored ground truth. `promotion_uplift.json` holds the response curve `a` and
`b`, but the store-level `_promo_responsiveness` multiplier is latent and
deliberately unpublished. Promotional validation is therefore **directional**
(sign, and monotonicity in discount depth) rather than point-accurate. Stated
rather than glossed.

## Results (dev profile, seed 42, 5.06 M panel rows)

Six candidates, ranked on WMAPE against true demand on clean test rows:

| Model | Approach | Latent WMAPE | Bias | Stockout lift | ÷ latent | Backtest | Coverage |
|---|---|---|---|---|---|---|---|
| **lightgbm** | **exclude** | **40.4%** | +6.7% | 2.48 | 0.68 | 39.0% ±0.7% | 92.0% |
| lightgbm | control | 40.6% | +7.0% | 2.72 | 0.74 | 39.3% ±1.4% | 92.0% |
| ridge | exclude | 43.7% | **−0.5%** | 2.86 | 0.78 | 43.1% ±1.1% | 87.9% |
| ridge | control | 45.0% | −3.0% | 2.88 | 0.78 | 44.3% ±0.9% | 88.7% |
| seasonal_naive | either | 54.3% | +17.4% | 2.32 | 0.63 | 52.7% ±1.3% | 88.6% |

**Selected: `lightgbm__exclude`.**

- **40.4% against a 35.0% noise floor = 1.15×.** Most of the remaining error is
  irreducible negative-binomial noise, not unexploited signal.
- **Every candidate now passes the stockout check** (2.32–2.88 versus observed
  sales). The `÷ latent` column is the sanity check: the theoretically correct
  value is ~0.64, and the seasonal naive lands at 0.63 almost exactly.
- **Backtest is stable** across four expanding quarterly folds at ±0.7%.
- **Approach C ("exclude") wins, but by 0.2 points** — 40.4% versus 40.6%. That
  is far too narrow to call a general result. It says the two biases roughly
  cancel on *this* dataset, not that Approach C is better.
- Top features are all demand-side: rolling 56/28/14-day means, festival flag,
  price versus rolling average, day of week.

### An unresolved tension worth naming

This document argues that **bias matters more than error** for a baseline, and
that is genuine — random error averages out over a campaign, while a consistent
skew does not. But the selection rule ranks on **WMAPE**, and on this data the
two disagree:

| | Latent WMAPE | Bias |
|---|---|---|
| lightgbm exclude | 40.4% (better) | **+6.7%** |
| ridge exclude | 43.7% | **−0.5%** (better) |

LightGBM is 3.3 points more accurate; Ridge is essentially unbiased. Since the
baseline over-predicts by 6.7%, uplift measured against it will be
*understated* by roughly that much — the safer direction of the two (it hides
real uplift rather than inventing it), but not nothing.

The rule as specified ranks on accuracy, so LightGBM is selected and that is
what ships. Flagged here rather than quietly resolved, because adding a bias
term to the selection criterion is a real decision with a real trade-off, and it
belongs to whoever owns the uplift numbers in Step 5.

## Prediction intervals

Split conformal, calibrated on a dedicated fold. Distribution-free, with a
finite-sample coverage guarantee.

The property that matters is not the guarantee but the practice: **achieved
coverage is measured on test data and reported whatever it is**. A 90% interval
that covers 71% is a finding worth surfacing. This is what separates a real
interval from a fabricated `confidence: 0.92`, which costs nothing to emit and
means nothing.

`is_significant` then means something precise: **the gap falls outside the
prediction interval**, so it is larger than the model's normal error. That is
the test Step 17 needs before claiming a decline is real rather than noise.

## Cold start

Products with fewer than 60 days of history fall back to a category x
store-channel mean, and the row is flagged `fallback_used`.

Flagging is not optional. A caller must be able to distinguish an estimate from
a guess; an unflagged fallback is a guess wearing an estimate's clothes.

## Feature importance

LightGBM gain importance plus **permutation importance** on a validation sample.

SHAP is not installed and is not planned. At panel scale it costs considerably
more than it adds for the question "what drives baseline demand", and
permutation importance answers that directly by measuring the WMAPE degradation
when a feature is shuffled. Recorded as a decision, not an omission.

## Known limitations

- **Conformal intervals assume exchangeability**, which a trend violates mildly.
  Coverage is measured rather than assumed, so a shortfall surfaces instead of
  hiding.
- **Ridge fits on a 750,000-row subsample.** One-hot encoding produces a dense
  design matrix; fitting on the full panel needs >6 GB and pushes a 16 GB
  machine into swap. A memory decision, not an accuracy one - coefficient
  standard errors at that sample size are already far tighter than the model's
  specification error.
- **Promotional baseline validation is directional only** (see above).
- **The baseline does not model cannibalisation or halo.** A promotion on one
  SKU depresses the baseline-relative performance of its substitutes, and this
  model attributes that to the substitute rather than the promotion.
- **A gap is not automatically causal uplift.** Attributing it to a promotion
  requires causal assumptions this model does not test. Carried explicitly in
  every response's `assumptions`.

## Usage

```powershell
# Train, compare, select, persist
uv run python scripts/train_baseline.py --profile dev --seed 42

# Faster iteration on a sampled panel
uv run python scripts/train_baseline.py --sample-pairs 200 --no-backtest --no-track

# Tests
uv run pytest tests/models -v
```

```python
from app.services.container import Container
from app.schemas.baseline import BaselineRequest

service = Container().baseline_service
response = service.predict(
    BaselineRequest(start_date=date(2025, 10, 1), end_date=date(2025, 10, 14))
)
```

The service returns a structured error rather than raising for expected failures
- a missing model, an empty slice, a reversed date range - each with an error
code and a `recoverable` flag. By Step 16 a supervisor agent needs to re-plan
around failures, and it can only do that with a failure it can read.

## Files

| Path | Contents |
|---|---|
| `ml/baseline/models.py` | The three estimators |
| `ml/baseline/training.py` | Splits, row filtering, training loop, backtest |
| `ml/baseline/evaluation.py` | Metrics, segments, error analysis, noise floor |
| `ml/baseline/conformal.py` | Split-conformal intervals and coverage measurement |
| `ml/baseline/comparison.py` | Candidate comparison and selection rules |
| `ml/baseline/tracking.py` | MLflow experiment, params, metrics, registration |
| `ml/baseline/model.py` | `FittedBaselineModel`, implementing the Step 1 ABC |
| `ml/baseline/pipeline.py` | End-to-end orchestration |
| `app/services/baseline_service.py` | Service seam, assumptions and warnings |
| `tests/models/` | Unit, ground-truth, leakage and business-scenario tests |
