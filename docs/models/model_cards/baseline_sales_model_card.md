# Model Card — Baseline Sales Model

| | |
|---|---|
| **Name** | `baseline_sales` |
| **Version** | v1.0 |
| **Stage** | Stage 1, Step 4 |
| **Type** | Counterfactual demand estimation (regression on counts) |
| **Status** | Local MVP — not production |
| **Owner** | Agentic Revenue Intelligence Platform |
| **Tracking** | MLflow experiment `baseline_sales` |

---

## Intended use

### What it is for

Estimating **units that would have sold under normal trading conditions** — no
promotion running, stock available — for a product/store/date slice.

Downstream consumers:

- Promotion uplift measurement (Step 5): uplift = actual − baseline
- Root-cause analysis (Step 17): "down versus what?"
- Scenario simulation: the starting point for every projection

### What it is *not* for

- **Forecasting.** It deliberately excludes planned promotions and known
  stockouts. Used as a forecast it will under-predict any period with a
  promotion scheduled.
- **Individual-row decisions.** Error at row level is dominated by irreducible
  count noise. Aggregate before acting.
- **Causal attribution on its own.** A gap between actual and baseline is a
  *difference*. Calling it uplift requires causal assumptions this model does not
  test.
- **Products with under 60 days of history.** These fall back to a
  category × channel mean and are flagged `fallback_used`.
- **Anything outside the trained catalogue and date range.**

---

## Training data

| | |
|---|---|
| Source | Step 2 synthetic CPG/Retail dataset (`dev` profile) |
| Grain | product × store × day |
| Period | 2023-01-01 → 2025-12-31 |
| Panel rows | ~5.06 M |
| Rows used for training | ~4.02 M (Approach C) / ~4.82 M (Approach B) |
| Features | 91 (Approach C) / 100 (Approach B) |

### Rows deliberately excluded

| Exclusion | Rows | Why |
|---|---|---|
| Stockouts | ~237,000 | Target is censored — records availability, not demand |
| Promotional (Approach C only) | ~800,000 | Target contains promotional lift, which is not baseline |

### Not used as features

Identifiers (`date`, `product_id`, `store_id`), the target, and every
target-derived column (`revenue`, `cost`, `gross_profit`, `units_uncensored`,
`sold_units`). `revenue = units × price` recovers the target exactly — the most
innocent-looking leak in the column list and the most damaging.

Free-text and high-cardinality labels (`product_name`, `city`, …) are dropped as
well: no signal a tree can use, and they bloat any encoder fitted downstream.

**No personal or customer-level data is used.** The panel is aggregate
product/store/day throughout.

---

## Model

| | |
|---|---|
| Selected estimator | LightGBM |
| Objective | **Poisson** |
| Promotion handling | Selected empirically between Approach B and C |
| Categoricals | Native LightGBM handling |
| Early stopping | On a chronological validation fold |
| Intervals | Split conformal, 90% nominal |

**Why Poisson and not `log1p`/`expm1`:** the log-transform route introduces
retransformation bias — by Jensen's inequality `E[exp(X)] ≠ exp(E[X])`, so the
back-transformed mean is systematically low. A baseline biased low manufactures
uplift on every promotion measured against it.

### Candidates compared

Three estimators × two promotion approaches = six candidates, each scored against
true demand. Selection rules, in order: correctness (stockout check) → accuracy →
simplicity on a near-tie → stability. See `docs/models/baseline_sales.md`.

---

## Evaluation

### Splits

Chronological, never random: train → calibration (60d) → validation (90d) →
test (120d). Calibration precedes validation so that early stopping reflects the
regime closest to test; it is a separate fold so conformal coverage is not
measured on the data that set the interval width.

Also backtested over four expanding quarterly windows to show whether accuracy is
stable rather than a lucky split.

### Metrics

**WMAPE is the headline** — volume-weighted, so a large error on a hero SKU is
not hidden by small errors on a long tail. MAPE is reported only over non-zero
actuals, with the excluded count alongside.

### Interpreting the accuracy figure

The **noise floor is 35.0% WMAPE** on this dataset: that is the score a model
knowing the *true* conditional mean would still achieve against realised counts,
because demand is drawn from an over-dispersed negative binomial.

| | |
|---|---|
| Selected model, clean rows vs true demand | ~40% WMAPE |
| Noise floor | 35.0% WMAPE |
| **Ratio** | **~1.13×** |

Read the ratio, not the absolute number. A score of 20% would be impossible on
honest features and should be treated as evidence of leakage.

### Ground-truth validation

Step 2 retains true uncensored demand, permitting checks a real project cannot
run:

| Check | Result |
|---|---|
| Baseline vs censored sales during stockouts | **1.38–2.32×** — sees through the censoring |
| Prediction-interval coverage (90% nominal) | **~91%** measured on test data — calibrated |
| Bias on clean rows | Small and reported signed |
| Promotional gap direction | Positive, scaling with discount depth |

**On the stockout ratio:** the criterion is predicted ÷ *observed*, not ÷ latent.
Stockouts here are endogenous — they occur *because* demand spiked, and latent
demand during one runs 1.57× normal — so a correct baseline is legitimately
*below* stockout-period latent demand. An earlier revision used the latent ratio
with a 0.7 floor and disqualified all six candidates. Documented because the
mistake is easy to repeat.

---

## Limitations and risks

| Limitation | Consequence |
|---|---|
| Promotional baseline cannot be point-validated | `_promo_responsiveness` is latent and unpublished; promotional validation is directional only |
| Conformal assumes exchangeability | A trend violates it mildly; coverage is *measured*, so a shortfall surfaces |
| No cannibalisation or halo modelling | A promotion on one SKU distorts its substitutes' baseline-relative performance |
| Ridge fits on a 750,000-row subsample | Memory constraint, not accuracy — dense one-hot design exceeds 6 GB on the full panel |
| Cold-start rows use a category × channel mean | Materially less accurate; always flagged `fallback_used` |
| Trained on synthetic data | Real retail has structure this simulator does not reproduce |

### Failure modes to watch

- **Silent bias** is the dominant risk. It does not fail loudly; it propagates a
  consistent distortion into every downstream number, and because a low baseline
  produces flattering uplift, nobody investigates.
- **Learning the censoring.** Guarded by the stockout check, which disqualifies a
  candidate regardless of headline accuracy.
- **Distribution shift.** Coverage degrades first; it is measured, not assumed.

---

## Ethical and safety considerations

- **No personal data.** Aggregate product/store/day only.
- **No fabricated numbers.** Every figure originates from a deterministic model.
  The LLM layer added in later steps interprets these outputs; it never invents
  them.
- **Uncertainty is surfaced, not hidden.** Intervals are measured and reported
  even when coverage falls short of nominal.
- **Assumptions travel with the answer.** Every response carries the
  approach used, the cold-start share, and the caveat that a gap is not
  automatically causal.
- **Commercial impact.** Outputs influence pricing and promotion decisions
  affecting revenue. Aggregate before acting; treat row-level figures as
  indicative.

---

## Maintenance

| | |
|---|---|
| Retrain trigger | New dataset version, or backtest instability |
| Monitor | Bias on clean rows, stockout ratio, conformal coverage |
| Artifacts | `data/local/models/baseline/` (joblib + JSON sidecar) |
| Reproduce | `uv run python scripts/train_baseline.py --profile dev --seed 42` |

Deterministic under a fixed seed — verified by test, because a 0.4% gap between
two candidates means nothing if re-running would reorder them.
