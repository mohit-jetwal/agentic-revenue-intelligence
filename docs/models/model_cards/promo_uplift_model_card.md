# Model card — Promo Uplift

| | |
|---|---|
| **Name** | `promo_uplift` |
| **Version** | v1.0 |
| **Type** | Causal effect estimator (doubly robust ATT) |
| **Step** | Stage 1, Step 7 |
| **Owner** | Revenue Intelligence platform |
| **Status** | Built and validated on synthetic data. **Not approved for production decisions** |

## What it estimates

The **average treatment effect on the treated**: the incremental units, revenue
and profit caused by the promotions that actually ran, against the counterfactual
in which they did not.

Not a prediction. The quantity has no observed value anywhere and never will —
see [`../../promo_uplift/causal_methodology.md`](../../promo_uplift/causal_methodology.md).

## Intended use

- Measuring what a past promotion generated.
- Ranking products, stores, regions and mechanics by promotional return.
- Identifying value-destroying promotions.
- Supplying `event_impact` to Step 8's trade promotion optimiser.

## Out of scope

| Not for | Use instead |
|---|---|
| Predicting future sales | [Demand forecasting](../../forecasting/README.md) |
| Setting a price | Step 10 |
| Allocating a budget | Step 8, which consumes this |
| Cross-product effects | Step 9 |
| **Any decision without reading `validation_status`** | — |

**Do not use a `failed` estimate as a causal claim.** The number is returned so
it can be inspected, not so it can be quoted.

## Training and analysis data

| | |
|---|---|
| Source | Synthetic CPG/retail panel, `data/generation/` |
| Grain | date × product × store |
| Analysed | 300 product-store pairs, 4,417 promotion events |
| Treatment | Promotion of any mechanic, depth ≥ 5%, ≥ 2 days |
| Control | Same-listing unpromoted days within 45 days, plus never-treated listings in the same category and region |
| Excluded | Washout rows, sub-threshold promotions, stockout rows, rows without 56 days of history |

## Method

**Augmented IPW** (doubly robust), selected because it makes the weakest
identifying assumptions among the methods that passed validation — not because it
produced the largest number. It is usually *smaller* than the naive and IPW
estimates.

- Propensity: L2 logistic with explicit season × category interactions.
- Outcome: LightGBM with a Poisson objective, per arm.
- Cross-fitting: 5 folds, holding out **whole listings**.
- Standard errors: **cluster-robust** on the product-store listing.

Five other estimators run alongside and are reported in the comparison table,
including the naive one, marked ineligible.

## Measured performance

### Recovery of known effects (synthetic, exact truth)

| Scenario | True | Naive | AIPW | Error | CI covers |
|---|---|---|---|---|---|
| positive | +63.6% | +53.5% | +65.9% | 2.3% | yes |
| negative | −9.4% | −14.3% | −5.9% | 3.5% | yes |
| null | 0.0% | −5.6% | +3.8% | 3.8% | yes |
| confounded | +65.9% | +123.5% | +60.9% | 5.1% | yes |
| confounded_null | 0.0% | +34.9% | +0.2% | 0.2% | yes |
| heterogeneous | +67.7% | +126.1% | +63.1% | 4.5% | yes |

**6/6 recovered.** CATE ranking on the heterogeneous scenario: A > B > C, correct.

### Recovery on the platform dataset (4,417 events)

| | |
|---|---|
| Expected, from the generator's recorded parameters | **+71.3%** |
| Estimated | **+72.0%** |
| **Absolute error** | **0.7 pp** |

Validates the **average** effect only: two per-event terms are not persisted, so
no individual promotion's effect is checkable.

### Diagnostics on a representative run

| Diagnostic | Value |
|---|---|
| Placebo | +2.11% against a +62.4% real estimate — PASS |
| Sensitivity spread | 5.3 pp = 9% of the headline |
| Overlap trimmed | 0.0% |
| Effective sample size | 38.9% of rows |
| Worst SMD after weighting | 0.139 (2 of 37 covariates above 0.10) |
| Parallel trends | p = 0.64, not rejected |

## Limitations

1. **Unmeasured confounding is untestable.** In this dataset ignorability holds
   because the generator's targeting is observable. Real merchandiser judgement
   is not, and no diagnostic here would detect it.
2. **Cannibalisation is not deducted.** Every profit figure is an **upper bound**
   on category profit.
3. **Stockout exclusion narrows the estimand** to "days where stock was
   available", and is selective: 8.2% of treated rows against 1.8% of control.
   Biases the estimate downward.
4. **Clustering is at the listing level.** Substitutes in a store and listings in
   a region are correlated, so the intervals remain slightly too narrow.
5. **DiD is reported but rarely trustworthy here.** Passing the pre-trend test
   did not prevent a 21-point error.
6. **Segment estimates are noisier than the aggregate**, and segments below 30
   treated rows return null rather than a number.
7. **The analysis is a snapshot.** Nothing monitors whether it has gone stale.
8. **Synthetic data.** Real retail has structure the generator does not
   reproduce.

## Ethical and business considerations

**Asymmetric cost of error.** Overstating uplift keeps bad promotions funded;
understating it kills good ones. The estimator does not tilt either way, but the
stockout exclusion biases downward and that is stated rather than corrected out
of sight.

**Automation.** This model informs a budget decision that affects retail
partners and category teams. Step 8's optimiser and Step 19's human approval gate
exist so that no allocation reaches a partner without review. An unapproved
analysis (`ModelMetadata.approved = False`) must not be served to an agent.

**No personal data.** The grain is product × store × date. No customer-level
information is used, stored or inferable.

**Interpretability.** Propensity coefficients answer "what drives promotion
assignment" — *who gets promoted*, not what promotions do. Reading a propensity
coefficient as an effect is a category error, and the report says so where the
coefficients are shown.

## Maintenance

| Trigger | Action |
|---|---|
| Treatment definition changes | Re-run. Estimates under different definitions are not comparable |
| New promotion cycle | Re-run; the analysis covers only the promotions present when it ran |
| `validation_status` becomes `failed` | Investigate before publishing anything |
| Placebo effect grows | The method is picking up something other than treatment |
| Sensitivity spread exceeds 50% of the estimate | The specification is doing the work |

## Reproduction

```powershell
uv run python scripts/estimate_uplift.py --synthetic --all-scenarios
uv run python scripts/estimate_uplift.py --validate-ground-truth --sample-pairs 300
uv run python scripts/estimate_uplift.py --sample-pairs 300 --seed 42
uv run pytest tests/promo_uplift -v
```
