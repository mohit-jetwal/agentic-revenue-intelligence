# Causal methodology

## Prediction versus causal inference

Steps 5 and 6 answer predictive questions. Step 7 does not, and the difference is
not one of difficulty — it is one of kind.

| | Question | Testable by holding out data? |
|---|---|---|
| Forecasting | What **will** sales be? | Yes — the answer arrives |
| Baseline | What are **normal** sales? | Yes — measured against latent demand |
| **Uplift** | What **would** sales have been without the promotion? | **No** |

The counterfactual is missing from every dataset that will ever exist. Hold out
any share of the rows and it is still missing from the held-out part. So a model
can predict the outcome perfectly and be wrong about the effect by any margin,
and **nothing in the data will say so**.

That single fact explains most of the design decisions here: the synthetic
generator, the placebo test, the sensitivity sweep, the method comparison, and
the refusal to report a number without a validation status.

## Potential outcomes

For each product-store-day `i`:

```
Y_i(1)   sales if promoted
Y_i(0)   sales if not promoted
```

Exactly one is ever observed. That is the **fundamental problem of causal
inference**, and it is a missing-data problem, not an estimation problem.

```
ITE_i  = Y_i(1) − Y_i(0)                individual effect. Never identified
ATE    = E[Y(1) − Y(0)]                 average over everything
ATT    = E[Y(1) − Y(0) | T = 1]         average over what was promoted
CATE(x)= E[Y(1) − Y(0) | X = x]         average within a covariate profile
```

**This package targets the ATT.** Two reasons.

It is the business question: "what did the promotions we ran achieve", not "what
would happen if we promoted everything, including SKUs nobody would ever
promote". And it needs overlap only on the treated support — there is no
requirement that every unpromoted store-day could plausibly have been promoted,
which is a materially weaker assumption than the ATE needs.

## Identification

The ATT is identified under three conditions.

**1. Conditional ignorability.** Given the pre-treatment covariates `X`,
treatment is as good as random:

```
Y(0) ⊥ T | X
```

Untestable. Always. No diagnostic can confirm it, and any document claiming
otherwise is wrong.

**In this dataset it happens to hold, and that is stated plainly rather than
quietly enjoyed.** `promotion_generator.py:128-131` draws promotion start days
with weights `exp(targeting × 2 × seasonal)`, where `seasonal` is the category's
own annual cycle. Nothing else in assignment depends on demand — event *count* is
`U{3..9}` per year, independent of product velocity. So the back-door set is
`{category seasonality on the date, product, store}`, all observed.

Real promotion data will not be so kind. A merchandiser's judgement about which
SKU deserves investment is not in any table. The estimator is therefore built to
*test* what can be tested and to refuse when the tests fail, rather than to lean
on an assumption that happens to hold here.

**2. Positivity (overlap).** Every treated unit had a non-zero chance of not
being promoted:

```
0 < P(T = 1 | X) < 1
```

**Testable**, and tested. Where the propensity approaches 1 the inverse-probability
weight `e/(1−e)` diverges — a control row scored 0.98 receives weight 49, and the
weighted control mean becomes that row's outcome with extra arithmetic. See
[`validation.md`](validation.md#overlap).

**3. SUTVA.** One unit's treatment does not affect another's outcome. **Violated
here**, knowably: promoting a SKU takes volume from its substitutes through the
cross-price term in `sales_generator.py:270`. The consequence is that the control
group for a promoted SKU includes days when its substitutes were promoted, which
biases the estimate. Recorded in [`assumptions.md`](assumptions.md), not solved.

## The adjustment set, and the two ways to get it wrong

A covariate belongs in the adjustment set if it is a **common cause** of
treatment and outcome. Two failure modes matter more than any other.

### Over-adjustment: conditioning on a mediator

`discount_percentage` and `selling_price` are *consequences* of the promotion,
not causes of it:

```
promotion ──→ discount ──→ demand      (mediated path)
promotion ──────────────→ demand      (direct, the mechanic)
```

Conditioning on discount blocks the first path. What survives is the mechanic
alone — and on this data that is **+17.7% against a true +71.3%**, a number that
looks entirely plausible and is wrong by a factor of four.

This is the single most likely way to produce a confident wrong answer here,
which is why those columns sit in `POST_TREATMENT_FEATURES` with a runtime guard
rather than merely being omitted.

### Collider adjustment: conditioning on a consequence

`stockout_flag` is caused by the promotion (demand outruns the reorder policy)
and correlated with demand. Conditioning on it **opens** a path that was closed.
It is used to filter rows and never enters the adjustment set — see
[`assumptions.md`](assumptions.md#stockouts) for why the filtering itself is a
compromise.

### What is in

| Group | Examples | Why |
|---|---|---|
| Demand history | lags 1/7/14/28, rolling means 7/14/28/56, volatility, momentum | The listing's state before the decision |
| Prior promotion intensity | `promo_share_28`, `promo_share_90`, `days_since_promotion` | Heavily promoted listings differ in demand too |
| **Seasonality** | `season_sin_1/2`, `season_cos_1/2`, month, day of week | **The confounder itself** |
| Price level | `regular_price_lag_1`, `price_vs_trailing_mean` | Regular price, not the promotional one |
| Static | category, region, channel, store segment | Stratify the comparison |

All trailing statistics are anchored at the **event start**, not the row's own
date. On day five of a promotion, a trailing 7-day mean anchored at the row would
contain four days of the effect being estimated. See
[`feature_catalog.md`](feature_catalog.md).

## The estimators

Six, targeting the same ATT and differing only in what must be true.

| # | Method | Identifying assumption | Role |
|---|---|---|---|
| 0 | `naive_during_vs_before` | none — it is wrong | The foil |
| 1 | `baseline_counterfactual` | Step 5's baseline is unbiased | Reuses the existing artifact |
| 2 | `difference_in_differences` | parallel trends | Implemented **with a test that can reject it** |
| 3 | `inverse_probability_weighting` | propensity model correct | Fragile at extreme scores |
| 4 | **`augmented_ipw`** | **either** nuisance model correct | **The headline** |
| 5 | `dr_learner` | as AIPW, plus a CATE model | Segment ranking |

### AIPW, the whole estimator

```
τ̂ = (1/n₁) Σᵢ [ Tᵢ(Yᵢ − μ̂₀(Xᵢ)) − (1−Tᵢ)·(êᵢ/(1−êᵢ))·(Yᵢ − μ̂₀(Xᵢ)) ]
```

The first term is the treated residual against the outcome model. The second
removes the part of it explained by control rows that look equally promotable.

**Double robustness**: consistent if **either** `μ₀` **or** `e` is correctly
specified — not both. Two chances to be right instead of one.

It is not magic. If both are wrong the estimate is wrong, and the property says
nothing about which is more likely to be right on your data. What it does buy was
observable here: on the confounded synthetic panel the worst standardised
difference after weighting was **0.38**, far past the 0.10 threshold — the
propensity model plainly did not balance the demand covariates — and AIPW still
recovered **+65.2%** against a true **+63.3%**. The outcome model carried it.
That is why balance failure *blocks* IPW and only *warns* for AIPW.

### The influence function and the interval

```
ψᵢ = (1/π)[ Tᵢ(Yᵢ − μ̂₀) − (1−Tᵢ)(ê/(1−ê))(Yᵢ − μ̂₀) − Tᵢτ̂ ],   π = E[T]
```

Standard errors are **clustered on the product-store listing**, not computed from
`sd(ψ)/√n`.

That correction came from a measurement, not from theory. With the i.i.d. formula
the intervals on the synthetic panels were 0.5–1.5 percentage points wide and
**failed to cover the known truth in four of six scenarios**, while the point
estimates were within 2–5 points throughout. The estimates were fine; the
intervals were three to five times too narrow, because rows within a listing are
strongly serially correlated and the effective sample size is closer to the
number of listings than the number of rows. Clustering fixed it: **6/6 scenarios
now covered**.

It also matches what the bootstrap does — resampling whole series — so the
analytic and resampled intervals rest on the same independence assumption rather
than two different ones.

### Cross-fitting

Nuisance models are fitted out-of-fold, holding out **whole listings**.

The tempting alternative — contiguous date blocks, as a forecasting split would
use — is actively harmful here. Every fold is then predicted by a model trained
only on *other* periods, so any covariate with a time trend is extrapolated
rather than interpolated. Measured: a linear `time_index` under time-block folds
drove propensity scores to the clip boundaries, left the control weights summing
to **43×** the treated count, and returned **−424%** against a true **+65%**.

The check that catches it is now permanent: since `E[(1−T)·e/(1−e)] = P(T=1)`,
the control weights must sum to roughly the treated count, and a ratio outside
[0.7, 1.4] raises a warning on the estimate.

### The outcome model scale

Poisson objective, **not** a log transform of the target. Both assume
multiplicative structure, which is right. The difference is what comes back out:
a Poisson objective returns the conditional **mean** on the units scale, which is
the scale the AIPW residuals `(y − μ₀)` are taken on. Fitting `log1p(y)` and
inverting returns a median-like quantity biased low by an amount that varies with
residual variance — so the bias does not cancel between arms and lands directly
in the effect. Step 6 hit exactly this.

## Difference-in-differences: evaluated, not assumed

```
effect = (treated_post − treated_pre) − (control_post − control_pre)
```

DiD differences away every time-invariant difference between the groups, which is
genuinely powerful. It rests entirely on **parallel trends**: absent treatment,
the two groups would have moved together.

That is not implied by the data, not implied by randomisation, and routinely
asserted rather than checked. Here it is checked by regressing the outcome on
`time × treated` over the pre-period; under parallel trends the interaction is
zero.

**And the test passing is not enough.** On the confounded synthetic panel the
pre-trend test did **not** reject (p = 0.64) and DiD still returned **+45.8%**
against a true **+66.6%** — a 21-point error. Failing the test disqualifies DiD;
passing it does not vindicate DiD. Both directions are reported.

When DiD *is* the right tool here: a promotion that ran in some stores and not
others, at the same time, where store choice was unrelated to the demand
trajectory. When it is not: a promotion timed to a demand upswing, which is most
of them.

## What was considered and not built

| Approach | Why not |
|---|---|
| S-learner | One model with treatment as a feature lets a tree ignore it wherever other splits pay better, shrinking the effect for reasons about the loss function rather than the data |
| T-learner alone | Already the nuisance structure inside AIPW; standalone it loses the residual correction |
| X-learner | Designed for badly imbalanced arms. The arms here are 1:4, which does not call for it |
| Synthetic control | Built for one treated unit and many controls. Here there are thousands of treated units |
| Instrumental variables | No credible instrument. Commodity cost shifts price but not the promotion decision |
| econml / dowhy | Two heavy dependencies for ~200 lines of statistics that are better understood written out |

The brief warns against implementing every algorithm for its own sake. Three more
meta-learners on this data would land within noise of each other and of AIPW.
