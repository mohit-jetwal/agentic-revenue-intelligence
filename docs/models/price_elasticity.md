# Price elasticity — own and cross

## The question

> How does demand respond to price, and what does this product compete with?

Elasticity is the percentage change in demand per percentage change in price. The
number a pricing decision actually turns on is not the coefficient but a single
bit derived from it: **is |e| > 1?** Elastic demand means a price rise *reduces*
revenue; inelastic means it raises it.

## Why it is not a regression

If prices moved at random, elasticity would be arithmetic. They do not.
`data/generation/generators/pricing_generator.py` raises prices into anticipated
strong demand and discounts into weak demand, so a regression of log quantity on
log price partly recovers the **pricing manager's** behaviour rather than the
**shopper's**.

The bias is toward zero: products look less price-sensitive than they are, which
encourages exactly the wrong recommendation.

**Measured across 30 products against `ground_truth/elasticity.json`:**

| Method | MAE | Signed | est/true | corr | Verdict |
|---|---|---|---|---|---|
| **panel_fe** | **0.074** | −0.052 | **1.032** | **0.99** | selected |
| randomised | 0.910 | +0.210 | 0.886 | −0.04 | biased |
| naive_ols | 0.817 | +0.299 | **0.596** | 0.19 | biased |
| iv_2sls | 1.457 | +0.698 | 0.828 | 0.23 | biased |

**Naive OLS recovers 60% of the true elasticity.** A pricing team acting on it
would systematically under-estimate how much volume a price rise costs them.

## The four estimators

Ordered by what they assume, not by what they produce.

**`naive_ols`** — `log q ~ log p`. Assumes price is exogenous. It is not. Kept in
the comparison precisely to show the size of the attenuation, and marked
non-selectable so it can never be reported as the answer.

**`panel_fe`** — adds product and time fixed effects, absorbed by
within-transformation rather than dummy columns (with hundreds of listings and a
thousand dates the dummy design matrix is enormous and almost entirely zeros).
**Selected**, on measured recovery.

**`iv_2sls`** — instruments price with the category commodity cost index.

**`randomised`** — restricts to price regimes the generator tags
`randomised_test`, where price is exogenous by construction. The cleanest
identification available and the smallest sample.

## Two findings worth the space

### A strong first stage does not make an instrument valid

2SLS is the textbook answer here, and the generator explicitly built a commodity
cost index to be the instrument. It fails.

First-stage F statistics were **enormous** — median 484, maximum 10,038 — so this
is not weak-instrument bias. The problem is the exclusion restriction, and not
because cost enters demand. **The cost index varies only at category × date**
(verified: exactly one value per category-date). Within any one product it is a
pure time series, and time is what has to be controlled for, because seasonality
drives demand directly. The only variation the instrument has is variation that
other things also have.

The F statistic is the number that looked reassuring while the estimate was
wrong, so the guard does not use it. `instrument_diagnostics()` checks directly
whether the instrument takes a single value per date across the estimation frame,
and says so in the estimate's warnings.

### Fixed effects fix seasonal endogeneity, not idiosyncratic endogeneity

`panel_fe` works here because the generator's endogeneity operates through
*anticipated* demand, which is largely seasonal — and seasonality lives in the
date dimension that time fixed effects absorb.

It would **not** work against a manager responding to a daily demand shock. That
confounder varies within every dimension being absorbed, so there is nothing for
the transformation to remove. Anyone reading `panel_fe` as a general cure for
endogeneity is reading it wrong, and
`test_fixed_effects_cannot_fix_idiosyncratic_endogeneity` exists so the
limitation is tested rather than asserted.

## Cross-price

Sign convention, since it is the whole point: **positive means substitutes**
(B gets dearer, A sells more), **negative means complements**.

**The multiple-comparisons problem is the real difficulty.** With 300 products
there are 89,700 ordered pairs; testing them all at the 5% level returns roughly
4,485 "significant" findings from nothing. Two defences:

1. **Restrict the candidate set** to within-category pairs and declared
   relationships, chosen *before* looking at any outcome.
2. **Benjamini-Hochberg** over what remains. FDR rather than Bonferroni: with a
   few dozen candidates Bonferroni leaves nothing significant and hides the real
   substitutions.

Measured on `P00003`, whose ground truth declares exactly one relationship
(`P00036` at +0.4397):

```
40 candidates tested, 37 estimable
P00036 -> P00003:  +0.397  substitute, strong, significant     TRUE +0.440
```

**One substitute, zero complements, zero false positives** across 40 tested pairs.

### The bug that cost the true substitute

The first version dropped promoted rows from *both* products. That is right for
the focal product — a promotion moves its demand through a mechanic unrelated to
any candidate's price — and badly wrong for the candidate, because **a
candidate's promotional price cut is the largest single source of the price
variation that identifies the cross effect**.

Measured on `P00003`/`P00036`: the source's log-price standard deviation fell
from 0.155 to 0.120 and the joined sample from 4,999 rows to 3,640. The true
+0.44 substitute did not surface at all, and a spurious complement did. The two
panels are now prepared differently, and `test_keeping_source_promotions_preserves_the_signal`
guards it.

## Standard errors are clustered

On the listing, not row-wise. Rows within a product-store are strongly serially
correlated, so the effective sample size is closer to the number of listings than
the number of rows.

This was caught by a test rather than anticipated: the unclustered interval
excluded the known elasticity by 0.013 while the point estimate was within 0.02
of it. The estimate was fine and the interval was lying — the same failure Step 7
hit on the uplift influence function.

## Assumptions

- Estimated on **unpromoted, in-stock** days only.
- **Log-log**, so the coefficient is a constant elasticity. Real demand curves
  bend; this is a local approximation around the observed price range and does
  not extrapolate to a price nobody has charged.
- The recorded product elasticity is modulated by a **store-level
  price-sensitivity multiplier the platform does not observe**
  (`beta_own = product_elasticity × store_price_sensitivity` in the generator).
  A pooled estimate therefore recovers the product elasticity times the average
  multiplier — which is why the measured ratio is 1.032 rather than exactly 1.
- Fixed effects absorb seasonal endogeneity only. See above.

## Usage

```powershell
uv run python scripts/estimate_elasticity.py --validate-ground-truth
uv run python scripts/estimate_elasticity.py --product P00003 --cross
uv run ari elasticity --product P00003 --cross --compare
uv run pytest tests/elasticity -v
```

## Files

```
ml/price_elasticity/estimator.py    four estimators, selection, clustered SEs
ml/price_elasticity/data.py         the panel join, in one place
ml/price_elasticity/model.py        FittedElasticityModel
ml/cross_price_elasticity/estimator.py   pairwise, BH correction
ml/cross_price_elasticity/model.py       FittedCrossElasticityModel
app/services/elasticity_service.py  the service seam
app/tools/elasticity_tool.py        the agent-facing contract
scripts/estimate_elasticity.py      validation against ground truth
tests/elasticity/                   51 tests
```
