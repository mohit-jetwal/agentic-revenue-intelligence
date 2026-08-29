# Assumptions, and what breaks each one

Every uplift number this package produces rests on the assumptions below. They
travel with the result — the service returns them in `assumptions`, the tool
passes them to the agent, and MLflow logs them as a structured artifact so two
runs can be diffed.

They are listed in order of how likely they are to be the thing that is wrong.

---

## 1. Conditional ignorability

> Given the pre-treatment covariates, treatment is as good as random.

**Untestable.** No diagnostic confirms it. Balance and overlap check whether the
*observed* covariates were adjusted for; neither says anything about the ones
nobody recorded.

**In this dataset it holds**, and that is a property of the generator rather than
an achievement of the method. `promotion_generator.py:128` targets promotion
*timing* at each category's seasonal peak, and nothing else in assignment depends
on demand — event count is `U{3..9}` per year regardless of velocity. The
back-door set is `{category seasonality, product, store}`, all observed.

**What breaks it in production**: a merchandiser's judgement. "This SKU deserves
investment this quarter" is a real driver of assignment and is in no table. If
that judgement correlates with expected demand — it does — the estimate is
biased and no diagnostic here will say so.

**Mitigation, not solution**: report the estimate alongside its balance,
overlap and sensitivity diagnostics, so a reader can see what *was* adjusted for.
A sensitivity analysis to unmeasured confounding (Rosenbaum bounds, E-values) is
in [`../models/promo_uplift_model_card.md`](../models/promo_uplift_model_card.md)
as future work, not implemented.

---

## 2. Positivity / overlap

> Every treated unit had a non-zero chance of not being promoted.

**Testable**, and tested. Propensity scores are trimmed to [0.02, 0.98]; the
trimmed share, the effective sample size and the common-support range are all
reported, and the estimate is refused past a configured trimming threshold.

**What breaks it**: a listing promoted almost continuously. Its propensity
approaches 1, the ATT weight `e/(1−e)` diverges, and the weighted control mean
becomes one row's outcome with extra arithmetic. The `always_treated_series`
quality check flags these before the estimator sees them.

**Consequence when it partially fails**: trimming changes the estimand to
"listings that were sometimes not promoted". Reported, not absorbed.

---

## 3. SUTVA / no interference

> One unit's treatment does not affect another's outcome.

**Violated, knowably.** `sales_generator.py:270` computes cross-price effects
using the *selling* price, so promoting a SKU lowers its price and takes volume
from its substitutes in the same store.

**Two consequences**, in opposite directions:

- The control pool for a promoted SKU contains days when its substitutes were
  promoted, depressing the baseline and **overstating** uplift.
- The promoted SKU's own gain is partly transferred volume, so **category**
  profit is lower than the reported figure.

**Every business impact figure is therefore an upper bound on category profit**,
and says so in its assumptions. Cross-price effects arrive in Step 9 and can be
subtracted then.

---

## 4. Stockouts, and the estimand they change

Censored rows are excluded by default. This is a genuine compromise and it is
worth being precise about why it is not simply correct.

**The problem.** `observed = min(latent, available)`. Promotions raise demand,
demand outruns the reorder policy, so **treated rows censor more than control
rows**. Measured on the synthetic panel with censoring enabled: **8.2% of treated
rows against 1.8% of control rows**.

**Why dropping them is not neutral.** Stockout is a *post-treatment* variable.
Conditioning on it selects on a consequence of treatment and removes precisely
the highest-demand promotion days, biasing the estimate **downward**.

**Why they are dropped anyway.** The alternative is worse. A censored outcome
records what was available to sell, not what customers wanted; keeping it treats
a supply failure as a demand failure and understates the promotion's effect on
exactly the days it worked best.

**What ships with the compromise:**

1. **The estimand is restated**: "the effect on sales among promotion-days where
   stock was available". That sentence is in every result.
2. **Censoring is reported per arm.** A gap between them means the exclusion is
   selective, not incidental, and a warning fires above 5 percentage points.
3. **A bracketing sensitivity** holds censored outcomes at the availability cap —
   a lower bound on true demand — to show how much the exclusion moved things.

Silently zeroing stockout sales, which the brief warns against, would be the
worst of the three: it would treat every stockout as a demand collapse.

---

## 5. Parallel trends (difference-in-differences only)

> Absent treatment, treated and control series would have moved together.

**Testable**, and tested by regressing the outcome on `time × treated` over the
pre-period.

**Passing the test is not enough.** On the confounded synthetic panel the
pre-trend test did not reject (p = 0.64) and DiD still returned **+45.8%** against
a true **+66.6%**. Failing the test disqualifies DiD; passing it does not
vindicate it. Both directions are reported.

---

## 6. Correct specification of at least one nuisance model

AIPW is consistent if **either** the outcome model or the propensity model is
right. Not both.

**What breaks it**: both wrong. The property offers two chances, not a guarantee,
and says nothing about which is more likely to be right on your data.

**Observed here**: the propensity model did *not* balance the demand covariates
on confounded data — worst standardised difference **0.38** against a 0.10
threshold — and AIPW still recovered **+65.2%** against a true **+63.3%**. The
outcome model carried it. That is why balance failure blocks IPW and only warns
for AIPW, and it is a measured result rather than a theoretical concession.

---

## 7. Independence between listings

Standard errors cluster on the product-store listing. That assumes different
listings are independent of each other.

**Partly false**: substitutes in the same store are correlated, and everything in
a region shares regional shocks.

**Direction**: the intervals are slightly too narrow. Clustering at the region or
store level would be more conservative, at the cost of far fewer clusters and a
less stable variance estimate.

**What clustering already fixed**: with the naive i.i.d. formula the intervals
covered the known truth in only **four of six** synthetic scenarios while the
point estimates were within 2–5 points throughout. Clustering on the listing
widened them three- to five-fold and brought coverage to **6/6**.

---

## 8. The Step 5 baseline is unbiased

Applies only to the `baseline_counterfactual` estimator.

**Known false, and quantified.** `docs/models/baseline_sales.md` records the
selected LightGBM baseline over-predicting by **+6.7%**, and explicitly defers
the trade-off to "whoever owns the uplift numbers in Step 5".

**This step owns them.** The bias is corrected explicitly in
`ml/promo_uplift/baseline.py`, the correction factor is reported in the
estimator's diagnostics, and the uncorrected variant is available behind a flag.

---

## 9. Synthetic data resembles real retail

Every measured number in this documentation comes from generated data.

The generator is unusually good — log-additive demand, negative-binomial
over-dispersion, endogenous stockouts, targeted promotion timing, pull-forward,
cross-price substitution — and it is still a simulation. Real retail has
structure it does not reproduce: competitor reactions, assortment changes,
execution failures, and a merchandiser's private information about which
promotions will work.

**The recovery results establish that the estimator is correct given the
assumptions.** They do not establish that the assumptions hold in production.
Those are different claims and this document keeps them apart.
