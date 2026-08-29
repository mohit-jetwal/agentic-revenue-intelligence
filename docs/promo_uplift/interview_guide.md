# Explaining this in an interview

Twenty-five questions, answered with the reasoning rather than the conclusion.
Every number here was measured on this codebase.

The single most useful habit: **when asked what a method does, say what has to be
true for it to work, and what you did to check.** That is what separates someone
who has read about causal inference from someone who has shipped it.

---

## 1. What is promo uplift?

The incremental sales caused by a promotion — observed sales minus what would
have sold without it.

The second term is the whole problem. It is a **counterfactual**: the promotion
ran, so the world where it did not is unobservable, and no dataset will ever
contain it. Every method in this package is a different way of constructing that
missing quantity.

## 2. Why isn't "promo sales − non-promo sales" valid?

Two compounding errors, both pointing the same way.

**Selection.** Promotions are scheduled into rising demand. The generator does
this explicitly — start days are drawn with weights `exp(targeting × 2 ×
seasonal)` — and real merchandisers do it for the same reason: you promote when
shoppers are already in the aisle. So the comparison books the season as
promotional lift.

**Pull-forward.** Buyers load their pantry and buy less afterwards, 32% of the
lift over ten decaying days. The naive window ends before the dip, so borrowed
volume is counted as new.

**Measured**: on the confounded synthetic panel, naive **+123.5%** against a true
**+65.9%** — overstated by **57.6 points**. And on the scenario where promotions
are targeted but do *nothing*, naive finds **+34.9%** of entirely invented
uplift.

## 3. What is a counterfactual?

The outcome under the treatment that did not happen. For each unit there are two
potential outcomes, `Y(1)` and `Y(0)`, and exactly one is ever observed.

That is the **fundamental problem of causal inference**, and it is a missing-data
problem rather than an estimation problem. It is why you cannot validate a causal
model by holding out rows: the counterfactual is missing from the held-out part
too.

## 4. ATE versus CATE?

```
ATE     = E[Y(1) − Y(0)]                  average over everything
ATT     = E[Y(1) − Y(0) | T = 1]          average over what was actually treated
CATE(x) = E[Y(1) − Y(0) | X = x]          average within a covariate profile
```

**I target the ATT**, for two reasons. It is the business question — "what did
the promotions we ran achieve", not "what if we promoted everything including
SKUs nobody would ever promote". And it needs overlap only on the treated
support, which is a materially weaker assumption.

CATE is what a budget allocation needs. On the heterogeneous scenario the
aggregate is **+67.7%** while one store segment has a *negative* mechanic — an
aggregate alone would keep funding the segment that is losing money.

## 5. How did you deal with confounding?

Identify the back-door path, then close it with pre-treatment covariates.

Here the confounder is **category seasonality on the date**. Promotion timing is
targeted at each category's peak, so treated days are systematically higher-demand
before any promotion runs. The adjustment set carries seasonal harmonics, prior
promotion intensity, trailing demand at several horizons, price level and static
attributes — and the propensity model builds **season × category interactions
explicitly**, because the relationship between date and treatment differs by
category and an additive model cannot represent that.

**Measured**: adjustment removes 92% of the naive bias.

Worth saying out loud: in *this* dataset the confounding is observable by
construction, so ignorability genuinely holds. That is a property of the
generator, not an achievement of the method, and I say so in the docs rather than
letting a clean recovery imply I solved unmeasured confounding.

## 6. What is a propensity score?

`e(X) = P(T = 1 | X)` — the probability this store-day would have been promoted
given what was knowable beforehand. Rosenbaum and Rubin: if treatment is ignorable
given `X`, it is ignorable given `e(X)` alone, so a 30-dimensional adjustment
problem collapses to one dimension.

**The trap**: a propensity model is not trying to predict treatment well. Perfect
discrimination is a *disaster* — it means treated and control units are perfectly
separable, so no comparison exists. The target is **balance, not AUC**. I report
both so the difference is visible; an AUC near 0.5 with good balance beats an AUC
of 0.95.

## 7. What is overlap / positivity?

Every treated unit had a non-zero chance of not being treated:
`0 < e(X) < 1`.

Unlike ignorability this **is testable**, and it is the assumption that actually
bites. Where `e` approaches 1 the ATT weight `e/(1−e)` diverges: a control row
scored 0.98 gets weight 49, and the "estimate" becomes that row's outcome with
extra arithmetic.

I trim to [0.02, 0.98], report the trimmed share and the effective sample size,
and **refuse** past a threshold. Trimming means *dropping* rows, not clipping
scores — clipping keeps the extreme row, hands it the maximum weight, and makes
the diagnostic read zero.

I also cap weights at the 99th percentile, because of a measurement: with raw
weights, balance went from +0.27 before weighting to **−0.38 after**. Worse, in
the opposite direction, because a handful of rows *were* the weighted mean.

## 8. Why use matching?

Weighting and matching fail differently, which is why I run both.

Weighting keeps every observation but can hand enormous influence to a few.
Matching discards unmatched treated units — changing the estimand to "promotions
that had a comparable control" — but every retained pair is concretely comparable
and you can look at the pairs. Agreement is evidence; disagreement localises a
problem.

I match on the **logit** of the propensity, not the score: on the probability
scale 0.01→0.02 looks the same distance as 0.50→0.51, but the first differs by a
factor of two in odds.

**What matching does not do**: manufacture ignorability. It balances what you
matched on. Two store-days identical on every recorded covariate may still differ
in something nobody wrote down.

## 9. Why use difference-in-differences?

It differences away every *time-invariant* difference between the groups — store
size, SKU distribution, regional traffic — without modelling any of them. That is
genuinely powerful and it is why DiD survives where no covariate set would
convince.

## 10. What is the parallel trends assumption?

Absent treatment, the two groups would have moved **together**. Not that they are
at the same level — that differences away — but that their *trajectories* match.

I test it by regressing the outcome on `time × treated` over the pre-period;
under parallel trends the interaction is zero.

**The answer that matters**: on my confounded panel the test did **not** reject
(p = 0.64) and DiD still returned **+45.8%** against a true **+66.6%** — a
21-point error. **Failing the test disqualifies DiD; passing it does not
vindicate DiD.** The test has power against linear pre-trends and little against
anything else. I report both outcomes and keep DiD in the comparison table with
its diagnostic attached.

## 11. What is a doubly robust estimator?

```
τ̂ = (1/n₁) Σᵢ [ Tᵢ(Yᵢ − μ̂₀(Xᵢ)) − (1−Tᵢ)·(êᵢ/(1−êᵢ))·(Yᵢ − μ̂₀(Xᵢ)) ]
```

Consistent if **either** the outcome model `μ₀` **or** the propensity model `e`
is correctly specified — not both. Two chances to be right.

It is not magic: if both are wrong the estimate is wrong, and the property says
nothing about which is more likely to be right on your data.

**What I can show it bought**: on the confounded panel the propensity model
plainly failed to balance the demand covariates — worst standardised difference
**0.38** against a 0.10 threshold — and AIPW still recovered **+65.2%** against a
true **+63.3%**. The outcome model carried it. That measurement is why balance
failure *blocks* IPW in my pipeline and only *warns* for AIPW.

## 12. How did you validate the causal assumptions?

Six checks, two of which can invalidate a run:

| Check | Result |
|---|---|
| Recovery of known effects | **6/6 scenarios** |
| Ground-truth recovery on real data | **0.7 pp error** on 4,417 events |
| Placebo | +2.11% against a +62.4% real estimate |
| Overlap | trimmed share, ESS, common support |
| Balance | SMD per covariate, method-aware verdict |
| Sensitivity | spread across 10 specifications = 9% of the headline |

I use the standardised mean difference rather than a t-test on purpose. A t-test
answers "is this distinguishable from zero", which is a question about sample
size: with 40,000 rows a 0.5% difference is highly significant and irrelevant.

## 13. How did you handle stockouts?

This is the question I would most want to be asked, because the obvious answer is
wrong.

`observed = min(latent, available)`. Promotions raise demand, demand outruns the
reorder policy, so **treated rows censor more than control rows** — measured,
8.2% versus 1.8%.

So stockout is a **post-treatment variable**. Dropping those rows is conditioning
on a consequence of treatment; it removes precisely the highest-demand promotion
days and biases the estimate **downward**.

I drop them anyway, because the alternative is worse — a censored outcome records
what was available, not what was wanted. But three things ship with the
compromise: the **estimand is restated** ("the effect on sales among days where
stock was available"), censoring is **reported per arm** with a warning when the
gap is material, and a **bracketing sensitivity** holds censored outcomes at the
availability cap to show how much the exclusion moved things.

Silently zeroing stockout sales would be worst of all: it treats a supply failure
as a demand collapse.

## 14. How did you prevent temporal leakage?

Three mechanisms.

**Anchoring.** Every trailing covariate for a treated row is measured as of the
**event start**, not the row's own date. On day five of a promotion, a trailing
7-day mean anchored at the row contains four days of the effect being estimated.
Tested: covariates are constant within an event, and match a trailing mean
reconstructed by hand from the source panel.

**Post-treatment exclusion.** `selling_price` and `discount_percentage` are
*mediators*, not confounders. Conditioning on them holds the price cut fixed
across arms and measures only the mechanic — **+17.7% against a true +71.3%**, a
number that looks entirely plausible and is wrong by a factor of four.

**Cross-fitting by listing.** Whole product-stores are held out, so nothing about
a listing informs its own counterfactual.

## 15. How did you test the causal model with no real ground truth?

I generated data where the effect is known by construction, applied by hand and
recorded, so recovery is checkable against a number.

Then — and this is the part that makes it more than a toy — I validated on the
**platform dataset**, whose generator records the true promotion response curve
per product and mechanic. Rebuilding the expected effect from those parameters
gives **+71.3%** across 4,417 events; the estimator returns **+72.0%**.

I also state what that does *not* validate: two per-event terms are not
persisted, so it validates the average effect, never an individual promotion's.

## 16. What is a placebo test?

Pretend the promotion happened when it did not — here, shifted 30 days earlier
into an untreated window. The true effect is zero by construction, so anything
found is attributable to the method.

It is the closest thing causal inference has to a unit test, and it is the one
diagnostic that invalidates a run outright.

Two details that matter. The full pipeline **re-runs** rather than reusing
anything fitted — a placebo sharing a fitted model tests less than it appears to.
And the real treated and washout rows are **dropped entirely**; leaving them in
the control pool would put genuinely promoted days on the other side of the
comparison and produce a large negative "effect" for a mechanical reason.

**Measured**: +2.11%, which is 3% of the real estimate. Reported *next to* the
real estimate deliberately — a placebo of +2% beside a real +60% is reassuring;
the same +2% beside a real +3% is not.

## 17. Why is synthetic data useful?

Because causal validation is otherwise impossible. Hold out any share of real
data and the counterfactual is still missing, so a model can predict the outcome
perfectly and be wrong about the effect by any margin.

The scenario I would lead with is **`confounded_null`**: promotions targeted at
exactly the days that would have sold well anyway, doing nothing at all. The
naive method finds **+34.9%**; AIPW returns **+0.2%**. Any method that reported
otherwise would, on real data, invent effects for promotions that did nothing —
which is the specific failure the whole capability exists to prevent.

## 18. How did you calculate incremental profit?

```
incremental_units   = ATT × treated_days
incremental_margin  = incremental_units × promotional unit margin
incremental_profit  = incremental_margin − promotion_spend
roi                 = incremental_profit / promotion_spend
```

Four decisions change the answer:

- **Margin at the promotional price**, not the regular one. Those units were sold
  at a discount. Valuing them at full margin is the most common way a losing
  promotion is reported as a winner.
- **Incremental revenue, not total.** The promotion did not generate the sales
  that would have happened anyway.
- **Spend summed over events, not rows.** Broadcasting a per-event total across a
  twenty-day window and summing multiplies it by twenty.
- **ROI is `None` when spend is zero**, not infinity.

And the declared gap: **cannibalisation is not deducted**, so every profit figure
is an upper bound on category profit.

## 19. Why can uplift be negative?

Cannibalisation, poor design, stockout, wrong timing, brand damage from deep
discounting, or measurement error. Real, and nothing in the pipeline floors it at
zero.

The `negative` scenario exists to prove it: true −9.4%, estimated −5.9%, sign
preserved. An estimator that cannot report a negative effect is useless for
allocation, because the promotions worth cutting are exactly the ones it would
hide.

## 20. How will this support Trade Promotion Optimization?

It produces a per-event table — one row per promotion with incremental units,
revenue, profit, spend and ROI, ranked. Step 8 allocates budget against
`incremental_profit` subject to budget, frequency, margin and inventory
constraints.

Two design choices made for Step 8 specifically. **Value-destroying promotions
stay in the table** rather than being filtered — the optimiser's job is partly to
allocate away from them, which it cannot do if they are missing. And the
**DR-learner's CATE model can score hypothetical promotions** that have not run,
which is what a candidate evaluation needs — flagged as extrapolation, because
a candidate outside the observed covariate range is a guess rather than an
estimate.

## 21. How will an agent use this?

Through a typed tool that returns **evidence, not just a number**:
`treatment_definition` (what the number means), `validation_status` (whether it
may be called causal), `assumptions`, `method_reason`, the per-event table, and
the full method comparison including the naive estimate marked ineligible — so
the agent can state the size of the error it avoided.

When validation fails, the tool promotes that to the **first** warning, so a
supervisor reading a truncated list cannot miss it.

## 22. Why shouldn't Claude calculate uplift?

Because the arithmetic is trivial and the answer is wrong. An LLM handed a table
of promoted and unpromoted sales will subtract two averages and produce a
confident figure that overstates by around 58 points on this data.

Uplift needs a counterfactual absent from every dataset, a control group chosen
under stated assumptions, an adjustment set that must *exclude* mediators, and a
validation suite that can reject the whole estimate. None of that is inferable
from numbers in a context window.

The division of labour: the agent decides *which* analyses answer the business
question and composes the results. The deterministic layer produces every number.

## 23. How would this run on Databricks?

**Distributed feature engineering, single-node estimation.** The panel is
millions of rows and building trailing covariates over it is what Spark window
functions are for. Estimation is not: after control selection the analysis frame
is in the hundreds of thousands of rows and fits in one executor, so distributing
the propensity fit would add coordination overhead for no gain.

The service contract does not change. The riskiest part of the port is the
anchoring: `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING` — getting that `1` wrong
silently includes the current day, and on a treated row that is treatment effect
inside a covariate. I would port the hand-computed trailing-mean tests first.

Scheduled as a Workflow that **fails the job** on a data-quality FAIL or a
`failed` validation status, rather than publishing.

## 24. How would you monitor it in production?

| Signal | Why |
|---|---|
| Placebo effect over time | Rising means the method is picking up something other than treatment |
| Sensitivity spread | Widening means the specification is doing the work |
| Overlap trimming and ESS | Rising trimming means the groups are drifting apart |
| Balance, worst SMD | Adjustment stopped working |
| Censoring gap between arms | Supply problems making the exclusion more selective |
| ATT versus the previous run | A large jump is a regime change or a pipeline bug — both need a human |
| Treatment-definition fingerprint | Changed means the numbers stopped being comparable |

Note none of these is "prediction accuracy". Accuracy is not the property that
matters here.

## 25. What would make you reject an uplift estimate?

**Blocking:**
- Placebo finds a material effect where none can exist.
- Overlap fails after trimming — no weighting rescues absent common support.
- Balance fails for a propensity-only method.
- Data quality FAILs: overlapping promotions, duplicate grain, negative units.
- No control group.

**Warning, but I would not publish without investigating:**
- Sensitivity spread above half the estimate.
- Propensity weight calibration outside [0.7, 1.4].
- Censoring gap between arms above five points.
- Effective sample size a small fraction of rows.

**And one that is not a diagnostic**: if the estimate disagrees sharply with the
other five methods for a reason I cannot explain. Agreement is evidence;
unexplained disagreement means one of the assumptions I have not tested is the
one doing the work.

---

## The three stories worth having ready

**The one that shows debugging skill.** Cross-fitting on contiguous *date* blocks
— the intuitive choice, since it is what a forecasting split does — turned a true
+65% into **−424%**. Every fold was predicted by a model trained on other
periods, so the linear time covariate was extrapolated; propensity scores hit the
clip boundaries and control weights summed to **43×** the treated count. Fixed by
holding out whole listings. The permanent guard came out of it: since
`E[(1−T)·e/(1−e)] = P(T=1)`, a weight ratio outside [0.7, 1.4] now raises a
warning on the estimate.

**The one that shows statistical judgement.** The point estimates were fine and
the *intervals* were wrong. With i.i.d. standard errors, coverage of the known
truth was **4/6** while every point estimate was within 2–5 points. Rows within a
listing are strongly serially correlated, so the effective sample size is closer
to the number of listings than the number of rows. Clustering on the listing
widened the intervals three- to fivefold and brought coverage to **6/6**.

**The one that shows honesty about your own tests.** My first falsifiability test
patched the post-treatment exclusion set and rebuilt the covariates, expecting an
assertion to fire. It could never fire — the same set filters the feature list
*and* backs the assertion, so a planted name is removed before the check sees it.
The test passed while proving nothing. I now exercise the guard directly and
document that the real protection is the allow-list: a column cannot become a
covariate by appearing in the panel, only by being constructed by a feature
builder.
