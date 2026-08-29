# Validation

## Why cross-validation is not enough

Hold out any share of the rows and the counterfactual is still missing from the
held-out part. A model can predict the outcome perfectly and be wrong about the
effect by any margin, and no split of the data will reveal it.

So the validation here is not a held-out score. It is six checks, each of which
can fail, and two of which can invalidate a run outright.

---

## 1. Recovery of known effects

The only test that can establish correctness. `ml/promo_uplift/synthetic.py`
generates panels where the effect is applied by hand and recorded, so recovery is
checkable against a number rather than a plausibility judgement.

The DGP is log-additive and matches the platform generator's structure, so the
estimators face the same functional form they will meet in production. Both
potential outcomes share one noise draw — that is what makes them potential
outcomes for the *same* unit rather than two different worlds.

### Measured results

200 series, 365 days, seed 42. `--synthetic --all-scenarios`.

| Scenario | True ATT | Naive | AIPW | Error | SE | CI covers truth |
|---|---|---|---|---|---|---|
| positive | +63.6% | +53.5% | **+65.9%** | 2.3% | 7.1% | yes |
| negative | −9.4% | −14.3% | **−5.9%** | 3.5% | 2.4% | yes |
| null | 0.0% | −5.6% | **+3.8%** | 3.8% | 3.4% | yes |
| confounded | +65.9% | **+123.5%** | **+60.9%** | 5.1% | 4.1% | yes |
| confounded_null | 0.0% | **+34.9%** | **+0.2%** | 0.2% | 0.5% | yes |
| heterogeneous | +67.7% | **+126.1%** | **+63.1%** | 4.5% | 5.2% | yes |

**6/6 recovered.** Tolerance is 2.5 standard errors, floored at 2 percentage
points — taken from the estimator's own reported uncertainty rather than from a
round number, because a flat threshold means two different things at +65% and at
0%.

### What each row establishes

- **positive / negative / null** — the estimator is unbiased under random
  assignment, and **does not clip a negative effect at zero**. A promotion that
  destroys volume is the finding Step 8 most needs to act on.
- **confounded** — the naive method overstates by **57.6 points**; adjustment
  removes 92% of that bias.
- **confounded_null** — the sharpest test in the suite. Promotions targeted at
  exactly the days that would have sold well anyway, doing nothing. The naive
  method finds **+34.9%** of entirely spurious uplift; AIPW returns **+0.2%**.
  Any method that reported otherwise would, on real data, invent effects for
  promotions that did nothing.
- **heterogeneous** — the aggregate is strongly positive while segment C has a
  negative mechanic, which is why an aggregate ATT is not enough to allocate a
  budget with. CATE ranking: **A > B > C, correct**.

---

## 2. Recovery on the real dataset

The platform generator records the true promotion response curve per product and
mechanic in `ground_truth/promotion_uplift.json`, and the true own-price
elasticity in `elasticity.json`. The expected effect is rebuilt from them:

```
τ_log = a·(1 − e^(−b·d))   +   beta_own · log(1 − d)
        └── mechanic ──┘       └── price cut ──┘
```

**Measured.** 300 product-store pairs, **4,417 real promotion events**:

| | |
|---|---|
| Expected (from the generator's parameters) | **+71.3%** |
|   mechanic | +17.7% |
|   price channel | +45.6% |
| **Estimated (AIPW)** | **+72.0%** |
| **Absolute error** | **0.7 percentage points** |

Two things this shows. The estimator works on the actual platform data, not only
on a purpose-built panel. And the **price channel is 2.6× the mechanic** — the
measurement that justifies defining treatment as the whole event rather than the
promotion flag.

### What this validates, and what it does not

The store-level `_promo_responsiveness` and a per-event `N(1, 0.18)` regional
draw are **not persisted**. Both have expectation near 1, so averaging over 4,417
events recovers the mean effect — but **no individual event's effect is
checkable**. This validates the ATT in expectation, never an ITE, and the result
object carries that caveat rather than a docstring.

---

## 3. Placebo

Move the treatment window 30 days earlier, into a period where no promotion ran.
The true effect there is zero by construction, so anything found is attributable
to the method.

The full pipeline re-runs — treatment frame, control pool, covariates, nuisance
models, AIPW — rather than reusing anything already fitted. A placebo sharing a
fitted model with the real run would test less than it appears to.

Real treated and washout rows are **dropped entirely** from the placebo frame.
Leaving them in the control pool would put genuinely promoted days on the other
side of the comparison, and the placebo would find a large negative effect for an
entirely mechanical reason.

**Measured**: +2.11% against a +62.4% real estimate — 3% of it. Threshold 5%.
**A placebo failure blocks the run.**

The result is reported next to the real estimate deliberately: a placebo of +2%
beside a real +60% is reassuring; the same +2% beside a real +3% is not, and a
bare pass/fail hides the difference.

---

## 4. Overlap and balance

**Overlap.** Propensity scores trimmed to [0.02, 0.98]; the trimmed share, the
effective sample size and the common-support range are reported. Past
`max_trimmed_share` the estimate is refused.

Trimming means **dropping** rows, not clipping scores to the boundary. Clipping
keeps an extreme observation and hands it the largest weight the range allows —
the row still dominates, and the trimmed-share diagnostic reads zero so nobody
notices.

**Balance.** Standardised mean difference per covariate, before and after
weighting, against a 0.10 threshold. The SMD rather than a t-test: with 40,000
rows a 0.5% difference is highly significant and irrelevant; with 200 rows a 40%
difference can be non-significant and fatal. The SMD measures how far apart the
groups are, not how confident we are that they differ.

**Balance failure is method-aware**, and the reason is measured:

| Method | Balance fails | Why |
|---|---|---|
| IPW, matching | **blocks** | Nothing but the propensity model. Unbalanced covariates mean the comparison is not adjusted |
| AIPW, DR-learner | **warns** | Consistent if the outcome model is right |

On the confounded synthetic panel the worst SMD after weighting was **0.38** —
far past threshold — and AIPW recovered **+65.2%** against a true **+63.3%**.
That is the doubly robust property doing its job, observed rather than asserted.

**A measured fix.** Weight stabilisation exists because of a failure. With raw
`e/(1−e)` weights, one control row scored 0.98 received weight 49 while the 99th
percentile of weights sat below 1 — a handful of rows *were* the weighted control
mean. Balance went from +0.27 before weighting to **−0.38 after**: worse, in the
opposite direction. Capping at the 99th percentile fixed it.

---

## 5. Sensitivity

Re-estimate while varying the choices nobody can prove correct: washout length
(0/5/10/21 days), control window (21/45/90 days), trimming level
(0.01/0.02/0.05). Ten specifications.

The estimator is **not** varied — that is what the comparison table is for.

**Measured** (150-pair panel): 10/10 specifications estimable, spread 5.3 points
= **9% of the headline estimate**.

The relative figure is the one that matters. A 5-point spread around a 60% effect
is robustness; the same spread around a 4% effect means the specification is
doing the work, not the data. Above 50% relative spread a warning fires.

**It is expensive, and that is a real constraint rather than a footnote.** Ten
specifications means ten full pipeline runs — panel, controls, covariates,
cross-fitted nuisance models, estimate. On a 300-pair panel (~330,000 rows) the
sweep did not complete within 50 minutes and the run was abandoned. It is
therefore **opt-out** via `--no-sensitivity`, and the routine refresh should skip
it while a scheduled deeper run keeps it.

The report says which mode produced it. A number without its sensitivity sweep is
not wrong, but it is less established than one with it, and a reader has to be
able to tell the difference.

---

## 6. Parallel trends

Regress the outcome on `time × treated` over the pre-period. Under parallel
trends the interaction is zero.

**Measured**: slope difference +0.0103 units/day, t = 0.46, **p = 0.64 — not
rejected**.

And DiD still returned **+45.8%** against a true **+66.6%**. Passing the test does
not vindicate DiD; only failing it disqualifies DiD. Both outcomes are reported,
and DiD stays in the comparison table with its diagnostic attached.

---

## The validation verdict

| Status | Meaning | Estimate returned? |
|---|---|---|
| `passed` | Every check the method depends on held | yes |
| `warnings` | Something is off; the estimate is still identified | yes |
| `failed` | An assumption the method cannot do without was violated | **yes, labelled** |

A `failed` estimate is returned rather than suppressed. Withholding it does not
protect anyone — the caller can compute a naive number in one line of SQL — so
the choice is between our number labelled and theirs unlabelled. The tool
promotes the failure to the **first** warning so a supervisor reading a truncated
list cannot miss it, and the CLI exits non-zero.

---

## Test coverage

**175 tests** in `tests/promo_uplift/`; **777** repo-wide.

| File | Tests | Covers |
|---|---|---|
| `test_service.py` | 30 | Every documented refusal, provenance, persistence, the tool contract |
| `test_estimators.py` | 27 | AIPW arithmetic and the clustered SE reconstructed by hand, influence function mean-zero, fold schemes, weight stabilisation |
| `test_data_quality.py` | 22 | Every check twice — clean panel passes, corrupted panel fires |
| `test_features.py` | 21 | Anchoring, mediator and collider exclusion, hand-computed trailing means |
| `test_placebo.py` | 18 | Placebo frame construction, sensitivity spread, the method-aware verdict |
| `test_business_metrics.py` | 15 | ROI, negative uplift, value-destroying events |
| `test_treatment.py` | 15 | Windows, qualification, washout precedence, duplicate grain |
| `test_synthetic_effect.py` | 15 | Recovery, confounding, CATE ranking, generator soundness |
| `test_controls.py` | 12 | Pool composition, distance to event, refusals |

Two properties the suite enforces that are easy to miss:

- **Every quality check is proven to fire** against a deliberately corrupted
  frame. A check that has never failed is indistinguishable from one that does
  nothing.
- **The falsifiability test is honest about what it can prove.** An earlier
  version patched `POST_TREATMENT_FEATURES` and rebuilt the covariates — which
  can never fail, because the same set filters the feature list *and* backs the
  assertion. It passed while proving nothing. The guard is now exercised
  directly, and the real protection is documented as the allow-list: a column
  cannot become a covariate by appearing in the panel, only by being constructed
  by one of the three feature builders.
