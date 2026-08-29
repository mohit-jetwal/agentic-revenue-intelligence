# Step 7 — Promo Uplift: final report

Written against measured results. Every number here came from running the code.

---

## 1. What was already present

Further along than the brief assumed. `ml/promo_uplift/interface.py` already
defined `UpliftResult` — with `confidence_interval`, `pull_forward_units` and
`cannibalisation_units` — from Step 1. `features/engineering/promotion.py`
already separated the promotion *schedule* (knowable in advance) from *realised*
spend and units. `DataRepository.get_promotions` was on the ABC. `ml/base.py`
supplied the model contract and the two exception bases the service maps through.

So Step 7 added an estimator and a service, not a data layer.

**Reused unchanged**: `FittedBaselineModel.predict_panel` (the Step 5
counterfactual), `expand_promotion_calendar`, `PointInTimeView`,
`sample_series`, `AnalyticalTool`, the DI container, and the service/tool/CLI
patterns from Step 6.

**No new dependencies.** AIPW, IPW, propensity, matching and the DR-learner are
~200 lines on scikit-learn, statsmodels and scipy. No cloud infrastructure of any
kind was introduced.

## 2. Four findings from the data that drove the design

**The true effect has two channels, and the smaller one is the obvious one.**
`sales_generator.py:334-341` applies a promotion through the mechanic *and*
through the price cut via own-price elasticity. Measured across 4,417 real
events: mechanic **+17.7%**, price channel **+45.6%**. Conditioning on discount —
the natural thing to do — measures the smaller half.

**The confounder is observable by construction.** `promotion_generator.py:128`
draws start days with weights `exp(targeting × 2 × seasonal)`; event *count* is
`U{3..9}`/year independent of velocity. So ignorability genuinely holds here.
That is a property of the generator, said plainly rather than quietly enjoyed.

**Stockouts are post-treatment.** Promotions raise demand, demand outruns the
reorder policy, so treated rows censor more — measured **9.8% vs 3.1%**.

**The Step 5 baseline over-predicts by 6.7%**, and `baseline_sales.md`
explicitly deferred the trade-off to "whoever owns the uplift numbers". This step
owns them; the correction is applied and reported.

## 3. Architecture

```
ml/promo_uplift/     18 modules
  treatment.py       treated / washout / control / excluded roles
  controls.py        two control pools, sufficiency, refusal
  features.py        pre-treatment covariates, anchored at the event start
  propensity.py      P(T|X), overlap, weight stabilisation
  matching.py        SMD balance, nearest-neighbour on the logit
  estimators.py      IPW, AIPW, DR-learner, cross-fitting, clustered SEs
  baseline.py        naive foil + Step 5 counterfactual
  did.py             DiD + a pre-trend test that can reject it
  diagnostics.py     placebo, sensitivity, verdict, ground-truth comparison
  business.py        units -> revenue -> margin -> profit -> ROI
  evaluate.py        comparison table, selection rule, Qini/AUUC
  quality.py         14 causal data-quality checks
  synthetic.py       exact-truth generator, six scenarios
  pipeline.py        orchestration + report
  model.py           persist / load / serve
  tracking.py        MLflow, versioning the treatment definition
  data.py            the panel join, in one place
  config.py          the estimand, as configuration
```

Plus `app/schemas/promo_uplift.py`, `app/services/promo_uplift_service.py`,
`app/tools/promo_uplift_tool.py`, three CLI commands, and
`scripts/estimate_uplift.py`.

## 4-7. Treatment, control, causal methodology

Full detail in [`treatment_definition.md`](treatment_definition.md),
[`control_definition.md`](control_definition.md) and
[`causal_methodology.md`](causal_methodology.md). In summary:

**Treatment** is the whole promotion event, price cut included — depth ≥ 5%,
duration ≥ 2 days. Three windows: event (gross), washout (pull-forward), net.

**Control** comes from two pools that fail differently: same-listing unpromoted
days within 45 days, plus never-treated listings in the same category and region.
Washout rows are in neither arm.

**Estimand**: the ATT.

**Six estimators**, differing only in what must be true:

| Method | Assumption |
|---|---|
| naive | none — it is wrong, and kept as the foil |
| baseline counterfactual | Step 5's baseline is unbiased |
| DiD | parallel trends, tested |
| IPW | propensity correct |
| **AIPW** | **either** nuisance model correct |
| DR-learner | as AIPW, plus a CATE model |

## 8. Mathematical formulation

```
τ̂_ATT = (1/n₁) Σᵢ [ Tᵢ(Yᵢ − μ̂₀(Xᵢ)) − (1−Tᵢ)·(êᵢ/(1−êᵢ))·(Yᵢ − μ̂₀(Xᵢ)) ]

ψᵢ     = (1/π)[ Tᵢ(Yᵢ − μ̂₀) − (1−Tᵢ)(ê/(1−ê))(Yᵢ − μ̂₀) − Tᵢτ̂ ]

SE     = √( (G/(G−1)) · Σ_g ( Σ_{i∈g} ψᵢ )² ) / n        clustered on the listing
```

DR-learner pseudo-outcome, winsorised at the 1st/99th percentile before the
CATE regression:

```
Y*ᵢ = μ̂₁ − μ̂₀ + Tᵢ(Yᵢ − μ̂₁)/êᵢ − (1−Tᵢ)(Yᵢ − μ̂₀)/(1−êᵢ)
```

## 9-11. Features, confounding, stockouts

Covariates are strictly pre-treatment and **anchored at the event start**, not
the row's own date. 28–76 covariates depending on the panel: demand history,
prior promotion intensity, seasonal harmonics, price level, static attributes.

Mediators (`selling_price`, `discount_percentage`) and the collider
(`stockout_flag`) are excluded by name with a runtime guard, and the real
protection is an allow-list: a column cannot become a covariate by appearing in
the panel.

Stockout rows are excluded, the estimand is restated to say so, censoring is
reported per arm, and a bracketing sensitivity exists. See
[`assumptions.md`](assumptions.md#4-stockouts-and-the-estimand-they-change).

## 12-14. Validation results

### Synthetic recovery — exact known truth

| Scenario | True | Naive | AIPW | Error | CI covers |
|---|---|---|---|---|---|
| positive | +63.6% | +53.5% | **+65.9%** | 2.3% | yes |
| negative | −9.4% | −14.3% | **−5.9%** | 3.5% | yes |
| null | 0.0% | −5.6% | **+3.8%** | 3.8% | yes |
| confounded | +65.9% | **+123.5%** | **+60.9%** | 5.1% | yes |
| confounded_null | 0.0% | **+34.9%** | **+0.2%** | 0.2% | yes |
| heterogeneous | +67.7% | **+126.1%** | **+63.1%** | 4.5% | yes |

**6/6 recovered**, intervals cover truth in all six. CATE ranking on the
heterogeneous scenario: **A > B > C, correct**.

The two rows that carry the argument: **confounded** shows the naive method
overstating by **57.6 points** and adjustment removing 92% of it;
**confounded_null** shows the naive method inventing **+34.9%** of uplift where
the true effect is exactly zero, and AIPW returning **+0.2%**.

### Ground-truth recovery on the platform dataset

| Run | Events | Expected | Estimated | Error |
|---|---|---|---|---|
| **300 pairs** | **4,417** | **+71.3%** | **+72.0%** | **0.7 pp** |
| 150 pairs | 2,180 | +70.3% | +78.6% | 8.3 pp |

The 300-pair run is the better-powered one and is the headline. The 150-pair
figure is included because reporting only the better number would misrepresent
how much of the accuracy is sample size.

Expected is rebuilt from `ground_truth/promotion_uplift.json` and
`elasticity.json`. It validates the **average** effect only — two per-event terms
are not persisted — and the result object carries that caveat.

### Placebo

+2.11% where the true effect is zero, against a +62.4% real estimate. **PASS**.

### Diagnostics on the persisted run

| | |
|---|---|
| Overlap | 0.0% trimmed, ESS 67.3% of rows, AUC 0.559 |
| Balance | 1 of 76 covariates above 0.10 SMD (worst `promo_share_28` at 0.123) |
| Differential censoring | treated 9.8% vs control 3.1% — warning fired |
| Parallel trends | p = 0.64, not rejected — and DiD was still 21 points out |

## 15. Business impact, and an honest problem with it

The persisted run reports **+80.1% uplift, 810,777 incremental units, ₹97.4M
incremental revenue** — and **ROI −0.89** on ₹234.5M of spend, with 96.8% of the
1,995 events value-destroying.

**The uplift is validated. The ROI is a generator artefact, and reporting it as a
business finding would be wrong.**

The evidence, per event:

| | Median | Aggregate |
|---|---|---|
| Promotion spend | ₹118,464 | ₹234.5M |
| Incremental margin | ₹5,934 | ₹26.5M |
| **Margin ÷ spend** | **0.05** | **0.11** |

Spend is roughly **twenty times** the achievable margin at the product-store
grain. That is not a promotional-effectiveness result; it is a units problem in
the generator. `configs/data/dev.yaml` draws `fixed_spend` from ₹15,000–220,000
*per event*, scaled to a business-wide `annual_budget` of ₹100M — but each event
in this analysis is a **single product in a single store**, which sells a few
hundred incremental units over a fortnight and earns a few thousand rupees of
margin. Business-scale trade investment attributed to one listing makes negative
ROI arithmetically inevitable regardless of how well the promotion worked.

The distribution is consistent with that reading rather than with a real finding:
ROI is tightly bunched between −1.0 and −0.84 across the interquartile range,
which is what you get when a large near-constant spend swamps a variable margin,
not what a mix of good and bad promotions looks like.

So the ROI **machinery** is correct and tested — `test_business_metrics.py`
verifies break-even, negative uplift, value-destroying events and interval
propagation against hand-computed figures — while the ROI **numbers on this
dataset** are not interpretable. Step 8 will need either a spend feed at the
right grain, or spend allocated across the listings a trade promotion actually
covers. Recorded as a limitation rather than presented as a result.

## 16. MLflow

Experiment `revenue_intelligence_promo_uplift`. The difference from a forecasting
run: **the treatment and control definitions are logged as parameters**, because
they define *what the number means* rather than how it was produced. Two runs
with different treatment fingerprints are not comparable, and without the
definition recorded there is no way to discover that afterwards.

Logged: the definition as a readable sentence, the config fingerprint, every
method's uplift/units/profit/ROI, the diagnostics, and a structured `assumptions`
artifact that can be diffed between runs.

## 17-18. Service API and output schema

See [`service_contract.md`](service_contract.md). Three fields a forecast
response does not have: `treatment_definition` (required), `validation_status`
(`passed`/`warnings`/`failed`), and `assumptions`. `confidence_interval` is
present only when measured.

## 19. Tests

**175 tests** in `tests/promo_uplift/`, across nine files. Repo total **777**.

Two properties worth naming. Every data-quality check is **proven to fire**
against a corrupted frame. And the falsifiability test is honest about its own
limits — an earlier version patched the exclusion set and rebuilt the covariates,
which can never fail because the same set filters the list *and* backs the
assertion. It passed while proving nothing, and was replaced.

## 20. Failure modes

Nine refusal codes, nine warning conditions, a method-aware balance verdict, and
a `failed` status that returns the estimate **labelled** rather than suppressing
it. See [`failure_modes.md`](failure_modes.md), which also lists the six bugs the
diagnostics caught during this build.

## 21. Known limitations

1. Unmeasured confounding is untestable; no sensitivity bound implemented.
2. Cannibalisation not deducted — profit is an **upper bound** on category profit.
3. Stockout exclusion narrows the estimand and biases downward.
4. Clustering is at the listing level; substitutes and regions remain correlated.
5. DiD passed its pre-trend test and was still 21 points out.
6. **ROI on this dataset reflects the generator's spend scale, not promotional
   economics.**
7. The sensitivity sweep did not complete in 50 minutes at 300 pairs; it is
   opt-out and the report says which mode produced a number.
8. Synthetic data.

## 22. Databricks migration

[`databricks_migration.md`](databricks_migration.md). Design only. Recommended
pattern: **distributed feature engineering, single-node estimation**. The riskiest
part of the port is the anchoring window — `ROWS BETWEEN 7 PRECEDING AND 1
PRECEDING` — where an off-by-one silently puts treatment effect inside a
covariate.

## 23. Future agent integration

`PromoUpliftTool` is registered and returns **evidence, not a number**. Its
description tells the agent not to compute uplift itself, because the arithmetic
is trivial and the answer is wrong by ~58 points. A `failed` validation is
promoted to the first warning.

**No Claude, no LangGraph, no supervisor, no A2A, no optimiser** were introduced.

## Commands

```powershell
uv run python scripts/estimate_uplift.py --synthetic --all-scenarios
uv run python scripts/estimate_uplift.py --validate-ground-truth --sample-pairs 300
uv run python scripts/estimate_uplift.py --sample-pairs 150 --seed 42 --no-sensitivity
uv run ari uplift --promotion PR0000123
uv run ari uplift-quality
uv run ari uplift-validate
uv run pytest tests/promo_uplift -v
```
