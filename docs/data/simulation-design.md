# Simulation Design

How the synthetic dataset is generated, and why it is built this way.

## The problem this solves

If each column were sampled independently, then in Steps 4–11 there would be no
way to tell a model that works from one that doesn't. Every elasticity estimate
would be equally unfalsifiable, and the honest answer to *"how do you know your
model is right?"* would be *"I don't."*

So the data comes from a **structural causal model with hidden ground-truth
parameters**. Elasticities, cross-price coefficients and promotion response
curves are drawn *first*, written to `data/local/ground_truth/`, and only then
used to simulate sales. Every relationship the platform later claims to
*estimate* has a known correct answer recorded before the data existed.

Two properties follow, and they are the point of the whole exercise:

1. A correctly specified estimator can be shown to **recover** the true value.
2. A naively specified one can be shown to be **wrong**, in a reproducible way.

The second matters as much as the first. If an uncontrolled regression recovered
truth just as well, the data would be too easy and Step 8's careful
specification would be theatre.

## Demand equation

For product `p` in store `s` on day `t`:

```
log λ[p,s,t] = log base_demand[p] + log store_scale[s] + region[s] + channel[s]
             + trend[p,t] + annual_season[cat(p),t] + dow[t] + holiday[t] + festival[t]
             + β_own[p,s]  · log(price[p,s,t] / ref_price[p,s])
             + Σ_j β_cross[p,j] · log(price[j,s,t] / ref_price[j])
             + promo_lift[p,s,t] + pull_forward[p,s,t]
             + γ[p] · log(comp_price[p,t] / comp_ref[p])
             + launch_ramp[p,t] + scenario_shock[s,t]
             + ε[p,s,t]

latent_units   ~ NegativeBinomial(mean = exp(log λ), dispersion[p])
observed_units = min(latent_units, available_inventory)
```

### Why log-additive

It makes a log-log panel regression the *correctly specified* estimator, so
recovery can be demonstrated rather than asserted. It also makes every effect
compose multiplicatively on the demand scale, which is how retail demand
actually behaves — a 20% promotional lift means 20% more of whatever the
seasonal and price-adjusted base happened to be, not a fixed number of units.

### Why competitor price enters through its own reference

Written as `γ · log(comp_price / comp_ref)`, not `γ · log(comp_price / our_price)`.
The ratio form smuggles a second own-price term into the equation and
contaminates `β_own` — the coefficient Step 8 is trying to recover would no
longer be the parameter that was drawn.

### Why negative binomial rather than Gaussian noise

Sales are counts. Real POS data is over-dispersed relative to Poisson, and slow
movers genuinely sell zero units on many days. Gaussian noise on a float erases
the zero-inflation and makes forecasting look easier than it is. Implemented as
a gamma-Poisson mixture: draw a gamma-distributed rate, then a Poisson count.

### Why censoring is kept separate

`latent_units` is the demand that existed; `observed_units` is what the till
recorded. Both are generated, but only the latter reaches the analytical tables —
latent demand lives in ground truth. That gap is what lets Step 4 be tested on
whether it can tell a **demand collapse** from a **supply failure**, which is the
distinction the Root Cause agent must make in Step 17.

## The confounders

Deliberate, each with a documented cure. Without them, recovering elasticity
would be arithmetic.

| # | Confounder | Mechanism | Effect on naive estimation | Cure |
|---|---|---|---|---|
| 1 | Price endogeneity | Managers price into anticipated seasonal demand (`pricing.endogeneity_strength`) | Biases elasticity toward zero | Store + time fixed effects |
| 2 | Cost pass-through | A commodity cost index shifts price but not demand | — | It *is* the instrument: valid for 2SLS |
| 3 | Randomised price tests | A fraction of changes are exogenous, tagged `price_change_reason = randomised_test` | — | A clean identification subset |
| 4 | Promotion targeting | Promos skew toward products with softening baselines (`promotions.targeting_strength`) | Overstates uplift | Proper control group |
| 5 | Competitor correlation | Our price and theirs both track the cost index | Flips the sign of the competitor effect | Control for own price |
| 6 | Stockout endogeneity | Stockouts happen *because* demand spiked | Censoring is non-random | Exclude censored rows |
| 7 | Pull-forward | Post-promotion pantry dip (`promotions.pull_forward_fraction`) | Naive during-vs-before overstates incrementality | Baseline model + post-period |

### Measured effect

`data/local/validation_report.md` carries the authoritative numbers for whatever
was last generated. From the **dev** profile (seed 42):

| Measure | Result |
|---|---|
| Panel FE, store + month effects | **7.1%** median relative error vs true elasticity |
| Naive pooled OLS | **28.4%** median relative error — 4× worse |
| Cross-price sign agreement (own price controlled) | **7/7** |
| Competitor effect, own price controlled | **+1.21** (correct sign) |
| Stockout suppression of observed sales | **71.6%** inside injected windows |
| Censoring when in stock | **0.0%** — the gap comes only from stockouts |

Individual recoveries at dev scale, for a sense of the spread: true −0.6810 →
−0.6742 (1.0% error); true −1.7988 → −1.7927 (0.3%); true −1.1949 → −1.5173
(27%). Median matters more than any single product — a handful will always be
poorly identified because their price happened to move little.

**On bias direction.** The naive estimator is *not* uniformly attenuated toward
zero, which is what price endogeneity alone would predict. Promotional
confounding — a price cut coinciding with an additive uplift — pushes it away
from zero, and usually dominates. The two pull in opposite directions and the
net varies by product, so the test asserts the *magnitude* of the error rather
than a single direction. Worth knowing before claiming "attenuation bias" in an
interview: here it is really "confounding bias, direction product-dependent".

**Profile sensitivity.** Smoke and dev give different magnitudes. Smoke has
fewer products with more observations each, so its recovery is tighter (~6%);
dev spreads observations across 300 products (~7%). Both comfortably clear the
35% tolerance, but the numbers are not interchangeable between profiles.

## Promotion response

```
promo_lift = a[p,type] · (1 − exp(−b · discount)) · store_promo_responsiveness
```

Additive in log space, so multiplicative on demand, with diminishing returns by
construction. Calibrated so 10% ≈ 12% uplift, 20% ≈ 22%, 30% ≈ 28% (brief §15).

Without saturation, Step 7's optimiser would pour the entire budget into the
single deepest discount available — both wrong, and obviously wrong to any
category manager.

**Pull-forward** follows each promotion: an exponentially decaying negative term
for ~10 days, sized as a fraction of the incremental units. This is *why* naive
during-vs-before uplift overstates incrementality, and it gives the Step 18
Critic something real to catch.

## Customer segments reach a panel with no customer axis

Each store draws a Dirichlet segment mix around the national shares. A store's
effective price sensitivity is the mix-weighted average of its segments'
`price_sensitivity`, and `β_own[p,s] = true_elasticity[p] × store_sensitivity[s]`.

So a Value-heavy catchment really is more price-sensitive, and Step 8 has genuine
store-level heterogeneity to model rather than one global elasticity per product
with noise sprinkled on top. Transactions then draw customers according to the
same mix, so segment analysis in BI agrees with the demand simulation.

## Inventory

An order-up-to `(s, S)` policy on **inventory position** (on-hand + on-order),
with a lead time. Ordering against on-hand alone re-triggers every day of the
lead time and stacks duplicate orders — a bug that inflated stock to ~170 days
of cover during development and made stockouts essentially impossible.

`opening + received − sold = closing` holds exactly, by construction, and is
asserted without tolerance in the validation suite.

Supply failures suppress **deliveries**, not stock already on the shelf. That is
physically right — a distribution failure means goods stop arriving, so the store
sells down what it has and only then runs dry — and it keeps the reconciliation
identity intact.

## Scenarios

Injected into the **driver matrices before demand is simulated**, never by
editing sales afterwards. A price scenario raises the price path; a stockout
scenario throttles the supply cap. The effect then propagates through the same
causal chain as everything else.

That ordering has a useful consequence: Scenario A (price increase) automatically
produces Scenario F (substitutes gain) and Scenario G (complements lose) through
the cross-price terms, without either being separately fabricated.

| ID | Scenario | Implementation |
|---|---|---|
| A | Price increase | Regular price path raised for a window |
| B | Successful promotion | Shallow discount, positive ROI |
| C | Bad promotion | Deep discount; uplift positive, margin destroyed |
| D | Stockout | Deliveries throttled; latent demand untouched |
| E | Competitor price cut | Competitor series reduced |
| F | Substitution | Emerges from A via cross-price |
| G | Complement | Emerges from A via cross-price |
| H | Regional decline | **Distribution loss**, not demand loss |
| I | Seasonal peak | Festival multipliers (registered, not re-injected) |
| J | Product launch | Saturating launch ramp (registered, not re-injected) |

**Scenario H deserves emphasis.** It is implemented as stores in the region
losing distribution outright, plus a *mild* residual demand softening. Both look
like "North is down" in a summary report; only one is a demand problem. An agent
that concludes "demand collapsed in North" has got it wrong, and the data can
prove it.

Every scenario is registered in `ground_truth/scenario_config.json` with its
exact products, stores, window and expected direction. That registry is also the
seed for the Step 21 golden evaluation set — *"why did Product X decline in
November?"* has a known correct answer precisely because it was injected here.

## Reproducibility

`numpy.random.SeedSequence` spawned per entity family, so adding a store
perturbs the store stream and nothing else. Chunk generators are derived from an
explicit `spawn_key = (stream, chunk)` rather than by calling `spawn()` on the
parent — the latter mutates a child counter, which would make the data depend on
iteration order and quietly turn `output.chunk_months` into a data-changing
setting rather than a performance knob.

Same seed and config ⇒ byte-identical output, asserted by hashing every Parquet
file in the gold layer.

## Known limitations

- **Basket composition is not modelled.** Transactions are disaggregated from the
  daily panel, so cross-product baskets are not correlated within a transaction.
  Market-basket analysis would need a different generator.
- **No competitor reaction function.** Competitors respond to the shared cost
  index but not to *our* price moves. Genuine price-war dynamics are out of scope.
- **Cross-price effects are within-store and contemporaneous.** No lagged
  substitution or stockpiling across products.
- **Customer-level loyalty dynamics are absent.** Segments are static; there is
  no churn, no lifecycle, no state dependence in individual purchase behaviour.
- **`latent_demand` is stored at full panel grain**, which makes ground truth
  roughly the size of the sales fact. Acceptable locally; at stress scale it is
  the largest artifact on disk.
