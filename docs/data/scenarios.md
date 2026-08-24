# Business Scenarios A–J

Brief section 19 requires the dataset to contain **identifiable** situations, not
just plausible noise. Each is injected at named products, stores and windows, and
registered in `data/local/ground_truth/scenario_config.json` with its expected
direction.

That registry does double duty: Step 2's validation confirms each scenario is
actually visible in the data, and it seeds the Step 21 golden evaluation set —
*"why did Product X decline in November?"* has a known correct answer precisely
because it was injected here.

## How injection works

Scenarios mutate the **driver matrices before demand is simulated**, never the
output afterwards. A price scenario raises the price path; a stockout scenario
throttles the supply cap; demand then responds through the same causal chain as
everything else.

Two consequences worth noting:

- Effects propagate automatically. Scenario A produces F and G through the
  cross-price terms without either being separately fabricated.
- Effects are censored by inventory like any other demand, so an injected price
  rise during a stockout behaves the way it really would.

Scenario products are chosen from the **well-observed** half of the catalogue. A
scenario applied to a SKU stocked in two stores would be invisible under noise,
and the resulting validation failure would be about sample size rather than about
the generator.

---

## A — Price increase

Regular price raised by `scenarios.price_increase.magnitude` (default 8%) for
~120 days. Focal products are preferentially chosen from those with declared
relationships, so F and G have something to act on.

**Expected:** own demand falls. Revenue direction depends on whether |elasticity|
exceeds 1 — which is exactly the question Step 10 must answer.

## B — Successful promotion

15% discount for ~21 days across the product's stores.

**Expected:** clear incremental uplift, positive ROI. The discount is shallow
enough that incremental margin exceeds spend.

## C — Bad promotion

38% discount for ~21 days.

**Expected:** sales rise *and* value is destroyed. Uplift is genuinely positive —
Step 6 should measure it as such — but gross margin falls far enough that ROI is
poor, so Step 7 should decline to fund it. **Measuring uplift and judging ROI are
different questions**, and the data separates them deliberately.

## D — Stockout

Deliveries throttled to ~12% of normal for 45 days across ~55% of the product's
stores.

**Expected:** observed sales fall sharply while **latent demand is unchanged**.

The duration must exceed `inventory.target_cover_days` — the scenario suppresses
deliveries, not shelf stock, so the store sells down what it has before running
dry. A shorter window would leave no observable stockout at all. That delay is
also realistic: a real distribution failure shows up as a gradual decline, not a
cliff.

Measured at dev scale: **71.6%** of latent demand suppressed inside the window,
with `0.0%` gap outside it — confirming censoring is the only source.

## E — Competitor price cut

Competitor price reduced 12% for ~75 days.

**Expected:** our demand falls while our own price is unchanged — a decline that
own-price elasticity alone cannot explain. A root-cause analysis that only looks
at our own pricing will find nothing.

## F — Substitution *(emergent)*

**Expected:** when A's price rises, its declared substitutes gain volume.

Not separately injected. It falls out of A through the cross-price terms, which
is the stronger demonstration — the relationship exists in the causal structure,
not as a hand-placed effect.

## G — Complement *(emergent)*

**Expected:** when a product's price rises, its complements lose volume. Same
mechanism as F, opposite sign.

## H — Regional decline

**Implemented as distribution loss, not demand loss.** ~22% of product-store
listings in the region stop being stocked for ~90 days, plus a *mild* residual
demand softening (−6%) across the remainder.

**Expected:** regional sales fall, driven mainly by lost distribution.

This is the sharpest test in the set. Both a distribution collapse and a demand
collapse look like *"North is down"* in a summary report, and only one is a
demand problem. An agent that concludes "demand collapsed in North" has got it
wrong — and because the mix of causes is recorded, the data can prove it.

## I — Seasonal peak *(registered, not injected)*

Festival multipliers in the base simulation already produce this. Injecting it
again would double-count, so it is registered for validation only.

**Expected:** materially higher demand on festival and pre-festival days.
Measured at dev scale: **+54%** versus comparable non-festival days.

## J — Product launch *(registered, not injected)*

~8% of products launch mid-history and build distribution over a saturating
120-day ramp.

**Expected:** zero sales before launch, then a gradual build rather than a step
to full rate.

---

## Verifying them yourself

```powershell
uv run ari generate-data --profile dev --seed 42
uv run ari validate-data --profile dev
```

Then open `data/local/ground_truth/scenario_config.json` for the exact products,
stores and windows, and query `data/local/gold/` in DuckDB. The scenario tests in
`tests/data/test_scenarios.py` assert each one is present and behaves as
described — including that Scenario C's margin is genuinely worse than B's, and
that Scenario D's lost units are non-zero.
