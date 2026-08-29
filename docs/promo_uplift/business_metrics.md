# Business metrics

An uplift percentage settles nothing. "+18% incremental volume" is compatible
with a promotion that made money and one that lost a fortune, and the difference
is entirely in the margin and the spend.

## The arithmetic

```
incremental_units   = ATT × treated_days
incremental_revenue = incremental_units × promotional selling price
incremental_margin  = incremental_units × promotional unit margin
incremental_profit  = incremental_margin − promotion_spend
roi                 = incremental_profit / promotion_spend
```

## Four decisions that change the answer

### 1. Margin is taken at the promotional price

The incremental units were sold at a discount, so they earn the discounted
margin. Valuing them at full margin is the most common way a losing promotion is
reported as a winner, and on a deep discount the two differ by more than the
uplift being measured.

Margin comes from `revenue − cost` in the sales fact rather than a configured
constant, so it reflects the price actually paid on each day. The configured
`default_gross_margin` is a fallback for when cost is genuinely absent, not a
default assumption.

### 2. Revenue is incremental, not total

The promotion did not "generate" the sales that would have happened anyway.
Reporting total promoted revenue against promotional spend is the other common
inflation, and it makes almost every promotion look profitable.

### 3. Spend is summed over events, not rows

Spend is an event-level fact. Broadcasting a per-event total across every day of
a twenty-day window and then summing would multiply it by twenty — and ROI would
come back at a twentieth of its true value, which looks plausible enough to
survive review.

### 4. ROI is `None` when spend is zero

Not infinity, not zero. A display-only mechanic with no recorded spend has no
return on investment. It has an incremental profit, which is reported.

## Negative uplift is never floored

| Cause | Mechanism |
|---|---|
| Cannibalisation | A promoted substitute took the volume |
| Poor design | Discount too shallow to change behaviour, mechanic unattractive |
| Stockout | Demand was created and could not be served |
| Wrong timing | Clashed with a competitor's deeper promotion |
| Brand damage | Deep discounting cheapened a premium SKU |
| Measurement error | Genuinely possible, and why the interval is reported |

The `negative` synthetic scenario exists to prove the estimator returns these:
true −9.4%, estimated −5.9%, sign preserved. An estimator that cannot report a
negative effect is useless for allocation, because the promotions worth cutting
are precisely the ones it would hide.

## Value-destroying promotions

Positive uplift and negative ROI is the common case worth surfacing:

```
incremental_units  = +36        the promotion worked
incremental_margin = 2,100
promotion_spend    = 4,700
incremental_profit = −2,600     it still lost money
roi                = −0.55
```

These are kept in the event table and ranked last rather than filtered out. The
optimiser's job in Step 8 is partly to allocate *away* from them, which it cannot
do if they are missing.

## Uncertainty propagates

When the effect estimate has a measured interval, the profit and ROI bounds come
from it:

```
profit_lower = ci_lower × treated_days × unit_margin − spend
profit_upper = ci_upper × treated_days × unit_margin − spend
```

Where the effect has no interval, the profit has none either. A point estimate of
profit with an invented band is worse than a point estimate labelled as one.

## The declared gaps

Every `BusinessImpact` carries these:

| Assumption | Consequence |
|---|---|
| Margin at the realised promotional rate | Stated per result, with the rate |
| Revenue and profit are **incremental**, not total | Sales that would have happened anyway are excluded |
| **Cannibalisation is not deducted** | Category profit is lower. **This figure is an upper bound** |
| Pull-forward handled by the estimand | Net includes the dip; gross does not |

Cannibalisation is the one that matters most. A promotion takes volume from its
substitutes, this model measures the promoted SKU only, and Step 9's cross-price
elasticities are what will let it be subtracted.

## Segment classification

Segments are labelled by what should be done about them, not ranked:

| Label | Meaning | Action |
|---|---|---|
| `high_uplift` | Top tercile, interval excludes zero | Candidate for more investment |
| `low_uplift` | Positive but thin | Works — check margin before scaling |
| `negative` | Effect below zero | Stop |
| `uncertain` | Interval spans zero, or too few treated rows | **Measure before acting** |

Four labels rather than a ranking because the actions differ in kind. A negative
segment should be stopped; an uncertain one should be measured. Collapsing both
into "low" invites the same decision for two different problems.

Segments below `min_treated` return a **null** effect rather than a number. A
segment-level uplift computed from eight promotions is a rounding error with a
label on it, and Step 8 would allocate budget against it.

## A caveat about ROI on the current dataset

On the shipped synthetic data, **ROI is not interpretable** and the numbers
should not be read as a finding about promotions.

`configs/data/dev.yaml` draws `fixed_spend` from ₹15,000–220,000 per event,
scaled to a business-wide `annual_budget` of ₹100M — but each event here is a
single product in a single store. Measured across 1,995 events:

| | Median | Aggregate |
|---|---|---|
| Spend | ₹118,464 | ₹234.5M |
| Incremental margin | ₹5,934 | ₹26.5M |
| **Ratio** | **0.05** | **0.11** |

Spend is ~20× the achievable margin at this grain, so 96.8% of events come back
value-destroying and ROI bunches between −1.0 and −0.84. That tight bunching is
the tell: a real mix of good and bad promotions would spread out.

The arithmetic here is correct and tested. The spend feed is at the wrong grain.
Step 8 needs either event-level spend that matches the event-level effect, or
spend apportioned across every listing a trade promotion actually covers.

## The event table Step 8 consumes

One row per promotion, ranked by ROI:

| Column | Purpose |
|---|---|
| `promotion_id`, `product_id`, `store_id` | Identity |
| `treated_days` | Duration |
| `incremental_units` | From the CATE model, per event |
| `incremental_revenue`, `incremental_margin` | At promotional prices |
| `promotion_spend` | The cost |
| `incremental_profit` | The objective to maximise |
| `roi` | The efficiency measure |
| `value_destroying` | `incremental_profit < 0` |

This is what the future optimiser allocates against, subject to budget, frequency
and margin constraints. See
[`databricks_migration.md`](databricks_migration.md) for how it moves.
