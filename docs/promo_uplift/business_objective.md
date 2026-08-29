# What promo uplift is, and why the obvious answer is wrong

## The question

> How much additional sales did this promotion generate, compared with what would
> have happened without it?

Roughly ₹100M of trade spend a year runs through this platform's promotion
calendar. Reallocating it requires a defensible incremental number per event, and
the number most organisations use is wrong in a knowable direction.

## Four quantities people conflate

| Quantity | Definition | Where it comes from |
|---|---|---|
| **Observed sales** | What the till recorded during the promotion | The `sales` fact |
| **Baseline sales** | What would have sold *without* the promotion | Estimated. Never observed |
| **Incremental sales** | Observed − baseline | The causal effect |
| **Uplift %** | Incremental ÷ baseline | The effect, expressed as a rate |

The second row is the whole problem. Baseline sales for a promoted store-day do
not exist anywhere and never will — the promotion ran, so the world where it did
not is unobservable. Every method in this package is a different way of
constructing it.

## The numerical example

A store sells **1,000 units/week** normally. During the promotion week it sells
**1,600**. The four weeks before averaged **1,150**, because a seasonal peak was
building.

| Quantity | Units | What it is |
|---|---|---|
| Observed | 1,600 | What the till recorded |
| **Naive** (during − before) | **+450** | Wrong |
| Baseline (no-promo counterfactual) | 1,320 | The seasonal peak, without the promotion |
| **Gross incremental** | **+280** | Effect during the promotion |
| Pull-forward payback | −90 | The next fortnight's dip |
| **Net incremental** | **+190** | What Step 8 must optimise |

**The naive number is 2.4× the truth.** Two errors compound:

1. **The pre-period is the wrong baseline.** Promotions are scheduled into
   rising demand. The generator does this explicitly —
   `promotion_generator.py:128` draws start days with weights
   `exp(targeting × 2 × seasonal)` — and real merchandisers do it for the same
   reason: you promote when shoppers are already in the aisle. So the weeks
   before a promotion are systematically weaker than the promotion week would
   have been anyway, and the comparison books the season as promotional lift.

2. **Pull-forward is outside the window.** Buyers load their pantry and buy less
   afterwards. `pull_forward_fraction: 0.32` over ten decaying days. The naive
   window ends before the dip, so borrowed volume is counted as new.

Both errors point the same way: **up**.

## What "caused by the promotion" includes

A promotion is not just a mechanic. It is a mechanic **and a price cut**, and in
this data the price cut is the larger half.

`sales_generator.py:334-341` shows both channels entering log demand:

```
log λ = ... + beta_own · log(selling_price / ref) + promo_lift + pull_forward + ...
                         └── selling_price = regular_price × (1 − discount)
```

Measured on 4,417 real promotion events from the generator's own recorded
parameters:

| Channel | Contribution |
|---|---|
| Mechanic (display, BOGO, bundle) | +17.7% |
| **Price cut, via own-price elasticity** | **+45.6%** |
| **Combined** | **+71.3%** |

The price channel is **2.6× the mechanic**. Any method that conditions on
discount — which is the natural thing to do, since discount looks like an
obviously relevant covariate — measures only the mechanic and reports it as the
whole effect. That is why `discount_percentage` and `selling_price` are in
[`POST_TREATMENT_FEATURES`](../../ml/promo_uplift/features.py) rather than in the
adjustment set. See [`causal_methodology.md`](causal_methodology.md) on mediators.

## Gross versus net, and why both are reported

| Window | Days | Estimand |
|---|---|---|
| Treatment | `start..end` | **Gross uplift** — effect during the event |
| Washout | `end+1 .. end+10` | Pull-forward payback |
| Net | both | **Net incrementality** |

Reporting only gross is how promotions get renewed that never paid back. Net is
the number a budget decision needs, because a promotion that moved a fortnight's
sales forward by a week grew nothing.

## Why the number has to be causal

Incremental profit is what Step 8 allocates against:

```
incremental_profit = incremental_units × promotional_unit_margin − promotion_spend
roi                = incremental_profit / promotion_spend
```

An uplift figure inflated by 2.4× makes almost every promotion look profitable,
because the inflation multiplies the numerator while spend stays fixed. A
category manager acting on it will keep funding the promotions that destroy the
most value — they are the ones with the biggest discounts, which get the biggest
seasonal targeting and the biggest pull-forward, and therefore the biggest naive
overstatement.

## What this model does not answer

- **Cannibalisation.** A promotion takes volume from its substitutes. This model
  measures the promoted SKU only, so its profit figure is an **upper bound** on
  category profit. Cross-price effects arrive in Step 9.
- **What price to set.** That is Step 10.
- **How to allocate the budget.** That is Step 8, which consumes this output.
- **What will happen next.** That is [forecasting](../forecasting/README.md),
  which is a different question with a different answer — see
  [`causal_methodology.md`](causal_methodology.md#prediction-versus-causal-inference).
