# Treatment definition

An uplift number is uninterpretable without this document. "+18%" measured over
the event window and "+18%" net of pull-forward are different quantities, and two
analysts using different definitions will disagree about the number and both be
right.

The definition is therefore **configuration, not code** — it lives in
[`configs/models/promo_uplift.yaml`](../../configs/models/promo_uplift.yaml),
is hashed into every MLflow run, and is attached as a sentence to every result
the service returns.

## The definition

> Treated = a promotion of any mechanic with depth ≥ 5% running at least 2 days.
> Effects are measured over the event window and a 10-day washout. Control =
> unpromoted days from the same listing plus never-treated listings in the same
> category and region, within 45 days of the event and outside its washout
> window. Estimand: ATT. Rows censored by stockout are excluded, so the estimand
> is the effect on sales among days where stock was available.

## Unit of analysis

`date × product_id × store_id` — the same grain as the sales fact, the baseline
model and the forecaster. A duplicated key is refused rather than deduplicated:
where duplicates disagree on promotion status the treatment indicator is
genuinely ambiguous, and an ambiguous treatment makes the estimand undefined
rather than merely imprecise.

## What counts as treatment

`promotion_id` is non-null and the event satisfies:

| Rule | Default | Why |
|---|---|---|
| `min_discount_depth` | 0.05 | Screens out trivial price noise being read as a promotion |
| `min_duration_days` | 2 | A one-day price glitch is not a promotion; its "uplift" is one day of noise |
| `include_types` | all | Restrict to compare mechanics |
| `require_price_reduction` | false | A display or bundle with no price cut is still a promotion — it changes demand without changing price |

**Treatment is the whole event, price cut included.** This is the decision that
most affects the number. A promotion moves demand through two channels, and on
this data the price cut is 2.6× the mechanic — see
[`business_objective.md`](business_objective.md#what-caused-by-the-promotion-includes).
Defining treatment as "the promotion flag" and controlling for discount measures
the smaller half.

**Sub-threshold promotions are excluded from *both* arms.** A 2% discount is not
treatment, and it is not a clean "no promotion" observation either. Leaving it in
the control pool would depress the comparison baseline slightly and inflate
uplift.

## The three windows

| Window | Days | Role |
|---|---|---|
| **Treatment** | `start .. end` | Gross uplift |
| **Washout** | `end+1 .. end+10` | Pull-forward payback. **Neither arm** |
| **Net** | both | Net incrementality |

Washout rows are depressed *by* the treatment. Counting them as controls would
deflate the comparison baseline and inflate uplift — the exact error the
pull-forward term exists to expose. They get their own role and are excluded from
both arms.

A washout window that runs into the next promotion yields to it: those days are
treated, not washout. Calling them "recovery from the previous promotion" would
attribute one promotion's lift to another's payback.

The config refuses a control window that does not clear the washout, because
control rows drawn from the pull-forward dip are depressed by the very promotion
whose effect they are supposed to anchor.

## Overlapping promotions

The generator forbids two promotions on one listing on one day
(`promotion_generator.py:141`). The code checks rather than trusts: a real
promotion feed would not be so disciplined, and simultaneous promotions make
uplift unattributable between them. The data-quality check
`overlapping_promotions` is a hard **FAIL**.

## Estimand: the ATT

The average effect on the promotions that actually ran, not on a hypothetical
world where everything is promoted.

It is the question a category manager asks, and it needs overlap only on the
treated support — a materially weaker requirement than the ATE, which would need
every unpromoted store-day to have had a plausible chance of promotion.

## Configuration

```yaml
treatment:
  min_discount_depth: 0.05
  include_types: []
  require_price_reduction: false
  washout_days: 10
  min_duration_days: 2
```

Changing any of these changes **what the number means**, not how well it is
estimated. `PromoUpliftConfig.fingerprint()` hashes the whole file into MLflow
params so two runs can be told apart, and `treatment_definition()` renders the
sentence above so a reader knows *what* changed rather than only *that*
something did.
