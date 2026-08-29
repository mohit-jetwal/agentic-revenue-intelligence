# Control definition

A control observation has one job: to stand in for what the treated observation
*would* have been. Every way of choosing controls is a claim about
comparability, and it is usually the weakest link in a causal estimate — weaker
than the estimator, weaker than the covariates, weaker than the sample size.

## Two pools, because they fail differently

### Within-series controls

Unpromoted days from the **same product in the same store**, within
`same_series_window_days` (default 45) of the event.

Product identity, store identity, shelf position, local competition and customer
mix are held fixed *exactly* rather than adjusted for. That removes whole
families of confounders without modelling any of them, including confounders
nobody has thought of.

**How it fails**: temporally. The control days are at a different point in the
season, and if that difference is what drove the promotion decision it is also
what drives the sales gap. The ±45-day fence limits how far apart they can be.

### Cross-sectional controls

Days from **never-promoted listings in the same category and region**,
restricted to the treated period.

Contemporaneous, so seasonality, weather and any market-wide shock are shared
between the arms rather than needing adjustment.

**How it fails**: compositionally. A listing that never gets promoted is usually
different in kind — slower, more niche, worse distribution — and those
differences are exactly what the propensity model then has to carry. The
synthetic generator reproduces this deliberately: under confounding, the
never-treated listings are the lowest-volume ones, because that is who gets
passed over in practice.

Neither pool is sufficient alone. Both are built, and the balance diagnostics in
[`validation.md`](validation.md) are not optional decoration.

## What is excluded from both pools

| Excluded | Why |
|---|---|
| **Washout rows** | Depressed *by* the treatment. Using them deflates the baseline and inflates uplift — the most common way pull-forward becomes apparent incrementality |
| **Sub-threshold promotions** | A 2% discount is not a clean no-promotion observation |
| **Stockout rows** | Censored outcomes. See [`assumptions.md`](assumptions.md#stockouts) — this one is a genuine compromise |
| **Rows without complete history** | No trailing covariates means no adjustment set. Dropped, not imputed |

## Sufficiency, and refusal

| Threshold | Default | Behaviour below it |
|---|---|---|
| `min_control_rows` | 30 | `NoControlGroupError`, recoverable |
| `min_treated_rows` | 5 | `NoControlGroupError`, recoverable |

These are not power calculations. They are floors below which an estimate is
arithmetic rather than inference: five treated rows and twenty controls will
happily produce a point estimate and a confidence interval, and both will be
meaningless.

The refusal is **recoverable** and says what would have worked — widen
`same_series_window_days`, enable cross-sectional controls, or request a longer
date range. A recoverable error that leaves an agent able to conclude only "it
failed" is not enough to re-plan on.

## When there is genuinely no control group

Real cases, all of which produce a refusal rather than a number:

- A product promoted continuously for the whole window has no unpromoted days of
  its own.
- If every store ran the same promotion, there is no cross-sectional control.
- A newly listed product has no pre-period.

A listing promoted on more than 95% of days triggers the `always_treated_series`
quality check: its propensity approaches 1, so it contributes an unbounded
weight, and overlap trimming removes it. That **changes the estimand** to
"listings that were sometimes not promoted", which is reported rather than
absorbed.

## Matching, as a second opinion

Nearest-neighbour matching on the **logit** of the propensity score, with a
caliper of 0.2 standard deviations.

The logit scale matters: on the probability scale the distance from 0.01 to 0.02
looks identical to 0.50 → 0.51, but the first pair differs by a factor of two in
odds and the second by a few percent.

Matching and weighting fail differently, which is the point of having both.
Weighting keeps every observation but can hand enormous influence to a handful.
Matching discards unmatched treated units — changing the estimand to "promotions
that had a comparable control" — but every retained pair is concretely
comparable and inspectable. Agreement between the two is evidence; disagreement
is a finding worth chasing.

**Matching does not manufacture ignorability.** Two store-days identical in
trailing demand, price and season may still differ in something nobody recorded.
Matching improves comparability on observables and does nothing else.

## Configuration

```yaml
controls:
  same_series_window_days: 45
  use_cross_sectional_controls: true
  min_control_rows: 30
  min_treated_rows: 5
  pre_period_days: 56
```

`pre_period_days` must be at least 56 — the longest trailing covariate window —
or every treated row is dropped for an incomplete adjustment set. The config
refuses a smaller value rather than silently shrinking the covariate set.
