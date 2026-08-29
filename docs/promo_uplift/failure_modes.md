# Failure modes

The governing rule (brief §33): **never return a confident recommendation when
the causal assumptions fail.**

Everything below either refuses, or returns a number labelled with why it cannot
be called causal. Nothing fails silently.

## Refusals

| Failure | Code | Recoverable | What the caller is told |
|---|---|---|---|
| No control observations | `no_control_group` | yes | Rows found vs required; which config knob widens it |
| Too few treated rows | `no_control_group` | yes | Same |
| Overlap violated after trimming | `assumptions_violated` | yes | Trimmed share vs threshold |
| Pre-period too short | `insufficient_data` | yes | Days available vs required |
| Unknown promotion / product / store | `insufficient_data` | yes | The unknown identifiers |
| No qualifying events | `invalid_treatment` | yes | The definition that excluded them all, and the count |
| Duplicate grain | `invalid_treatment` | yes | The duplicate count |
| No persisted analysis | `model_not_found` | **no** | The command that produces one |
| Singular design / non-convergence | `uplift_failed` | yes | Which estimator |

Recoverable means: **could a different request succeed against the same system?**
A missing analysis is not recoverable — no reformulation helps until someone runs
one. Saying so lets a supervisor stop retrying rather than burning turns.

## Warnings — the estimate is returned, labelled

| Condition | Warning |
|---|---|
| Balance fails for a doubly robust method | Names the covariate and its SMD; says the DR property is now carrying the argument |
| Differential censoring between arms | Reports both rates and the direction of the resulting bias |
| Propensity weights miscalibrated | Control weights sum vs treated count, with the ratio |
| Sensitivity spread > 50% of the estimate | The specification is doing the work |
| ROI below break-even | Value destroyed despite positive uplift |
| Negative incremental units | Not floored; profit and ROI are negative accordingly |
| No cross-sectional controls | Every control is another day of the same listing |
| Missing promotion spend | ROI cannot be computed; profit is before promotional cost |
| Subset of the analysis requested | Diagnostics describe the full run, not the subset |

## The validation verdict

| Status | Meaning | Estimate returned |
|---|---|---|
| `passed` | Every check the method depends on held | yes |
| `warnings` | Something is off; the estimate is still identified | yes |
| `failed` | An assumption the method cannot do without was violated | **yes, labelled** |

A `failed` estimate is returned rather than suppressed. Withholding it does not
protect anyone — the caller can compute a naive number in one line of SQL — so
the choice is between our number labelled and theirs unlabelled.

The tool promotes the failure to the **first** warning, so a supervisor reading a
truncated list cannot miss it. The CLI exits non-zero.

## Method-aware failure

Balance failure is not uniformly fatal, and that is a statistical fact rather
than a convenience:

| Method | Balance fails |
|---|---|
| IPW, matching | **blocks** — nothing but the propensity model |
| AIPW, DR-learner | **warns** — consistent if the outcome model is right |

Measured: on the confounded synthetic panel the worst SMD after weighting was
0.38, far past the 0.10 threshold, and AIPW recovered +65.2% against a true
+63.3%.

## Estimator-level failures

One estimator failing is information, not a fatal error. Each is attempted
independently and the reason for any absence appears in the report:

- **`baseline_counterfactual`** without a trained Step 5 model — the other five
  do not depend on it.
- **`difference_in_differences`** with an empty cell, no pre-period, or only one
  group in the pre-period.
- **`dr_learner`** where the pseudo-outcome regression fails.

A pipeline that ran only the method it intended to use would produce the same
headline with none of the argument behind it.

## Bugs this code has already caught

Each of these was found by a diagnostic that now ships, not by inspection.

**Time-block cross-fitting → −424% against a true +65%.** Contiguous date folds
meant every fold was predicted by a model trained on other periods, so the linear
`time_index` was extrapolated; propensity scores hit the clip boundaries and
control weights summed to **43×** the treated count. Fixed by holding out whole
listings. **The permanent guard**: since `E[(1−T)·e/(1−e)] = P(T=1)`, a weight
ratio outside [0.7, 1.4] now raises a warning.

**Unstabilised weights → balance overshoot.** One control row at `e = 0.98`
received weight 49 while the 99th percentile sat below 1. Balance went from +0.27
before weighting to **−0.38 after** — worse, in the opposite direction. Fixed by
capping at the 99th percentile.

**i.i.d. standard errors → intervals 3–5× too narrow.** Coverage of the known
truth was **4/6** while point estimates were within 2–5 points throughout. Rows
within a listing are strongly serially correlated. Fixed by clustering on the
listing; coverage went to **6/6**.

**Heavy-tailed DR pseudo-outcome → +91% against a true +63%.** The score carries
a `1/e` factor, so an L2 regression fitted largely to its extremes. Fixed by
winsorising at the 1st/99th percentile.

**Wrong bias direction in a quality check.** `missing_treatment_label` claimed
uplift was *overstated*. A promoted day misfiled as control carries its lift into
the control arm, raising the baseline and **understating** uplift. Caught by a
test that asserted the direction rather than the presence of a warning.

**Distance-to-event NaN inside a window.** Both `merge_asof` passes returned NaN
for days in the middle of a promotion, which reads as "never treated". Harmless
in production — the function is only called on control rows — and wrong as
documented. Caught by a test.

## Not handled

| Gap | Consequence |
|---|---|
| Unmeasured confounding | Untestable. No sensitivity bound is implemented; the gap is declared |
| Cannibalisation | Profit is an upper bound on category profit. Step 9 |
| Overlapping promotions | Refused as a data-quality FAIL rather than modelled |
| Treatment contamination | If a "control" store saw the promotion's advertising, no diagnostic detects it |
| Data drift | The analysis is a snapshot; nothing monitors whether it has gone stale |
