# Explaining the forecasting system

Fifteen questions, answered in plain language. The point is not to memorise the
answers but to be able to reconstruct the reasoning — an interviewer will follow
up, and a memorised answer collapses on the second question.

---

## 1. What problem does the forecasting model solve?

**"How much will Product A sell in Store X over the next four weeks?"**

That number drives real decisions: how much to order, how much shelf space to
give it, whether a promotion is worth running, what to price it at. Get it wrong
high and you hold stock that gets marked down. Get it wrong low and you stock out
and lose the sale — and, in this dataset, stockouts happen *because* demand
spiked, so under-forecasting is self-reinforcing.

It also feeds the rest of the platform. Promo uplift measures against a
counterfactual; trade-promotion and price optimisation start from a forecast;
scenario simulation perturbs one. A biased forecast does not fail loudly — it
propagates the same distortion into every one of those.

## 2. How is a forecast different from the baseline model?

This is the distinction worth being crisp about, because conflating them is the
most common way uplift analysis goes wrong.

- A **baseline** asks *what would have happened without the intervention?* It
  deliberately excludes promotions.
- A **forecast** asks *what will happen given the plan?* It deliberately
  **includes** planned promotions.

Same machinery, opposite counterfactual. If you use a forecast as a baseline, you
measure the promotion against a number that already contains it — and report
roughly zero uplift.

There is also a technical difference that drove the whole design. The baseline
predicts units at date *D* using features at *D*, including yesterday's sales.
That is legitimate for a historical counterfactual, where yesterday is known. It
is **invalid for forecasting**: standing at as-of *T* predicting *T+30*, you do
not know *T+29*'s sales.

## 3. How did you actually structure the forecasting problem?

**Direct multi-step with the horizon as a feature.** One training row is:

> *(origin t, horizon step h, target = units at t + h)*

Every feature is placed by asking one question — **is this knowable at t?**

- **Origin side**: lags, rolling means, competitor price. These come from
  `sales_daily` and `competitor_pricing`, which are *observed* data and so
  clamped to the as-of date.
- **Target side**: calendar, planned promotion, planned price. These come from
  tables that are *known in advance* — the promotion calendar for next month
  already exists — so reading them forward is legitimate rather than leakage.

Target-side columns carry an `h_` prefix so the split is visible in the
feature-importance table, where a misplacement would otherwise hide.

## 4. Why one global model rather than one per series?

There are **6,128 real product-store series** in this dataset, and a real
retailer has orders of magnitude more.

- **Practically**: 6,128 models is 6,128 things to fit, store, version, monitor
  and debug, each on about a thousand observations.
- **Statistically**: a pooled model **shares strength**. A slow-moving SKU borrows
  the seasonal shape its whole category exhibits. Fitted alone, it would be
  learning that shape from very little data.
- Product and store identity enter as *features*, so the model can still
  differentiate — it just is not forced to learn each series from scratch.

The same reasoning is why per-series SARIMA is not used at that grain. And note
the honest version of the scalability claim: ETS fits in ~0.25s per series, so one
pass over all 6,128 is about 25 minutes — expensive, **not infeasible**. The real
argument is appropriateness: at product-store grain those series are sparse counts
against a 35% noise floor, and ETS scores 48.2% there against the global model's
43.8%. It is fitted at aggregate grain because that is where it is *correct*.

## 5. Why LightGBM (and XGBoost)?

- **Tabular data with mixed types** — lags, prices, flags, categoricals. Gradient
  boosting is the reliable default; deep learning buys little here.
- **Native categorical handling** in LightGBM, so no one-hot explosion.
- **Poisson objective**, which matters more than the library choice. Sales are
  over-dispersed counts. The obvious alternative — fit on `log1p` and
  back-transform with `expm1` — introduces **retransformation bias**: by Jensen's
  inequality `E[exp(X)] ≠ exp(E[X])`, so the back-transformed mean comes out
  systematically low. A forecast biased low manufactures uplift on every
  promotion measured against it.
- **Fast enough to iterate**, which is a correctness property: when a run takes
  hours, bugs survive.

Both were trained and compared. XGBoost won at 43.8% vs 47.5%.

## 6. Why not a random train/test split?

Because it lets the model see the future while predicting the past. With a random
split, a row from December can be in training while a row from June is in test —
and the model has effectively been told what happened later. The score comes out
excellent and the model is worthless in production, where the future genuinely
does not exist yet.

Time series must be split **chronologically**: train on the past, evaluate on the
future, exactly as production will run.

## 7. What is walk-forward validation?

Repeatedly re-fitting on an expanding window of history and scoring on the next
block:

```
train Jan–Jun  →  score Jul
train Jan–Jul  →  score Aug
train Jan–Aug  →  score Sep
```

It answers a question a single split cannot: **is accuracy stable, or was the
test period lucky?** Here it comes out at 0.3–1.8 percentage points of standard
deviation across folds — steady enough that a single headline number is not
misleading.

**Expanding, not rolling**, because a forecaster should get better as history
accumulates, and a rolling window discards the older seasons that make the
364-day lag meaningful.

## 8. What is the embargo, and why does it exist?

The part of the split that is specific to forecasting, and the answer that
usually earns follow-up questions.

A training row's *target* sits up to 90 days after its origin. So a training
origin sitting just before a fold boundary has its **target inside the evaluation
window** — the model is fitted on the very outcomes it is about to be scored on.
Nothing raises, and the test metric quietly improves.

The fix is a gap of `max_horizon` days between folds. It costs 90 days of
training origins at every boundary, and that cost is the point.

One subtlety worth mentioning: the check measures against the **nearest**
evaluation fold, not the test fold. Calibration and validation sit in between, so
measuring against test alone looks safe even with no embargo at all.

## 9. How did you prevent leakage more generally?

Five independent layers, because leakage never raises, never warns, and makes
every number look *better*:

1. **Structural** — `FeatureEngineer` requires a point-in-time view and raises
   `TypeError` on a bare repository. Leakage through the feature layer is a
   construction that will not run.
2. **Exclusion lists** — target-derived columns (`revenue = units × price`
   recovers the target exactly), supply features, `time_index`, `year`, the hive
   partition key.
3. **Per-row arithmetic tests** — `lag_7` at origin *t* is reconstructed **by
   hand** from the source panel and compared. A test that calls the
   implementation to compute its own expectation cannot detect a bug in that
   implementation.
4. **Behavioural tests** — long-horizon error must not *collapse*; nothing may
   score below the 35% noise floor.
5. **Falsifiability** — the suite **plants the exact bug** and asserts the
   detector fires. A test that has never failed proves nothing.

Layer 5 is the one to lead with. It is also how two real defects were caught:
festival columns going missing over short serving windows, and categorical dtypes
being inferred per frame.

## 10. How did you handle stockouts?

Observed sales during a stockout measure **availability, not demand**. If you sold
20 because you only had 20, training on 20 teaches the model that demand was 20.

Three mechanisms:

1. **Exclude rows whose target fell on a stockout.** The target is corrupted
   there.
2. **Keep stockout *origins*.** A stockout at the origin is a legitimate knowable
   state; dropping those origins would bias the feature distribution for no gain.
3. **Mask censored values before computing lags.** Excluding the rows is not
   enough — without masking, a stockout depresses the next eight weeks of lag
   features and the model learns the supply failure through the back door.

And **no inventory features at all**. Step 4 measured what happens otherwise:
with them, the model recovered only **0.30** of true stockout demand, having
learned that low stock predicts low sales — which is true, and is exactly the
censoring relationship to avoid.

**The honest cost**: stockouts here are *endogenous* — they happen because demand
spiked, with latent demand ~1.57× normal. Excluding them removes part of the
high-demand tail and biases the forecast slightly low. That is measured and
reported, not assumed away.

## 11. Why WMAPE?

**It weights by volume.** A 50% error on a SKU selling 10,000 units matters far
more than a 50% error on one selling three. Plain MAPE treats them as identical.

MAPE is also unusable as a headline here: it is undefined at zero and unstable
near it, and about 8.5% of rows are zero-unit days. So MAPE is computed **only
over non-zero actuals, with the excluded count reported** — silently dropping
them would overstate accuracy.

Alongside WMAPE: **MAE** (interpretable units), **RMSE** (error tail),
**bias** (signed — matters more than dispersion, because random error averages out
over a planning period and a consistent skew does not), and **MASE**
(scale-free, so comparable across series).

The single most important number, though, is the **35% irreducible noise floor**.
Demand is drawn from an over-dispersed negative binomial, so a model knowing the
*true* conditional mean would still score 35%. The model scores 43.8% — **1.25×
the floor**. Without that context, 43.8% reads as poor. With it, most of what is
capturable has been captured, and a model scoring 20% would be *impossible* and
should be treated as evidence of leakage.

## 12. How did you select the final model?

An explicit rule, applied in order:

1. **Accuracy** on the held-out test fold, by WMAPE.
2. **Simplicity on a near-tie** — if the seasonal naive comes within two
   percentage points, it wins. A benchmark that holds its own tells you the signal
   is simple.
3. **A leakage warning is emitted, not suppressed**, if error does not grow with
   horizon.

XGBoost won at 43.8% vs the naive's 56.0% — a 12.2-point margin, comfortably
outside the tolerance. **Forecast Value Added is positive at every horizon**, so
there is no bucket where the benchmark would be the better choice.

**The result worth telling**: XGBoost's *first* run scored 82.9% WMAPE at +58%
bias and looked far worse than LightGBM. The cause was parameter semantics —
`min_child_weight` sums *Hessians*, and under a Poisson objective the Hessian is
approximately μ, so it scales with the target level, while LightGBM's
`min_child_samples` counts *rows*. Setting both to 50 gave XGBoost ~38× less
regularisation. Reporting that run would have put a confident and completely false
line in the comparison table.

## 13. How did you track experiments, and did tuning help?

MLflow, experiment `revenue_intelligence_forecasting`. A parent *comparison* run
with nested candidate runs, because the comparison is the unit of work — a flat
run per candidate loses the fact that four models shared one dataset, one split
and one selection rule.

Logged: every parameter and hyperparameter, per-horizon-bucket metrics, feature
importance, the comparison table, the selection rationale, the evaluation report,
and a **config fingerprint** — one hash over the entire configuration, so two runs
are either comparable or visibly not.

**Tuning found nothing.** Twenty seeded trials on the validation fold: best 42.34%
against a default of 42.22%. Within fold-to-fold noise, so the defaults were kept.

That is the expected outcome at 1.25× the noise floor — there are only ~9 points
of learnable signal in total and hyperparameters compete for a fraction of it. The
right response is to report it, not to search harder until a number moves. The
code enforces this: `best_params()` returns "keep the defaults" below a
half-point threshold.

## 14. How does the ForecastingService work, and how will the Agent use it?

The service validates the request, loads the model once, builds point-in-time-safe
features, predicts, attaches calibrated intervals and provenance, and returns a
structured result.

**Expected failures come back as values, not exceptions** — a code, a message, and
a `recoverable` flag. That flag is what lets a supervisor agent re-plan rather
than give up.

The tool wraps the service for the agent. What the agent gets: the forecast, a
calibrated interval, measured accuracy at that horizon, provenance, assumptions
and warnings. What it never needs to know: that LightGBM exists, where the data
lives, how features are built, or what MLflow is.

**Why not let Claude calculate the forecast itself?** Three reasons, and the third
is the real one:

1. It cannot. A language model has no access to six million rows of history.
2. It would be non-reproducible — the same question could give different numbers.
3. **It would be unauditable.** Every number here traces to a fitted model, a
   dataset version and a code commit. A number an LLM produced traces to nothing,
   and nobody can check it. That is the constraint the whole platform is built
   around: Claude decides *what to compute and how to explain it*; deterministic
   models produce *every number*.

## 15. What happens when forecast quality is poor, or the request cannot be served?

Never a silently fabricated number.

- **Model missing** → structured error, `recoverable=False`. No reformulation
  helps until someone trains one, and saying so stops an agent retrying.
- **Unknown product or store** → refused. Returning an empty forecast would read
  as "no demand expected", which is a completely different claim from "not in the
  model".
- **Horizon past the planning calendar** → refused, *with the boundary*: "the
  latest as-of supporting a 90-day horizon is 2025-10-02." A recoverable error
  without that leaves an agent unable to re-plan. The alternative — assume no
  promotion is planned — produces a number that is systematically low and
  indistinguishable from a real forecast.
- **A series the model cannot serve** → falls back to the seasonal naive, then to
  a recent mean, and sets `fallback_used` with a reason. A caller must be able to
  tell an estimate from a guess.
- **Weak accuracy at long horizons** → surfaced as a warning on the response, not
  buried.
- **`confidence`** is measured interval coverage or it is absent. There is no
  third option where a plausible-looking number appears because the field exists.

---

## The through-line

If you take one idea into the room, take this: **every mechanism in this system
exists because a specific, plausible mistake would otherwise be invisible.**

The embargo exists because a boundary row's outcome leaks silently. The noise
floor is reported because a bare WMAPE is uninterpretable. The leakage suite
plants a bug because a test that never fails proves nothing. The forecaster
refuses to forecast past the calendar because the alternative looks exactly like a
real answer.

The argument is not that the model is accurate. It is that **when it is wrong,
something says so.**
