# Service contract

## `PromoUpliftService.estimate_uplift`

```python
from app.schemas.promo_uplift import UpliftRequest
from app.services.container import Container

response = Container().promo_uplift_service.estimate_uplift(
    UpliftRequest(
        promotion_ids=["PR0000123"],       # or omit to aggregate
        product_ids=None,
        store_ids=None,
        region=None,
        category=None,
        analysis_start_date=None,
        analysis_end_date=None,
        include_pull_forward=True,
        include_segments=True,
        include_events=True,
        max_events=500,
    )
)
```

Returns `UpliftResponse | UpliftErrorResponse`. **Expected failures come back as
values, not exceptions** — by Step 16 a supervisor has to re-plan around them,
and it can only do that with a failure it can read.

## What the service serves

A **persisted analysis**, not a live estimate.

Uplift is retrospective: the question is always about promotions that already
ran, and the answer does not change between requests. A full run — control
construction, cross-fitted nuisance models, placebo, sensitivity — takes minutes,
so it happens once in `scripts/estimate_uplift.py` and the service reads the
result.

The model loads lazily. Constructing the container never requires an analysis to
exist, so a clean checkout still starts and a missing artifact surfaces as a
readable error at the point someone asks for uplift.

## `UpliftResponse`

```json
{
  "status": "success",
  "model_name": "promo_uplift",
  "model_version": "v1.0",
  "dataset_version": "...",
  "feature_version": "...",

  "treatment_definition": "treated = a promotion of any mechanic with depth >= 5% ...",
  "method": "augmented_ipw",
  "method_reason": "weakest identifying assumptions among the methods that passed validation",

  "baseline_units": 1000.0,
  "observed_units": 1150.0,
  "incremental_units": 150.0,
  "uplift_pct": 0.15,
  "incremental_revenue": 5000.0,
  "incremental_profit": 1800.0,
  "promotion_spend": 700.0,
  "roi": 2.57,

  "confidence_interval": {"lower": 0.05, "upper": 0.24, "confidence_level": 0.95},
  "gross_uplift_pct": null,
  "pull_forward_units": null,

  "events_analysed": 42,
  "treated_days": 380,
  "events": [...],
  "segments": [...],
  "comparison": [...],
  "diagnostics": {...},

  "validation_status": "passed",
  "assumptions": [...],
  "warnings": [...],
  "execution_time_ms": 84
}
```

### The three fields a forecast response does not have

**`treatment_definition`** — required, not optional. An uplift number is
uninterpretable without it, and two figures computed under different definitions
are not comparable.

**`validation_status`** — `passed` | `warnings` | `failed`. A forecast is more or
less accurate; a causal estimate is either identified or it is not, and that is a
different kind of statement. A caller must be able to tell "we measured +18%"
from "we computed +18% but the design does not support calling it causal".
`response.is_causal` is the property that decides.

**`assumptions`** — not a disclaimer. These are the conditions under which the
number *is* the causal effect, and they are the first thing a reviewer should
attack.

### `confidence_interval` is absent, never invented

Present only when it was computed — analytic for AIPW (cluster-robust influence
function), bootstrap where analytic is unavailable, `None` otherwise. Same rule
as `confidence` in the forecast contract.

## Refusals

| Failure | `error_code` | Recoverable | Detail |
|---|---|---|---|
| No persisted analysis | `model_not_found` | **no** | Names `scripts/estimate_uplift.py` |
| Unknown promotion id | `insufficient_data` | yes | Lists the unknown ids |
| No matching events | `insufficient_data` | yes | Echoes the filters, suggests a re-run |
| Reversed date range | `invalid_input` | yes | Both dates |
| No control group | `no_control_group` | yes | Rows found vs required |
| Overlap violated | `assumptions_violated` | yes | Trimmed share vs threshold |
| Pre-period too short | `insufficient_data` | yes | Days available vs required |

**An unknown promotion is refused, not answered with zero.** "This promotion had
no effect" and "this promotion is not in the analysis" are different findings,
and a category manager acting on the first when the second is true concludes a
mechanic does not work on no evidence at all.

## The exception hierarchy

Typed failures in `ml/promo_uplift/exceptions.py`, each carrying its own
recoverability as a class attribute — a property of the failure kind rather than
the call site, so two call sites cannot disagree.

The **inheritance is load-bearing**: everything meaning "the data cannot support
this" subclasses `ml.base.InsufficientDataError`, and
`UpliftModelUnavailableError` subclasses `ml.base.ModelNotFittedError`. The
service maps through those two base classes, so a new subclass is routed
correctly without touching the service.

`CausalAssumptionsViolatedError` is the one this capability exists to be able to
raise. A promo uplift model that never refuses is not a causal model — it is a
regression with a causal label on it. The estimate that *would* have been
produced rides in `detail.unidentified_estimate`, clearly marked.

## `PromoUpliftResult` (the Step 1 contract)

`ml/promo_uplift/interface.py` fixed the shape in Step 1 and later steps
reference it, so it was extended rather than replaced. `UpliftResult` carries
`baseline_units`, `actual_units`, `incremental_units`, `uplift_pct`,
`incremental_revenue`, `incremental_profit`, `promotion_spend`, `roi`,
`confidence_interval`, `pull_forward_units`, `cannibalisation_units`, `method`,
`control_group_size`, `treatment_group_size` and `assumptions`.

`FittedUpliftModel.for_promotion(id)` returns one. It is a table lookup, not a
model call — the effect was estimated when the analysis ran.

## CLI

```powershell
uv run ari uplift --promotion PR0000123
uv run ari uplift --product P00003 --store S00155
uv run ari uplift-quality
uv run ari uplift-validate --scenario confounded
```

`ari uplift` prints the treatment definition **first**, then the number. It exits
non-zero when validation failed, so a script piping this into a report has to opt
in to using an estimate whose causal assumptions did not hold.
