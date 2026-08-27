# Step 6 — Demand Forecasting: final report

The A–T deliverables, written against measured results.

> **On numbering.** This brief calls the capability Step 6; the repository built
> it as Step 5 and the commit history says so. Same thing — see
> [`README.md`](README.md). Numbering follows the brief from here on.

---

## A. Architecture

Seven layers, dependencies pointing strictly downward. Forecasting occupies the
model layer and is reached through a service, which a tool wraps for the future
agent.

```
Agent layer          (Steps 13-20, not built)
Tool contract        ForecastingTool  ->  ToolResult envelope
Service layer        ForecastingService
Model layer          ml/forecasting/  (19 modules + 3 added this step)
Feature layer        FeatureEngineer, availability classes
Data access          DataRepository, PointInTimeView
Storage              Parquet gold tables
```

Three seams make the Databricks migration a redeployment rather than a rewrite:
`DataRepository`, MLflow tracking URI, and the DI container keyed on
`APP__ENVIRONMENT`. Nothing above them changes.

## B. Data flow

```
sample_series          800 real product-store pairs, volume-stratified,
                       store-clustered, semi-joined to an exact set
      |
build_history          FeatureEngineer over a PointInTimeView
                       + mask_censored before lags are computed
      |
build_horizon_dataset  self-join: (origin t, horizon h, target at t+h)
                       origin-side and h_-prefixed target-side features
      |
build_origin_split     train / calibration / validation / test,
                       90-day embargo between every fold
      |
train_forecaster       fit -> calibrate per bucket -> score on test
      |
persist + track        model.joblib, evaluation_report.md, MLflow
```

Serving reverses the last steps: `build_future_scaffold` generates rows for dates
that have not happened, using the **same** `target_side_features` function
training uses — the property a test asserts directly.

## C. Forecasting methodology

**Direct multi-step, one global model, horizon step `h` as a feature.**

A training row is `(origin t, h, target = units at t+h)`. Each feature is placed
by asking whether it is knowable at `t`, which Step 3's availability classes
already answer: demand history and competitor position from the origin; calendar,
planned promotion and planned price from the target date.

Rejected alternatives, with the dispositive reason for each:

| Alternative | Why not |
|---|---|
| Four per-horizon models on a cumulative target | Cannot produce the daily path the interface requires; nested totals would be mutually incoherent |
| Recursive one-step-ahead | **Feature-distribution collapse** — feeding conditional means forward drives rolling std and volatility to zero, so by h=30 the model sees inputs it never saw in training |
| Four direct models on daily targets | 4× compute, less data per model, no benefit once `h` is a feature |

Horizon steps are drawn at **random** from U{1..90} rather than a fixed grid: a
grid makes splits on `h` piecewise-constant, which shows as a staircase in the
daily path.

## D. Feature catalogue

83 features across eight groups. Full per-feature table with derived leakage risk:
[`feature_catalogue.md`](feature_catalogue.md).

| Origin side (`t`) | Target side (`h_` prefix) |
|---|---|
| lags 1/7/14/28/56/364 | calendar, season, holiday, festival |
| rolling mean and std 7/14/28/56 | planned promotion flag, type, discount, duration |
| momentum, volatility, trend | planned selling and regular price |
| price position, discount depth, index | — |
| competitor price, gap, ratio | — |
| product and store attributes | — |

## E. Leakage prevention

Five layers (§9 of [`interview_guide.md`](interview_guide.md) explains each).
Test coverage:

| Test | Asserts |
|---|---|
| Mutation | Corrupting all OBSERVED data after a cutoff leaves training features byte-identical |
| **Falsifiability** | The mutation test **fails** when the bug is planted |
| Arithmetic | `lag_7` at `t` equals units at `t−7`, reconstructed by hand from the source panel |
| Target leak | A planted target feature produces implausible accuracy — proving the floor check can fire |
| Collapse | Long-horizon error has not collapsed relative to short |
| Noise floor | Nothing scores below 35% WMAPE |
| Train/serve | Both feature paths produce identical vectors |
| Embargo | No training target lands inside an evaluation fold, in the structural worst case |

## F. Stockout strategy

Approach A with three qualifications: exclude rows whose **target** fell on a
stockout; **keep** stockout origins; **mask censored values before lags** so the
failure does not propagate into eight weeks of history. No inventory features at
all — Step 5 measured that they teach the model to read low stock as low demand
(0.30 recovery of true stockout demand).

**The cost is stated, not hidden**: stockouts are endogenous, so excluding them
removes part of the high-demand tail and biases the forecast low.

## G. Model comparison

800 series, 548,754 rows, seed 42, 475 seconds.

| Model | WMAPE | MAE | RMSE | MASE | Bias | Mean FVA | Train | Predict | Complexity |
|---|---|---|---|---|---|---|---|---|---|
| **xgboost** | **43.8%** | 24.86 | 53.13 | 0.68 | +8.4% | **+12.6 pp** | 34.4s | 0.42s | high |
| lightgbm | 47.5% | 26.94 | 53.78 | 0.74 | +19.9% | +9.5 pp | 11.3s | 0.63s | high |
| horizon_seasonal_naive | 56.0% | 31.78 | 67.69 | 0.80 | +7.2% | — | 0.4s | 0.11s | trivial |
| horizon_naive | 78.1% | 44.33 | 96.93 | 1.05 | +36.1% | −19.8 pp | 0.4s | 0.09s | trivial |

**Selection criterion**: accuracy → simplicity on a near-tie (seasonal naive wins
within 2 points) → leakage warning if error does not grow with horizon.

**XGBoost selected**, 12.2 points clear of the benchmark. Its first run scored
82.9% and was thrown out as a configuration error, not reported as a model result
— see [`evaluation.md`](evaluation.md) §6.

## H. Validation results

**Per horizon bucket** — never blended:

| Bucket | WMAPE | Bias |
|---|---|---|
| h1-3 | 43.6% | +9.8% |
| h4-7 | 42.2% | +6.9% |
| h8-14 | 43.4% | +7.2% |
| h15-28 | 43.0% | +7.6% |
| h29-56 | 44.2% | +8.6% |
| h57-90 | 44.7% | +9.5% |

**By aggregation level** — error falls 4.5× from SKU to total:

| Level | WMAPE |
|---|---|
| product × store | 43.6% |
| product | 35.7% |
| store | 18.7% |
| category | 15.0% |
| region | 11.8% |
| **total** | **9.6%** |

Backtest stability: 0.3–1.8 points std across folds, every bucket stable.
Interval coverage: near nominal per bucket, measured on held-out data.

**Against the noise floor**: 43.8% / 35.0% = **1.25×**.

## I. MLflow experiment design

Experiment `revenue_intelligence_forecasting`, registered model `demand_forecast`.
Parent *comparison* run with nested candidates; a separate `registered_*` run
carries the artifact. Full detail in
[`../models/demand_forecasting.md`](../models/demand_forecasting.md#mlflow-strategy).

Logged: all params and hyperparameters, per-bucket metrics, FVA, feature
importance, comparison table, selection rationale, evaluation report,
`code_version` and a `config_fingerprint`.

**Fixed this step**: `register_selected` returned a `runs:/…/model` URI even when
nothing had been registered — a naive benchmark has no artifact. It now returns
`None`, because a URI resolving to a 404 is worse than an explicit absence.

## J. ForecastResult schema

`ForecastResult` (`ml/forecasting/interface.py`) carries `product_id`, `store_id`,
`region`, `horizon`, `points[{date, predicted_units, lower_bound, upper_bound}]`,
`total_predicted_units`, `total_predicted_revenue`, `backtest_metrics`,
`model_used`.

`ForecastResponse` (`app/schemas/forecast.py`) is the service-level contract and
adds the provenance the brief's schema asks for: `model_name`, `model_version`,
`dataset_version`, `feature_version`, `as_of_date`, `horizon_days`, `confidence`,
`accuracy`, `fallback_used`, `fallback_reason`, `assumptions`, `warnings`,
`execution_time_ms`. `trace_id` is attached one layer up by the `ToolResult`
envelope, which owns tracing for every tool uniformly.

**Prediction intervals are implemented**, not deferred: split conformal calibrated
per horizon bucket, plus a separately calibrated quantile for the horizon total.
`confidence` is **measured interval coverage or absent** — never invented.

## K. ForecastingService design

```python
service.forecast(ForecastRequest(
    horizon=ForecastHorizon.D28,
    product_ids=[...], store_ids=[...], region=...,
    as_of_date=..., include_points=True,
))
```

Validates → loads the model once (lazily, so a missing artifact is not a boot
failure) → builds point-in-time-safe features → predicts → attaches intervals,
provenance, assumptions and warnings → returns `ForecastResponse` **or**
`ForecastErrorResponse`.

Expected failures are **values, not exceptions**, each with a code and a
`recoverable` flag — which is what lets a Step 16 supervisor re-plan.

## L. Test strategy

**602 tests total; 142 in `tests/forecasting/`** (up from 74 this step).

| File | Tests | Covers |
|---|---|---|
| `test_dataset.py` | 22 | Origin/target arithmetic, feature placement, censoring, determinism |
| `test_model.py` | 20 | Train, predict, output shape, non-negativity, determinism, intervals |
| `test_service.py` | 20 | Missing model/product/store, refusals, provenance, the 28-day horizon |
| `test_data_quality.py` | 17 | Every check twice — clean panel passes, corrupted panel fires |
| `test_tool_contract.py` | 16 | Declaration, failure behaviour, what the agent receives |
| `test_tuning.py` | 13 | Reproducibility, **test fold untouched**, marginal gains not adopted |
| `test_baselines.py` | 13 | Seasonal reference from the target date, fallback chain |
| `test_split.py` | 12 | Fold ordering, embargo, structural worst case |
| `test_leakage.py` | 9 | Mutation, falsifiability, train/serve, planted target leak |

Principles: reconstruct expectations **by hand**; prefer behavioural assertions;
contextualise against the noise floor rather than hard-coded thresholds; test the
failure path first.

## M. Failure handling

| Failure | Code | Recoverable | Behaviour |
|---|---|---|---|
| No trained model | `model_not_found` | **No** | Names the training command |
| Model loaded but unusable | `model_not_fitted` | **No** | — |
| Unknown product or store | `insufficient_data` | Yes | Refused, never an empty forecast |
| Horizon past the calendar | `insufficient_data` | Yes | **Names the latest valid as-of** |
| Series too short | `insufficient_data` | Yes | Carries available vs required days |
| Feature build failed | `forecast_failed` | Yes | Carries the failing stage |
| Unsupported horizon | `invalid_input` | Yes | Lists the supported set |
| Model cannot serve a row | — | — | Falls back to seasonal naive, sets `fallback_used` + reason |

Typed exceptions in `ml/forecasting/exceptions.py` carry their own recoverability,
so two call sites cannot disagree about whether the same failure is worth
retrying.

## N. Databricks migration

Design only — [`../forecasting_databricks_migration.md`](../forecasting_databricks_migration.md).

| Moves unchanged | Genuinely changes |
|---|---|
| config, split, conformal, evaluation, estimators, schemas, the whole tool contract | panel construction (pandas → PySpark), the horizon self-join (→ Delta join), the future scaffold, model loading (joblib → Unity Catalog), monitoring (→ Lakehouse Monitoring) |

Recommended pattern: **distributed feature engineering, single-node training** —
a global model on ~600k rows fits in one executor in minutes, and distributing the
boosting would add coordination overhead for no gain at this size.

## O. Known limitations

| Limitation | Consequence |
|---|---|
| Excluding stockout targets biases low | Removes part of the high-demand tail; measured and reported |
| No inventory signal | Forecasts **demand**, not shipments. Replenishment wants a different model |
| Competitor prices frozen at the origin | A competitor move inside the horizon is invisible |
| No cannibalisation or halo | A promotion on one SKU distorts its substitutes, unattributed |
| Cannot forecast past 2025-12-31 | A data limitation; refused explicitly rather than faked |
| Conformal assumes exchangeability | A trend violates it; coverage is measured per bucket so a shortfall surfaces |
| The horizon gradient is shallow | Only ~9 points of learnable signal exist, so the gradient is weak evidence about join correctness. The leakage tests carry that argument |
| Synthetic data | Real retail has structure this simulator does not reproduce |

## P. Future improvements

Eight candidates, each with the reason it was deferred rather than overlooked:
[`../models/demand_forecasting.md`](../models/demand_forecasting.md#v2-what-would-come-next-and-why-it-is-not-here).

The one that would help first is **probabilistic forecasting**, because inventory
decisions have asymmetric costs — and that becomes relevant once optimisation
exists to consume it.

## Q. Files created and modified

**Created this step**

```
ml/forecasting/exceptions.py           typed failures carrying recoverability
ml/forecasting/quality.py              16 data-quality checks on the forecasting grain
ml/forecasting/tuning.py               seeded, capped random search

tests/forecasting/test_model.py        20 tests
tests/forecasting/test_baselines.py    13 tests
tests/forecasting/test_data_quality.py 17 tests
tests/forecasting/test_tuning.py       13 tests

docs/forecasting/README.md             index + the numbering mapping
docs/forecasting/data_contract.md      grain, availability, thresholds, coverage
docs/forecasting/evaluation.md         metrics, selection, search outcome
docs/forecasting/feature_catalogue.md  per-feature leakage table (generated)
docs/forecasting/step6_report.md       this document
docs/forecasting/interview_guide.md    the system in plain language
docs/generate_feature_catalogue.py     generator for the catalogue
```

**Modified**

```
app/schemas/domain.py                  + ForecastHorizon.D28
app/cli.py                             + forecast, evaluate-forecast, forecast-quality
app/tools/forecasting_tool.py          28-day default, updated description
configs/models/forecasting.yaml        + 28 horizon, + tuning block
ml/forecasting/config.py               + TuningConfig
ml/forecasting/evaluate.py             + MASE, + seasonal_naive_scale
ml/forecasting/pipeline.py             + tuning wiring, + MASE in the comparison
ml/forecasting/tracking.py             fixed the fabricated model URI
ml/forecasting/{split,train,predict,model,xgboost_model}.py   typed exceptions
scripts/train_forecast.py              + --tune
tests/forecasting/{test_service,test_leakage}.py   + missing store, D28, planted leak
docs/models/demand_forecasting.md      + MLflow strategy, + V2 roadmap
pyproject.toml                         per-file lint ignores for tests/forecasting
```

## R. Commands — training

```powershell
uv run python scripts/train_forecast.py --smoke --no-track   # <60s correctness check
uv run python scripts/train_forecast.py --seed 42            # ~8 min, the deliverable
uv run python scripts/train_forecast.py --seed 42 --tune     # + ~4 min for the search
uv run python scripts/train_forecast.py --full               # every series; hours
```

## S. Commands — evaluation

```powershell
uv run ari evaluate-forecast        # print the stored evaluation report
uv run ari forecast-quality         # input-side data-quality checks, exits non-zero on FAIL
uv run pytest tests/forecasting -v  # 142 tests
.\tasks.ps1 check                   # ruff + mypy + pytest
```

## T. Commands — generating forecasts

```powershell
uv run ari forecast --product P00003 --store S00155 --horizon 28
uv run ari forecast --product P00003 --horizon 28 --daily
uv run ari forecast --product P00003 --store S00155 --horizon 90 --as-of 2025-09-01
```

```python
from app.schemas.domain import ForecastHorizon
from app.schemas.forecast import ForecastRequest
from app.services.container import Container

response = Container().forecasting_service.forecast(
    ForecastRequest(horizon=ForecastHorizon.D28, product_ids=["P00003"])
)
```

Exits non-zero on a refusal, printing the error code, whether re-planning could
succeed, and what would have worked instead.
