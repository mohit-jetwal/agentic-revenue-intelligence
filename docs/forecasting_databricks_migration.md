# Forecasting: Stage 1 → Databricks migration

> Design only. Nothing in this document is implemented — §41 is explicit that
> Databricks notebooks, Workflows, Delta tables, Unity Catalog and Model Serving
> belong to the production stage. **No AWS.**

## The claim being tested

Step 1 built three seams so that moving to production would be a redeployment
rather than a rewrite. Step 5 is the first capability complex enough to test that
claim properly: it has a training pipeline, a serving path, calibrated intervals,
a service and a tool.

The answer is encouraging but not unqualified — and the qualifications are the
useful part of this document.

| Seam | Stage 1 | Stage 2 | What changes |
|---|---|---|---|
| `DataRepository` | `LocalDataRepository` (DuckDB/Parquet) | `DatabricksDataRepository` (SQL Warehouse) | `APP__ENVIRONMENT=databricks` |
| `PointInTimeView` | wraps the local repository | wraps the Databricks one | nothing — it composes either |
| MLflow | `sqlite:///data/local/mlflow.db` | Databricks MLflow | `ML__TRACKING_URI=databricks` |
| Model registry | local MLflow registry | Unity Catalog | `ML__REGISTRY_URI=databricks-uc` |

---

## What moves unchanged

These carry over as pure Python, because none of them touches storage:

| Module | Why it is portable |
|---|---|
| `ml/forecasting/config.py` | YAML + Pydantic. No I/O beyond reading one file |
| `ml/forecasting/split.py` | Pure date arithmetic over origins |
| `ml/forecasting/conformal.py` | NumPy over residual arrays |
| `ml/forecasting/evaluate.py` | Pandas over a scored frame |
| `ml/forecasting/baselines.py`, `xgboost_model.py` | sklearn-shaped estimators |
| `app/schemas/forecast.py` | Pydantic contracts |
| `app/tools/forecasting_tool.py` | Calls the service; knows nothing about storage |

**The tool contract in particular does not change at all.** That is the whole
point of Step 1's envelope: the agent asks for a 30-day forecast and receives
units, an interval and provenance, whether the model is behind DuckDB or a
Serving endpoint.

---

## What genuinely changes

### 1. Panel construction becomes Spark — and this is the real work

`ml/forecasting/dataset.py:build_history` calls `FeatureEngineer`, which is
pandas. At 800 series that is a 1.5-million-row frame; at the full 6,128 it is
~6.7 million rows in memory, and a real retailer has two orders of magnitude more.

```python
# Stage 1
panel = FeatureEngineer(view).build(FeatureRequest(...))
panel = sample.restrict(panel)

# Stage 2 — sketch
panel = (
    spark.table("cpg.gold.sales_daily")
      .filter(F.col("date").between(start, end))
      .transform(add_demand_features)      # Window over (product, store) ordered by date
      .transform(add_price_features)
      .transform(add_promotion_features)
)
```

The lag and rolling features map cleanly onto Spark window functions, which is
the good news. The subtlety is that `mask_censored` must still run **before** the
windows — the ordering that keeps a stockout from propagating into the next eight
weeks of lags is a property of the transformation sequence, not of the engine.

### 2. The horizon self-join becomes a Delta join

The single most expensive operation, and the one that benefits most:

```python
origins = panel.filter(F.dayofyear("date") % stride == 0)
horizons = spark.range(1, max_horizon + 1).withColumnRenamed("id", "horizon_step")

rows = (
    origins.crossJoin(horizons.sample(fraction))
      .withColumn("target_date", F.expr("date_add(date, horizon_step)"))
      .join(targets, ["product_id", "store_id", "target_date"], "inner")
      .join(target_side, ["product_id", "store_id", "target_date"], "left")
)
```

The random horizon draw becomes a `sample()` on the cross join. Everything else is
the same operation expressed in a different dialect, and the **`h_` prefix
convention survives verbatim** — which matters, because it is what keeps the
origin/target split legible in a Spark plan too.

### 3. The future scaffold becomes a generated frame

`build_future_scaffold` exists because `FeatureEngineer` structurally cannot emit
rows for dates with no sales. In Spark the same constraint applies for the same
reason, and the same fix works: `sequence()` to generate future dates, cross-join
the pair set, then join the `KNOWN_IN_ADVANCE` tables. The train/serve equivalence
test moves across as-is and becomes *more* valuable, since the two paths would
then be written in different engines.

### 4. Training does not distribute, and should not

A single global LightGBM on ~600k rows fits in one executor's memory in a couple
of minutes. Distributing it via SynapseML would add coordination overhead and a
dependency for no gain at this size.

The right Stage 2 pattern is **distributed feature engineering, single-node
training**: Spark builds the horizon dataset and writes it to Delta; a single-node
task reads it as pandas and fits. If the panel grows past what one node can hold,
the honest first move is more aggressive series sampling — not distributed
boosting, whose accuracy gain at this scale is usually smaller than the variance
between folds.

### 5. Serving

```
Stage 1:  FittedForecastModel.load_from(dir) → joblib
Stage 2:  Unity Catalog model version → Databricks Model Serving endpoint
```

The `ForecastingService` interface is unchanged; only the model-loading line moves.
One genuine complication: **the serving endpoint needs origin-side features**,
which live in the panel. Two options, and the choice is a real trade-off:

- **Online Feature Store** (Databricks Feature Serving) — low latency, but the
  feature table must be materialised and kept fresh.
- **Batch-scored forecasts written to Delta**, served by lookup — much simpler,
  and adequate because a demand forecast is not a per-request decision. Daily
  batch scoring is the natural cadence.

The second is recommended for Stage 2, with the first held in reserve if an
interactive what-if flow appears in Step 20.

---

## Unity Catalog layout

```
cpg_revenue_intelligence
├── gold/                          sales_daily, pricing, promotions, calendar, ...
├── features/
│   ├── forecast_panel             the historical feature panel (Delta, partitioned by date)
│   └── forecast_horizon_dataset   the (origin, h, target) rows
└── ml/
    ├── demand_forecast            registered model
    └── forecast_predictions       batch-scored output, for monitoring
```

`forecast_predictions` is not decoration: it is what
`monitoring.forecast_vs_actual` joins against once actuals arrive, and without it
stored forecasts are lost and the model cannot be scored in production at all.

---

## Monitoring

`ml/forecasting/monitoring.py` deliberately implements only what is real now —
PSI drift, forecast-vs-actual, prediction distribution — and no alerting, because
nothing is serving traffic and a threshold would be invented rather than derived.

Stage 2 replaces most of it:

| Stage 1 function | Stage 2 |
|---|---|
| `drift_report` | Lakehouse Monitoring drift metrics over the feature table |
| `forecast_vs_actual` | Monitoring job joining `forecast_predictions` to `sales_daily` |
| `prediction_distribution` | Inference table profile metrics |
| `segment_performance` | Monitoring slice expressions |

The per-horizon-bucket breakdown does **not** come for free — Lakehouse Monitoring
slices on columns, so `horizon_bucket` must be materialised as a column on the
predictions table. Worth doing: it is the dimension along which a forecasting
model degrades first, and a blended drift metric would hide it.

---

## Workflows

```
forecast_training (weekly)
  1. build_horizon_dataset   (Spark → Delta)
  2. train_candidates        (single node, reads Delta)
  3. evaluate_and_select     (writes comparison + rationale artifacts)
  4. register_model          (Unity Catalog, stage transition gated on review)

forecast_scoring (daily)
  1. build_future_scaffold   (Spark)
  2. score                   (Model Serving batch)
  3. write_predictions       (Delta)

forecast_monitoring (daily)
  1. drift + forecast_vs_actual → monitoring tables
```

Step 4's hard-won guard carries over unchanged and matters more here, not less:
**the evaluation report is written before tracking, and a tracking failure does
not fail the run.** A three-hour local run was lost to an MLflow store rejection
raised after every model had been fitted; on a Workflows cluster the same failure
costs cluster time as well.

---

## What this migration would *not* fix

Honest limits, so the migration is not oversold as a solution to modelling
problems:

- **The stockout-exclusion bias** is a modelling decision, not a scale problem. It
  travels unchanged.
- **The frozen competitor price** is a data-availability fact — competitor pricing
  is `OBSERVED` in either environment.
- **The end-of-calendar refusal** becomes *less* visible in production, not more:
  a real promotion calendar extends further forward, so the refusal fires rarely.
  That is a reason to keep the validation, not to drop it — the failure mode it
  prevents is silent.
- **Distributing the statistical models would not help.** Spark would parallelise
  6,128 ETS fits across executors easily — the measured cost is only ~25 minutes
  single-threaded, so this was never the binding constraint. The reason those fits
  are done at aggregate grain is that ETS is *statistically appropriate* there and
  scores 48.2% WMAPE at product-store grain against the global model's 43.8%.
  Running an inappropriate model faster does not make it appropriate.
