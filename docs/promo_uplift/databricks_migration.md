# Local → Databricks migration

Design only. Nothing here is implemented, and no cloud infrastructure of any kind
is introduced in Step 7.

**The rule that shapes it**: the `PromoUpliftService` contract does not change.
Everything above the seams — the request and response schemas, the tool contract,
the treatment definition, the estimators, the diagnostics — moves unchanged.

## What moves unchanged

| Component | Why it is portable |
|---|---|
| `config.py`, `exceptions.py` | Pure Python and YAML |
| `treatment.py`, `controls.py` | pandas, but on the *analysis* frame, which stays small |
| `propensity.py`, `matching.py`, `estimators.py` | numpy and scikit-learn on a single node |
| `diagnostics.py`, `evaluate.py`, `business.py` | Arithmetic over results |
| `synthetic.py` | Self-contained |
| `app/schemas/promo_uplift.py`, the service, the tool | No storage knowledge |

## What genuinely changes

| Component | Local | Databricks |
|---|---|---|
| `data.py` panel build | DuckDB over Parquet | Spark SQL over Delta |
| Covariate construction | pandas `groupby.rolling` | Spark window functions |
| Event extraction | pandas `groupby` | Delta aggregate |
| Artifact | joblib on disk | Unity Catalog model |
| Tracking | MLflow on SQLite | Databricks MLflow |
| Scheduling | manual script | Databricks Workflows |
| Governance | filesystem | Unity Catalog |

## The recommended pattern

**Distributed feature engineering, single-node estimation.**

The panel is large — millions of product-store-days — and building trailing
covariates over it is exactly what Spark window functions are for. The
*estimation* is not: after treatment and control selection the analysis frame is
in the hundreds of thousands of rows at most, which fits comfortably in one
executor. Distributing the propensity fit and the AIPW sum would add coordination
overhead for no gain at this size.

```
Delta: sales, promotions, products, stores
   │
   ├── Spark: trailing covariates, anchored at the event start
   │          (window functions partitioned by product_id, store_id)
   │
   ├── Spark: treatment / control / washout roles
   │
   ├── collect() the analysis frame  ← the one deliberate collect
   │
   ├── single node: propensity, outcome models, AIPW, DR-learner
   │
   ├── Delta: event_impact table  ← what Step 8 reads
   └── Unity Catalog: registered analysis + MLflow run
```

The `collect()` is deliberate and bounded. Control selection is what makes it
safe: the analysis frame is treated rows plus their eligible controls, not the
whole panel.

## Where the anchoring must be reimplemented carefully

The single riskiest part of the migration.

Locally, trailing covariates are computed on `shift(1)` within each listing, then
joined at the **anchor date** — which is the event start for treated rows and the
row's own date for controls. In Spark that becomes:

```sql
-- Trailing statistics, valid as of each date
AVG(units) OVER (
  PARTITION BY product_id, store_id
  ORDER BY date
  ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING     -- excludes the current row
) AS demand_mean_7
```

then a join on `(product_id, store_id, anchor_date)` rather than on `date`.

Getting the `1 PRECEDING` wrong by one row silently includes the current day. On
a treated row that is one day of the treatment effect inside a covariate, and
nothing in the output would show it. The local tests that reconstruct
`demand_mean_7` by hand from the source panel should be ported first and run
against the Spark implementation before anything else.

## Point-in-time correctness

`PointInTimeView` masks realised columns beyond the as-of date. In Databricks the
equivalent is Delta time travel plus the same masking rules, or a feature store
with point-in-time lookups.

For uplift this matters less than for forecasting — the analysis is retrospective
and every covariate is anchored before treatment by construction — but the
promotion table still carries realised `promotion_spend` and `promotion_units`,
which are post-treatment and already excluded.

## Scheduling

A Databricks Workflow, monthly or after each promotion cycle:

1. Refresh the panel from Delta.
2. Run the data-quality checks. **Fail the job on a FAIL** rather than
   proceeding — a causal estimate on bad data misleads rather than degrades.
3. Estimate, with placebo and sensitivity.
4. **Fail the job if validation status is `failed`.** Do not publish.
5. Write `event_impact` to Delta and register the analysis in Unity Catalog.
6. Alert if the ATT has moved more than a configured amount since the last run —
   a large jump is either a real regime change or a pipeline bug, and both need a
   human.

## Governance

| Concern | Unity Catalog |
|---|---|
| Who can read the analysis | Table and model grants |
| Which run produced a number | MLflow run id on the registered model |
| What the number means | `treatment_definition` in run params |
| Lineage | Delta table lineage from sales through to `event_impact` |
| Approval | `ModelMetadata.approved`; an unapproved analysis must not be served to an agent |

The treatment definition in run params is the one that is easy to skip and
expensive to lose. Two analyses under different definitions are not comparable,
and without the definition recorded there is no way to discover that afterwards.

## Cost and scale

| Stage | Local (300 pairs) | Databricks (full) |
|---|---|---|
| Panel build | ~30 s | minutes, distributed |
| Covariates | ~10 s | minutes, distributed |
| Nuisance + estimation | ~60 s | similar — single node, same data size |
| Placebo | ~60 s | similar |
| Sensitivity (10 specs) | ~10 min | the dominant cost |

Sensitivity is the expensive part because it re-runs the whole pipeline ten
times. In a scheduled job it should run on a fixed cadence rather than every
time, with `--no-sensitivity` for the routine refresh — and the report should say
which mode produced it.

## What is deliberately not designed here

Model serving. Uplift is retrospective: there is no low-latency inference to
serve. The service reads a persisted analysis, and the only genuinely predictive
operation is the CATE model scoring hypothetical promotions, which Step 8 calls
in batch. A serving endpoint would be infrastructure for a request pattern that
does not exist.
