# Databricks Migration Mapping

How each Stage 1 component becomes its Stage 2 equivalent (brief sections 43–44).

**No AWS.** No S3, ECR, ECS, EKS, Lambda, API Gateway, RDS, ElastiCache,
CloudWatch, IAM or Secrets Manager anywhere in this project. Production is
Databricks-native throughout.

Nothing here is implemented yet. The point of writing it now is that the
interfaces were designed against it, so the migration is a known quantity rather
than a discovery exercise.

## The claim being made

> ML business logic does not change when the storage changes.

That is only true if the seams hold. Three of them do the work:

| Seam | Where | What it hides |
|---|---|---|
| `DataRepository` | [data/repositories/base.py](../data/repositories/base.py) | Parquet vs Delta vs Databricks SQL |
| `PointInTimeView` | [data/repositories/point_in_time.py](../data/repositories/point_in_time.py) | how the as-of cut is enforced |
| `FeatureRepository` | [features/repositories/base.py](../features/repositories/base.py) | local computation vs a Feature Store |

Nothing outside `data/repositories` imports `duckdb` or opens a file. Nothing
outside `features/` computes a lag. A model receives a `FeatureSet` and cannot
tell where it came from.

## Component mapping

| Concern | Stage 1 (local) | Stage 2 (Databricks) | What changes |
|---|---|---|---|
| Storage | Parquet under `data/local/gold/` | Delta tables in Unity Catalog | new repository implementation |
| Query engine | DuckDB, embedded | Databricks SQL Warehouse | same |
| Data access | `LocalDataRepository` | `DatabricksDataRepository` | one class, already declared |
| Point-in-time | `PointInTimeView` clamps date filters | `WHERE date <= ?` pushed down, or Delta time travel | inside the same class |
| Data contracts | Pandera at the boundary | Pandera **plus** Delta constraints and Lakehouse Monitoring | contracts move closer to the data |
| Feature computation | pandas, vectorised | PySpark, same shape | `features/engineering` ported |
| Feature storage | optional Parquet cache | Databricks Feature Tables | `DatabricksFeatureRepository` |
| Feature lookup | in-process join | `FeatureLookup` + `create_training_set` | point-in-time join becomes native |
| Lineage | `FeatureSetMetadata` JSON | Unity Catalog lineage + MLflow | metadata carries across |
| Orchestration | Typer CLI | Databricks Workflows | new job definitions |
| Secrets | `.env`, git-ignored | Databricks secret scopes | config binding only |
| Access control | none (single user) | Unity Catalog grants, row filters, column masks | no application code |
| Experiment tracking | local MLflow (`./mlruns`) | Databricks MLflow | `ML__TRACKING_URI=databricks` |

The right-hand column is the whole point. Every row is a new implementation or a
config change. No row requires editing a model, a feature definition, a tool or
an agent.

## Point-in-time: the part that genuinely gets better

Locally, as-of correctness is enforced by clamping date filters per table, using
the availability classes in
[data/repositories/availability.py](../data/repositories/availability.py). It
works, and it answers *"what had happened by date D"*.

Databricks can answer a second, harder question: *"what did we believe on date
D"*. Delta time travel (`VERSION AS OF`, `TIMESTAMP AS OF`) reconstructs the
table as it stood, including rows that were later corrected. That distinction
matters for restated sales and back-dated promotions — the local Parquet layer
simply cannot express it, because it has no history of its own history.

```sql
-- Stage 2: what the table actually looked like on 2024-06-30
SELECT * FROM cpg_revenue_intelligence.gold.sales_daily
  TIMESTAMP AS OF '2024-06-30'
 WHERE product_id = :product_id
```

The `KNOWN_IN_ADVANCE` classification survives unchanged: promotion calendars and
holidays are still visible past the as-of date, for the same reason.

## Feature engineering: pandas to PySpark

The primitives were written to translate. Each is a groupby-window operation
with an explicit shift, which is exactly what a Spark window expresses:

```python
# Stage 1 - features/engineering/panel.py
panel.groupby(["product_id", "store_id"])["units"].shift(7)
```

```python
# Stage 2
window = Window.partitionBy("product_id", "store_id").orderBy("date")
frame.withColumn("lag_7_units", F.lag("units", 7).over(window))
```

The shift discipline transfers directly: `rolling_on_shifted` becomes
`F.avg(...).over(window.rowsBetween(-7, -1))`, where the `-1` upper bound is the
same guarantee — the current row is excluded.

**What must be checked during the port.** Pandas `rolling` counts *rows*; Spark's
`rangeBetween` can count *days*. On a panel with gaps those differ, and the
difference is silent. The daily panel here is dense, so row-based windows are
correct — but that is a property of this data, not a general one, and the port
should assert it rather than assume it.

## Feature tables

```python
# Stage 2 sketch - registration
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
fe.create_table(
    name="cpg_revenue_intelligence.features.demand_features",
    primary_keys=["product_id", "store_id", "date"],
    timeseries_columns="date",          # enables native point-in-time joins
    df=demand_features_df,
)
```

```python
# Stage 2 sketch - training set with a point-in-time join
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=[
        FeatureLookup(
            table_name="cpg_revenue_intelligence.features.demand_features",
            lookup_key=["product_id", "store_id"],
            timestamp_lookup_key="date",   # as-of correctness, in the engine
        )
    ],
    label="units",
    exclude_columns=["revenue", "cost", "gross_profit"],
)
```

Two things collapse into platform features:

- The **warm-up window** and manual shifting become `timestamp_lookup_key`.
- The **target-derived column drop** (`revenue`, `cost`, `gross_profit`) becomes
  `exclude_columns`. Worth keeping the local guard anyway: a belt-and-braces
  exclusion costs nothing and the failure it prevents is a model that leaks its
  own target.

A model logged with `fe.log_model` records its feature lookups, so the
model-to-feature-version link becomes automatic rather than a convention someone
has to maintain. That is strictly better than
[`FeatureSetMetadata`](../features/contracts/specs.py), which relies on the
caller recording it.

## What does *not* move

| Stays exactly as it is | Why |
|---|---|
| `features/contracts/catalogue.py` | Feature *definitions* are business logic, not storage |
| `features/contracts/config.py` + `features.yaml` | Selection is configuration |
| `data/contracts/tables.py` | Schema expectations are platform-independent |
| `data/repositories/availability.py` | Which data is knowable when is a business fact |
| `features/datasets/builders.py` | Model framing does not depend on the engine |
| The leakage tests | The property being asserted is unchanged |

That list is the actual deliverable of Step 3. Everything in it would otherwise
have had to be rewritten during migration.

## Unity Catalog layout

```
catalog: cpg_revenue_intelligence
├── bronze     raw ingestion, audit columns
├── silver     validated, deduplicated, quarantined
├── gold       business-ready star schema        <- DataRepository reads here
├── features   feature tables                    <- FeatureRepository reads here
├── analytics  BI and Text-to-SQL surface
└── ml         models, experiments
```

The agent's service principal gets `SELECT` on `gold` and `features`, nothing
else. Read-only is enforced by the catalogue, not by application code — a grant
is a guarantee where an application check is a promise.

Column comments are written on every Gold table. Not documentation garnish: the
Step 14 Text-to-SQL agent reads them for schema discovery, so a missing comment
degrades agent accuracy directly. `describe_table` already returns a `comment`
column, empty locally because Parquet has no equivalent — the shape is stable so
the Stage 2 implementation fills it without changing callers.

## Migration order

1. **Bronze/Silver/Gold in Delta** — Auto Loader ingestion, Lakeflow pipelines.
2. **`DatabricksDataRepository`** — the existing test suite runs against it
   unchanged, which is the check that the abstraction held.
3. **Feature tables** — port `features/engineering` to PySpark, register tables.
4. **`DatabricksFeatureRepository`** — `FeatureLookup`-based training sets.
5. **Re-run the leakage suite against both** — features from the local path and
   the Databricks path must agree for the same as-of date. If they diverge, the
   port changed semantics, and that is exactly what needs catching.

Step 5 is the one that makes this real. Without it, "the logic did not change"
is an assertion rather than a result.

## Cost and scaling notes

- **SQL Warehouse over a cluster** for the repository. These are filtered reads
  on an agent's critical path; warehouse start-up latency is far lower than
  cluster spin-up, and serverless scales to zero between investigations.
- **Feature tables materialised on a schedule**, not computed per request. The
  local `materialise=False` default is right while definitions move; in
  production the economics invert — recomputing a 44M-row panel per request is
  absurd when a nightly job costs pennies.
- **Partition Gold facts by month**, matching the local layout, so partition
  pruning works the same way and a "last quarter" query stays cheap.
- **Z-order on `(product_id, store_id)`** — the filter combination nearly every
  repository call uses.
