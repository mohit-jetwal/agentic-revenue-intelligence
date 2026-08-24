# Databricks (Stage 2)

Empty until Stage 1 is validated end to end. This document records the mapping
so the migration is a known quantity rather than a discovery exercise.

**No AWS.** No S3, ECR, ECS, EKS, Lambda, API Gateway, RDS, ElastiCache,
CloudWatch, IAM or Secrets Manager appears anywhere in this project. Production
is Databricks-native throughout.

## Stage 1 → Stage 2 mapping

| Concern | Stage 1 (local) | Stage 2 (Databricks) | Code that changes |
|---|---|---|---|
| Analytical data | Parquet + DuckDB | Delta tables in Unity Catalog, read via Databricks SQL | `DatabricksDataRepository` only |
| Application state | SQLite via SQLAlchemy | Lakehouse-managed store | repository implementation only |
| Ingestion | generator writes Parquet | Auto Loader → Bronze Delta | new notebooks |
| Transformation | pandas / SQL | Lakeflow Declarative Pipelines or notebooks | new notebooks |
| Features | computed in-model | Databricks Feature Engineering tables | feature lookup in model code |
| Experiment tracking | local MLflow (`./mlruns`) | Databricks MLflow | `ML__TRACKING_URI=databricks` |
| Model registry | local MLflow registry | Unity Catalog model registry | `DatabricksModelRegistry` only |
| Model inference | in-process | Databricks Model Serving endpoints | registry implementation only |
| Vector store | Chroma | Databricks Vector Search (Delta Sync index) | `DatabricksVectorSearchStore` only |
| Orchestration | Typer CLI | Databricks Workflows / Jobs | new job definitions |
| Secrets | `.env` (git-ignored) | Databricks secret scopes + service principal | config binding only |
| Access control | none (single user) | Unity Catalog grants, table ACLs, row filters | no application code |
| Monitoring | JSON logs + `/metrics` | system tables, Lakehouse Monitoring, MLflow | no application code |

The right-hand column is the point of the exercise. Every row changes one
implementation class or adds a notebook. No row requires editing a model, a
tool, an agent or the graph — which is the property the three seams
(`DataRepository`, `ToolResult`, `Container`) exist to protect.

## Planned layout

```
databricks/
├── bronze/      # Auto Loader ingestion, raw Delta with audit columns
├── silver/      # schema validation, dedup, referential integrity, quarantine
├── gold/        # business-ready star schema
├── features/    # feature tables for demand, pricing, promotion
├── ml/          # training, evaluation, registration, serving deployment
└── workflows/   # job definitions: daily inference vs scheduled retraining
```

## Unity Catalog

```
catalog:  cpg_revenue_intelligence
schemas:  bronze | silver | gold | features | analytics | ml
```

The agent's service principal receives `SELECT` on `gold` and nothing else.
Read-only access is enforced by the catalog, not by application code — an
application-level check is a promise, a grant is a guarantee.

Column comments are written on every Gold table. They are not documentation
garnish: the Text-to-SQL agent reads them for schema discovery, so a missing
comment degrades agent accuracy directly.

## Notes carried forward

- Prefer a **SQL Warehouse** over a cluster for the repository. These are
  filtered reads on the agent's critical path, and warehouse start-up latency is
  far lower than cluster spin-up.
- Prefer a **served endpoint** over loading model artifacts into the API
  process: it keeps model dependencies out of the application image and lets
  model and application scale independently.
- Use a **Delta Sync** vector index rather than direct-access, so the policy
  corpus stays current as the source table changes without manual re-embedding.
- `dataset_version()` should return the Delta table version from
  `DESCRIBE HISTORY`, giving every recommendation an exact, reproducible data
  reference.
