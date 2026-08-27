# Demand forecasting — documentation index

Everything about the forecasting capability, and where to find it.

## A note on step numbering

The brief that specified this capability calls it **Step 6**, assuming Step 5 was
Baseline Sales. The repository built Baseline Sales as Step 4 and forecasting as
**Step 5**, so the commit history and some older docs say "Step 5".

They are the same thing. From here on the brief's numbering is used:

| Capability | Commit history | From now on |
|---|---|---|
| Baseline Sales | Step 4 | **Step 5** |
| Demand Forecasting | Step 5 | **Step 6** |
| Promo Uplift | Step 6 | **Step 7** |

The history is not rewritten — a renumbered commit log would be a worse lie than
a documented offset.

## Where things are

| Document | Contents |
|---|---|
| [`../models/demand_forecasting.md`](../models/demand_forecasting.md) | **Methodology.** The origin/target design, why not the alternatives, stockout handling, embargo, intervals, MLflow strategy, V2 roadmap |
| [`../models/model_cards/demand_forecasting_model_card.md`](../models/model_cards/demand_forecasting_model_card.md) | **Model card.** Intended use and non-use, training data, measured results, limitations, ethics |
| [`data_contract.md`](data_contract.md) | Grain, target, horizons, availability classes, forbidden features, quality thresholds, coverage limits |
| [`evaluation.md`](evaluation.md) | Every metric defined, the noise floor, FVA, per-bucket results, model selection, the hyperparameter search outcome |
| [`feature_catalogue.md`](feature_catalogue.md) | Per-feature table: definition, when it is knowable, derived leakage risk. Generated |
| [`step6_report.md`](step6_report.md) | The step's A–T deliverables against measured results |
| [`interview_guide.md`](interview_guide.md) | The system explained in plain language, as fifteen questions |
| [`../forecasting_databricks_migration.md`](../forecasting_databricks_migration.md) | Local → Databricks design. Design only; nothing implemented |

The methodology and model card stay in `docs/models/` alongside their Step 5
counterparts rather than moving here — relocating them would break links for
cosmetic tidiness.

## Where the code is

| Path | Purpose |
|---|---|
| `ml/forecasting/dataset.py` | The horizon self-join and the inference scaffold — the highest-risk module |
| `ml/forecasting/split.py` | Origin-based split with the embargo |
| `ml/forecasting/baselines.py` | `HorizonNaive`, `HorizonSeasonalNaive` |
| `ml/forecasting/train.py` | Fit, calibrate, score; the frozen feature schema |
| `ml/forecasting/tuning.py` | Seeded, capped random search |
| `ml/forecasting/quality.py` | Data-quality checks on the forecasting grain |
| `ml/forecasting/exceptions.py` | Typed failures carrying their own recoverability |
| `ml/forecasting/predict.py` | As-of validation and the fallback chain |
| `app/services/forecast_service.py` | The service seam |
| `app/tools/forecasting_tool.py` | The agent-facing contract |

## Commands

```powershell
# Train
uv run python scripts/train_forecast.py --smoke --no-track   # <60s correctness check
uv run python scripts/train_forecast.py --seed 42            # ~8 min, the deliverable
uv run python scripts/train_forecast.py --seed 42 --tune     # + the hyperparameter search
uv run python scripts/train_forecast.py --full               # every series; hours

# Evaluate
uv run ari evaluate-forecast
uv run ari forecast-quality

# Forecast
uv run ari forecast --product P00003 --store S00155 --horizon 28
uv run ari forecast --product P00003 --horizon 28 --daily

# Test
uv run pytest tests/forecasting -v
```

## Notebooks

`notebooks/forecasting/` — EDA, baseline comparison, model training, backtesting,
error analysis, forecast validation.
