"""End-to-end forecasting training run.

Orchestrates the sequence in one visible place: sample series, build history,
build the horizon dataset, split with an embargo, train every candidate,
backtest, compare, select, evaluate, persist, track.

Two guards are carried over from Step 4 because both were learned expensively:

* **The evaluation report is written before tracking.** A three-hour Step 4 run
  was lost when MLflow rejected its own store *after* every model had trained.
  The report is a deliverable - it is the justification for the selection - and
  reconstructing it means retraining.
* **Tracking failures do not fail the run.** Bookkeeping must never destroy the
  thing it is keeping books on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.baseline.evaluation import BaselineMetrics
from ml.forecasting.backtest import (
    backtest_by_horizon,
    stability_by_bucket,
)
from ml.forecasting.baselines import HorizonSeasonalNaive, attach_seasonal_reference
from ml.forecasting.config import ForecastConfig, get_forecast_config
from ml.forecasting.dataset import (
    HorizonDataset,
    build_history,
    build_horizon_dataset,
)
from ml.forecasting.evaluate import (
    PREDICTED,
    format_bucket_table,
    hierarchy_table,
    horizon_error_grows,
    revenue_impact,
    seasonal_naive_scale,
    segment_errors,
    zero_demand_summary,
)
from ml.forecasting.exceptions import FeatureGenerationError
from ml.forecasting.model import FittedForecastModel
from ml.forecasting.sampling import sample_series
from ml.forecasting.split import OriginSplit, build_origin_split, slice_fold
from ml.forecasting.train import (
    TrainedForecaster,
    build_estimator,
    forecast_value_added,
    train_forecaster,
)
from ml.forecasting.tuning import TuningResult, best_params, tune

logger = get_logger(__name__)

#: The benchmark FVA is measured against, and the fallback at serving time.
BENCHMARK = "horizon_seasonal_naive"


@dataclass
class ForecastPipelineResult:
    """Everything a training run produced."""

    candidates: list[TrainedForecaster]
    selected: TrainedForecaster
    split: OriginSplit
    config: ForecastConfig
    dataset_rows: int
    n_series: int
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    fva: dict[str, dict[str, float]] = field(default_factory=dict)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    hierarchy: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    revenue: dict[str, float] = field(default_factory=dict)
    zero_demand: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)
    tuning: TuningResult | None = None
    duration_seconds: float = 0.0
    model_path: Path | None = None
    mlflow_run_id: str | None = None

    def report(self) -> str:
        """The evaluation report, written before tracking runs."""
        lines = [
            "# Demand Forecasting - Evaluation Report",
            "",
            f"Series: {self.n_series:,} | Dataset rows: {self.dataset_rows:,}",
            f"Split: {self.split.describe()}",
            f"Config fingerprint: {self.config.fingerprint()}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "## Model comparison",
            "",
            self.comparison.to_string(index=False) if not self.comparison.empty else "(none)",
            "",
            f"**Selected: {self.selected.name}**",
            "",
        ]
        lines.extend(f"- {reason}" for reason in self.rationale)

        lines += ["", "## Accuracy by horizon bucket", ""]
        lines.append(format_bucket_table(self.selected.bucket_metrics))
        lines += [
            "",
            "Reported per bucket rather than blended. Forecast error grows with",
            "horizon by nature, so a single number averaged over 1-90 days",
            "describes no decision anyone actually makes.",
        ]

        if self.fva:
            lines += ["", "## Forecast Value Added (WMAPE percentage points)", ""]
            for model, buckets in self.fva.items():
                formatted = ", ".join(f"{k}: {v:+.1%}" for k, v in buckets.items())
                lines.append(f"- **{model}** vs {BENCHMARK}: {formatted}")
            lines += [
                "",
                "Positive means the model beat what a planner gets unaided. A bucket",
                "at or below zero is a bucket where the seasonal naive should be used",
                "instead - reported rather than smoothed over.",
            ]

        if self.tuning is not None and self.tuning.trials:
            lines += ["", "## Hyperparameter search", "", self.tuning.summary(), ""]
            lines.append(self.tuning.to_frame().head(8).to_string(index=False))
            lines += [
                "",
                "Deliberately small: the model sits near the irreducible noise floor,",
                "so hyperparameters compete for a few percentage points at most. A gain",
                "under the fold-to-fold standard deviation is noise, and the defaults",
                "are kept rather than adopting it.",
            ]

        if not self.stability.empty:
            lines += ["", "## Backtest stability by bucket", "",
                      self.stability.to_string(index=False)]

        if not self.hierarchy.empty:
            lines += [
                "", "## Accuracy by aggregation level", "",
                self.hierarchy.to_string(index=False),
                "",
                "Bottom-up aggregation is exactly coherent by construction, so no",
                "reconciliation is applied. What this shows is the price of that",
                "choice: independent errors average out as you aggregate, so the",
                "regional number deserves more trust than the SKU number - by this",
                "much.",
            ]

        if self.revenue:
            lines += ["", "## Revenue impact", ""]
            lines.append(
                f"- Forecast revenue error: {self.revenue.get('revenue_error', 0):,.0f} "
                f"({self.revenue.get('revenue_error_pct', 0):+.1%})"
            )
            lines.append(
                "- Priced at the **planned** price known at forecast time, never the "
                "realised price. Using the latter would fold pricing surprise into a "
                "demand-accuracy figure."
            )

        if self.zero_demand:
            lines += ["", "## Zero and intermittent demand", ""]
            lines.append(
                f"- {self.zero_demand.get('zero_share', 0):.1%} of rows are zero; "
                f"{self.zero_demand.get('under_five_share', 0):.1%} are under five units."
            )
            lines.append(
                "- This governs which metrics mean anything: MAPE is undefined at zero "
                "and unstable near it, which is why WMAPE is the headline and MAPE is "
                "computed only over non-zero actuals with its exclusion count stated."
            )

        if not self.segments.empty:
            lines += ["", "## Worst segments", "",
                      self.segments.head(10).to_string(index=False)]

        return "\n".join(lines)


def train_forecast_pipeline(
    repository: DataRepository,
    *,
    config: ForecastConfig | None = None,
    models: tuple[str, ...] | None = None,
    run_backtest: bool = True,
    run_tuning: bool = False,
    track: bool = True,
    output_dir: Path | None = None,
    settings: Settings | None = None,
) -> ForecastPipelineResult:
    """Run the whole thing."""
    started = time.perf_counter()
    config = config or get_forecast_config()
    settings = settings or get_settings()
    candidate_names = models or config.models.enabled()

    # -- data ---------------------------------------------------------------
    sample = sample_series(
        repository,
        n_series=config.sampling.n_series,
        seed=config.sampling.seed,
        stratify_by_volume=config.sampling.stratify_by_volume,
    )
    history = build_history(repository, config, sample)
    if history.empty:
        raise FeatureGenerationError(
            "the feature history is empty; generate a dataset first",
            stage="build_history",
        )

    as_of = pd.to_datetime(history["date"]).dt.date.max()
    view = repository.as_of(as_of)

    dataset = build_horizon_dataset(history, view, config, sample)
    if dataset.frame.empty:
        raise FeatureGenerationError(
            "the horizon dataset is empty; check the origin stride and warmup",
            stage="build_horizon_dataset",
        )

    # The seasonal benchmark needs a reference from the target date, which the
    # generic feature panel does not carry.
    dataset = HorizonDataset(
        frame=attach_seasonal_reference(dataset.frame, history),
        feature_names=[*dataset.feature_names, "seasonal_reference"],
        excluded=dataset.excluded,
    )

    split = build_origin_split(dataset.frame, config)

    # -- optional tuning ----------------------------------------------------
    # Runs before the candidates and scores on the VALIDATION fold only. The
    # test fold stays untouched until selection, or every number after this
    # point would be a self-report.
    tuned_params: dict[str, Any] = {}
    tuning_result: TuningResult | None = None
    if run_tuning:
        tuning_result = tune(
            dataset,
            split,
            config,
            lambda params: build_estimator(
                "lightgbm", seed=config.sampling.seed, params=params or None
            ),
        )
        tuned_params = best_params(
            tuning_result, threshold_pp=config.tuning.min_improvement_pp
        )
        logger.info("forecast.tuning_outcome", adopted=bool(tuned_params))

    # -- candidates ---------------------------------------------------------
    trained: list[TrainedForecaster] = []
    for name in candidate_names:
        # Tuned parameters apply only to the estimator they were searched for.
        params = tuned_params if name == "lightgbm" and tuned_params else None
        estimator = build_estimator(name, seed=config.sampling.seed, params=params)
        trained.append(train_forecaster(dataset, estimator, config, split))

    benchmark = next((t for t in trained if t.name == BENCHMARK), None)
    fva: dict[str, dict[str, float]] = {}
    if benchmark is not None:
        fva = {
            candidate.name: forecast_value_added(
                candidate.bucket_metrics, benchmark.bucket_metrics
            )
            for candidate in trained
            if candidate.name != BENCHMARK
        }

    # The MASE denominator comes from the TRAINING fold. Taking it from the
    # evaluation fold would make the metric partly self-referential.
    mase_scale = seasonal_naive_scale(
        slice_fold(dataset.frame, split.train_start, split.train_end)
    )
    comparison, selected, rationale = _compare(trained, fva, config, mase_scale)

    # -- diagnostics on the winner -----------------------------------------
    test = slice_fold(dataset.frame, split.test_start, split.test_end)
    scored = test.assign(**{PREDICTED: selected.predict(test)})

    backtest_table = pd.DataFrame()
    stability = pd.DataFrame()
    if run_backtest:
        backtest_table = backtest_by_horizon(
            dataset,
            lambda: build_estimator(selected.name, seed=config.sampling.seed),
            config,
        )
        stability = stability_by_bucket(backtest_table)

    result = ForecastPipelineResult(
        candidates=trained,
        selected=selected,
        split=split,
        config=config,
        dataset_rows=len(dataset),
        n_series=len(sample),
        comparison=comparison,
        fva=fva,
        backtest=backtest_table,
        stability=stability,
        hierarchy=hierarchy_table(scored),
        segments=segment_errors(scored),
        revenue=revenue_impact(scored),
        zero_demand=zero_demand_summary(scored),
        rationale=rationale,
        tuning=tuning_result,
        duration_seconds=time.perf_counter() - started,
    )

    # -- persist ------------------------------------------------------------
    directory = output_dir or default_output_dir(config, settings=settings)
    fallback = next((t.estimator for t in trained if t.name == BENCHMARK), None)

    model = FittedForecastModel(
        repository,
        trained=selected,
        config=config,
        pairs=sample.pairs,
        fallback=fallback if isinstance(fallback, HorizonSeasonalNaive) else None,
        metrics={
            f"bucket_{bucket}_wmape": metrics.wmape
            for bucket, metrics in selected.bucket_metrics.items()
        }
        | {"test_wmape": selected.metrics["test"].wmape if "test" in selected.metrics else 0.0},
    )
    model.save(directory)
    result.model_path = directory

    # Written before tracking, deliberately.
    (directory / "evaluation_report.md").write_text(result.report(), encoding="utf-8")

    if track:
        result.mlflow_run_id = _track(result, dataset, repository, settings)

    logger.info(
        "forecast.pipeline_completed",
        selected=selected.name,
        rows=len(dataset),
        series=len(sample),
        duration_seconds=round(result.duration_seconds, 1),
    )
    return result


def _compare(
    trained: list[TrainedForecaster],
    fva: dict[str, dict[str, float]],
    config: ForecastConfig,
    mase_scale: float = float("nan"),
) -> tuple[pd.DataFrame, TrainedForecaster, list[str]]:
    """Build the comparison table and select from it.

    Selection is on test WMAPE, with a simplicity preference: if the seasonal
    naive is within two percentage points of the best model, it wins. Section 9
    is explicit that the most complex model does not win by default, and a
    benchmark that holds its own is telling you the signal is simple.
    """
    rows = []
    for candidate in trained:
        test: BaselineMetrics | None = candidate.metrics.get("test")
        bucket_fva = fva.get(candidate.name, {})
        rows.append(
            {
                "model": candidate.name,
                "wmape": test.wmape if test else float("nan"),
                "mae": test.mae if test else float("nan"),
                "rmse": test.rmse if test else float("nan"),
                "mape": test.mape if test and test.mape is not None else float("nan"),
                # MAE scaled by the in-sample error of a weekly seasonal naive.
                # Below 1.0 means the model beats doing nothing.
                "mase": (
                    test.mae / mase_scale
                    if test and np.isfinite(mase_scale) and mase_scale > 0
                    else float("nan")
                ),
                "bias_pct": test.bias_pct if test else float("nan"),
                "mean_fva_pp": (
                    sum(bucket_fva.values()) / len(bucket_fva) if bucket_fva else float("nan")
                ),
                "error_grows_with_h": horizon_error_grows(candidate.bucket_metrics),
                "train_seconds": round(candidate.train_seconds, 2),
                "predict_seconds": round(candidate.predict_seconds, 3),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("wmape").reset_index(drop=True)
    rationale: list[str] = []

    ranked = sorted(trained, key=lambda c: c.metrics["test"].wmape if "test" in c.metrics else 1e9)
    best = ranked[0]
    selected = best
    rationale.append(
        f"{best.name} is most accurate at WMAPE {best.metrics['test'].wmape:.1%}."
    )

    naive = next((c for c in ranked if c.name == BENCHMARK), None)
    if naive is not None and naive is not best:
        gap = naive.metrics["test"].wmape - best.metrics["test"].wmape
        if gap <= 0.02:
            selected = naive
            rationale.append(
                f"Selected the seasonal naive instead: it is within {gap:.1%} of "
                f"{best.name}, which does not justify the added complexity and "
                f"training cost."
            )
        else:
            rationale.append(
                f"The seasonal naive benchmark trails by {gap:.1%}, so the added "
                f"complexity is earning its place."
            )

    if not horizon_error_grows(selected.bucket_metrics):
        rationale.append(
            "WARNING - error does not grow with horizon for the selected model. "
            "Forecasting 90 days out should be harder than forecasting tomorrow; "
            "if it is not, suspect that the origin/target join is leaking "
            "information from the target date."
        )

    for name, buckets in fva.items():
        weak = [b for b, value in buckets.items() if value <= 0]
        if weak and name == selected.name:
            rationale.append(
                f"{name} adds no value over the seasonal naive at {', '.join(weak)}; "
                f"at those horizons the benchmark is the honest choice."
            )

    return comparison, selected, rationale


def _track(
    result: ForecastPipelineResult,
    dataset: HorizonDataset,
    repository: DataRepository,
    settings: Settings,
) -> str | None:
    """Log to MLflow, never failing the run if it cannot.

    Learned the hard way in Step 4: a completed multi-hour training run was
    discarded because the tracking backend rejected its own URI *after* every
    model had already been fitted.
    """
    try:
        from ml.forecasting.tracking import (
            configure_mlflow,
            log_comparison,
            register_selected,
        )

        experiment_id = configure_mlflow(settings)
        dataset_version = repository.dataset_version()

        log_comparison(
            result.candidates,
            result.selected,
            result.config,
            dataset_version=dataset_version,
            experiment_id=experiment_id,
            dataset_rows=result.dataset_rows,
            comparison_table=result.comparison,
            rationale="\n".join(f"- {r}" for r in result.rationale),
            fva=result.fva,
        )
        run_id, _ = register_selected(
            result.selected,
            result.config,
            dataset_version=dataset_version,
            experiment_id=experiment_id,
            evaluation_report=result.report(),
        )
        return run_id
    except Exception as exc:  # noqa: BLE001 - tracking must not fail the run
        logger.warning(
            "forecast.tracking_failed",
            error=f"{type(exc).__name__}: {exc}",
            note="the model and evaluation report were still written",
        )
        return None


def default_output_dir(
    config: ForecastConfig, *, settings: Settings | None = None
) -> Path:
    """Where artifacts go when the caller does not say.

    A sampled or smoke run gets its own directory. Step 4 learned this the
    expensive way: a 400-pair verification run silently replaced a model trained
    on the full panel, and the only visible trace was a smaller calibration count
    buried in the metadata sidecar.
    """
    root = (settings or get_settings()).project_root / "data" / "local" / "models"
    full = config.sampling.n_series >= 6000
    return root / ("forecasting" if full else "forecasting_sampled")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
