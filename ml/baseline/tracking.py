"""MLflow tracking and model registration (brief sections 25-28).

Local MLflow now; Databricks MLflow in Stage 2. The only thing that changes is
the tracking URI - which is why the URI comes from settings rather than being
hard-coded, and why nothing here imports a Databricks module.

**What is logged, and why it is not optional.** A baseline number is consumed by
five downstream models and eventually presented to a business user as the reason
for a recommendation. When someone asks "where did 1,000 come from", the answer
has to be reconstructible: this model version, trained on this dataset version,
with these features at this feature version, on this code commit. Section 27
requires the triple; without it the number is attributable to "whatever the code
looked like at the time".

Artifacts are written to a temporary directory and uploaded, rather than left
beside the run, so the MLflow store stays the single source of truth.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from features.contracts.specs import FEATURE_VERSION, current_code_version
from ml.baseline.comparison import Candidate, ComparisonResult
from ml.baseline.training import TrainedBaseline

logger = get_logger(__name__)

EXPERIMENT_NAME = "baseline_sales"
REGISTERED_MODEL_NAME = "baseline_sales"

#: Bumped when the model's *definition* changes in a way that alters its
#: predictions - a new objective, a different promotion approach, a changed
#: feature set. Not bumped for a retrain on fresh data; that is what the MLflow
#: run id and dataset version are for.
MODEL_VERSION = "v1.0"


def configure_mlflow(settings: Settings | None = None) -> str:
    """Point MLflow at the configured store and ensure the experiment exists.

    Returns the experiment id. In Stage 2 ``ML__TRACKING_URI=databricks`` is the
    only change needed here.
    """
    config = (settings or get_settings()).ml
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.registry_uri:
        mlflow.set_registry_uri(config.registry_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(EXPERIMENT_NAME)
    )
    logger.info(
        "mlflow.configured",
        tracking_uri=config.tracking_uri,
        experiment=EXPERIMENT_NAME,
        experiment_id=experiment_id,
    )
    return str(experiment_id)


@contextmanager
def _artifact_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="baseline_artifacts_") as directory:
        yield Path(directory)


def _log_frame(frame: pd.DataFrame, name: str, directory: Path) -> None:
    if frame is None or frame.empty:
        return
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_json(payload: dict[str, Any], name: str, directory: Path) -> None:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    mlflow.log_artifact(str(path))


def log_candidate(
    candidate: Candidate,
    *,
    dataset_version: str,
    experiment_id: str,
    seed: int,
    parent_run_id: str | None = None,
) -> str:
    """Log one candidate as a run. Returns the run id.

    Every candidate is logged, not only the winner. The comparison is evidence
    for the selection, and evidence that was discarded is not evidence - a
    reviewer should be able to see what the seasonal naive actually scored.
    """
    trained = candidate.trained
    run_name = f"{trained.name}"

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, nested=parent_run_id is not None
    ) as run:
        mlflow.set_tags(
            {
                "model_type": trained.estimator.name,
                "promotion_approach": trained.approach.value,
                "model_version": MODEL_VERSION,
                "stage": "candidate",
            }
        )

        # Section 26: model type, hyperparameters, periods, versions.
        mlflow.log_params(
            {
                "model_type": trained.estimator.name,
                "promotion_approach": trained.approach.value,
                "seed": seed,
                "n_features": len(trained.feature_names),
                "dataset_version": dataset_version,
                "feature_version": FEATURE_VERSION,
                "model_version": MODEL_VERSION,
                "code_version": current_code_version() or "unknown",
                **trained.split.to_dict(),
                **{f"hp_{k}": v for k, v in trained.estimator.get_params().items()},
            }
        )

        for label, metric in trained.metrics.items():
            for key, value in metric.to_dict().items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    mlflow.log_metric(f"{label}_{key}", float(value))

        for label, metric in candidate.latent_metrics.items():
            for key, value in metric.to_dict().items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    mlflow.log_metric(f"latent_{label}_{key}", float(value))

        if trained.coverage is not None:
            for key, value in trained.coverage.to_dict().items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    mlflow.log_metric(f"interval_{key}", float(value))

        if candidate.backtest is not None:
            mlflow.log_metric("backtest_mean_wmape", candidate.backtest.mean_wmape)
            mlflow.log_metric("backtest_std_wmape", candidate.backtest.std_wmape)
            mlflow.log_metric("backtest_stable", float(candidate.backtest.is_stable))

        mlflow.log_metric("train_seconds", trained.train_seconds)
        mlflow.log_metric("predict_seconds", trained.predict_seconds)

        with _artifact_dir() as directory:
            importance = trained.estimator.feature_importance()
            if importance is not None:
                _log_frame(importance, "feature_importance", directory)

            if candidate.backtest is not None:
                _log_frame(candidate.backtest.to_frame(), "backtest_folds", directory)

            _log_json(
                {
                    "excluded_rows": trained.excluded_rows,
                    "feature_names": list(trained.feature_names),
                    "calibration": trained.calibration.to_dict() if trained.calibration else None,
                    "coverage": trained.coverage.to_dict() if trained.coverage else None,
                },
                "training_config",
                directory,
            )

        return run.info.run_id


def log_comparison(
    comparison: ComparisonResult,
    *,
    dataset_version: str,
    experiment_id: str,
    seed: int,
    panel_rows: int,
) -> str:
    """Log the whole comparison as a parent run with a child per candidate.

    Nested so the selection and the evidence for it live together. A winner
    logged alone leaves "why this one" unanswerable six weeks later.
    """
    started = datetime.now(UTC)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"comparison_{started:%Y%m%d_%H%M%S}"
    ) as parent:
        mlflow.set_tags({"stage": "comparison", "model_version": MODEL_VERSION})
        mlflow.log_params(
            {
                "candidates": len(comparison.candidates),
                "selected_model": comparison.selected.trained.estimator.name,
                "selected_approach": comparison.selected.trained.approach.value,
                "dataset_version": dataset_version,
                "feature_version": FEATURE_VERSION,
                "code_version": current_code_version() or "unknown",
                "seed": seed,
                "panel_rows": panel_rows,
            }
        )

        for candidate in comparison.candidates:
            log_candidate(
                candidate,
                dataset_version=dataset_version,
                experiment_id=experiment_id,
                seed=seed,
                parent_run_id=parent.info.run_id,
            )

        with _artifact_dir() as directory:
            _log_frame(comparison.to_frame(), "model_comparison", directory)
            (directory / "selection_rationale.md").write_text(
                comparison.summary(), encoding="utf-8"
            )
            mlflow.log_artifact(str(directory / "selection_rationale.md"))

        return parent.info.run_id


def register_selected(
    trained: TrainedBaseline,
    *,
    dataset_version: str,
    experiment_id: str,
    seed: int,
    evaluation_report: str | None = None,
) -> tuple[str, str]:
    """Log and register the selected model. Returns ``(run_id, model_uri)``.

    Registered under a stable name so Step 6 onward can load "the current
    baseline" without knowing which run produced it - the indirection MLflow's
    registry exists to provide, and the same one Unity Catalog will provide in
    Stage 2.

    A new version is created rather than an existing one overwritten (section
    27): a prediction made last week must remain reproducible after a retrain.
    """
    import mlflow.lightgbm
    import mlflow.sklearn

    from ml.baseline.models import LightGBMBaseline, RidgeBaseline

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"registered_{trained.name}"
    ) as run:
        mlflow.set_tags(
            {
                "stage": "registered",
                "model_type": trained.estimator.name,
                "promotion_approach": trained.approach.value,
                "model_version": MODEL_VERSION,
            }
        )
        mlflow.log_params(
            {
                "model_type": trained.estimator.name,
                "promotion_approach": trained.approach.value,
                "model_version": MODEL_VERSION,
                "dataset_version": dataset_version,
                "feature_version": FEATURE_VERSION,
                "code_version": current_code_version() or "unknown",
                "seed": seed,
                **trained.split.to_dict(),
            }
        )
        for label, metric in trained.metrics.items():
            for key, value in metric.to_dict().items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    mlflow.log_metric(f"{label}_{key}", float(value))

        estimator = trained.estimator
        if isinstance(estimator, LightGBMBaseline):
            mlflow.lightgbm.log_model(
                estimator.booster,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
        elif isinstance(estimator, RidgeBaseline) and estimator._pipeline is not None:
            mlflow.sklearn.log_model(
                estimator._pipeline,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
        else:
            # The seasonal naive has no serialisable artifact beyond its two
            # parameters. Logging those is enough to rebuild it exactly, and
            # pickling a trivial object would be ceremony.
            with _artifact_dir() as directory:
                _log_json(estimator.get_params(), "estimator_params", directory)

        with _artifact_dir() as directory:
            importance = estimator.feature_importance()
            if importance is not None:
                _log_frame(importance, "feature_importance", directory)
            _log_json(
                {
                    "model_version": MODEL_VERSION,
                    "dataset_version": dataset_version,
                    "feature_version": FEATURE_VERSION,
                    "code_version": current_code_version(),
                    "seed": seed,
                    "approach": trained.approach.value,
                    "feature_names": list(trained.feature_names),
                    "calibration": trained.calibration.to_dict() if trained.calibration else None,
                    "coverage": trained.coverage.to_dict() if trained.coverage else None,
                    "trained_at": datetime.now(UTC).isoformat(),
                },
                "model_metadata",
                directory,
            )
            if evaluation_report:
                (directory / "evaluation_report.md").write_text(
                    evaluation_report, encoding="utf-8"
                )
                mlflow.log_artifact(str(directory / "evaluation_report.md"))

        model_uri = f"runs:/{run.info.run_id}/model"
        logger.info(
            "baseline.registered",
            run_id=run.info.run_id,
            model=trained.estimator.name,
            model_uri=model_uri,
        )
        return run.info.run_id, model_uri
