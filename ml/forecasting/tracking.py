"""MLflow tracking for forecasting runs (brief sections 21-23).

Mirrors ``ml/baseline/tracking.py`` closely but cannot import it: that module
hard-codes ``EXPERIMENT_NAME = "baseline_sales"`` and types against
``TrainedBaseline``/``Candidate``. The shared parts - ``configure_mlflow``, the
artifact-directory helper - are imported; the rest is the same shape with
forecasting's own params and metrics.

Metrics are logged **per horizon bucket** rather than blended, for the reason
that runs through this whole step: one WMAPE averaged over 1..90 days describes
no decision anyone makes.
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
from ml.baseline.tracking import configure_mlflow as _configure_baseline_mlflow
from ml.forecasting.config import ForecastConfig
from ml.forecasting.train import TrainedForecaster

logger = get_logger(__name__)

EXPERIMENT_NAME = "revenue_intelligence_forecasting"
REGISTERED_MODEL_NAME = "demand_forecast"
MODEL_VERSION = "v1.0"


def configure_mlflow(settings: Settings | None = None) -> str:
    """Point MLflow at the store and ensure the forecasting experiment exists."""
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
    with tempfile.TemporaryDirectory(prefix="forecast_artifacts_") as directory:
        yield Path(directory)


def _log_frame(frame: pd.DataFrame | None, name: str, directory: Path) -> None:
    if frame is None or frame.empty:
        return
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_json(payload: Any, name: str, directory: Path) -> None:
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    mlflow.log_artifact(str(path))


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and pd.notna(value):
        return float(value)
    return None


def log_candidate(
    trained: TrainedForecaster,
    config: ForecastConfig,
    *,
    dataset_version: str,
    experiment_id: str,
    parent_run_id: str | None = None,
    fva: dict[str, float] | None = None,
    backtest: pd.DataFrame | None = None,
) -> str:
    """Log one candidate as a (possibly nested) run."""
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=trained.name,
        nested=parent_run_id is not None,
    ) as run:
        mlflow.set_tags(
            {
                "model_type": trained.name,
                "model_version": MODEL_VERSION,
                "stage": "candidate",
            }
        )
        mlflow.log_params(
            {
                "model_type": trained.name,
                "seed": config.sampling.seed,
                "n_features": len(trained.feature_names),
                "n_series": config.sampling.n_series,
                "max_horizon": config.max_horizon,
                "origin_stride_days": config.origins.stride_days,
                "horizons_per_origin": config.origins.horizons_per_origin,
                "embargo_days": config.validation.embargo_days,
                "dataset_version": dataset_version,
                "feature_version": FEATURE_VERSION,
                "model_version": MODEL_VERSION,
                "code_version": current_code_version() or "unknown",
                # One hash covering the entire configuration. Two runs sharing it
                # used the same setup; two that differ are not comparable, and
                # the difference is discoverable rather than argued about.
                "config_fingerprint": config.fingerprint(),
                **trained.split.to_dict(),
                **{f"hp_{k}": v for k, v in trained.estimator.get_params().items()},
            }
        )

        for label, metrics in trained.metrics.items():
            for key, value in metrics.to_dict().items():
                numeric = _numeric(value)
                if numeric is not None:
                    mlflow.log_metric(f"{label}_{key}", numeric)

        # The breakdown that matters more than the headline.
        for bucket, metrics in trained.bucket_metrics.items():
            mlflow.log_metric(f"bucket_{bucket}_wmape", metrics.wmape)
            mlflow.log_metric(f"bucket_{bucket}_bias_pct", metrics.bias_pct)
            mlflow.log_metric(f"bucket_{bucket}_n", float(metrics.n))

        for bucket, value in (fva or {}).items():
            mlflow.log_metric(f"fva_{bucket}_pp", value)

        mlflow.log_metric("train_seconds", trained.train_seconds)
        mlflow.log_metric("predict_seconds", trained.predict_seconds)

        with _artifact_dir() as directory:
            _log_frame(trained.estimator.feature_importance(), "feature_importance", directory)
            _log_frame(backtest, "backtest_folds", directory)
            _log_json(
                {
                    "feature_names": trained.feature_names,
                    "calibration": (
                        trained.calibration.to_dict() if trained.calibration else None
                    ),
                    "config": config.model_dump(mode="json"),
                },
                "training_config",
                directory,
            )

        return run.info.run_id


def log_comparison(
    candidates: list[TrainedForecaster],
    selected: TrainedForecaster,
    config: ForecastConfig,
    *,
    dataset_version: str,
    experiment_id: str,
    dataset_rows: int,
    comparison_table: pd.DataFrame | None = None,
    rationale: str = "",
    fva: dict[str, dict[str, float]] | None = None,
) -> str:
    """Log the whole comparison as a parent run with nested candidates."""
    started = datetime.now(UTC)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"comparison_{started:%Y%m%d_%H%M%S}",
    ) as parent:
        mlflow.set_tags({"stage": "comparison", "model_version": MODEL_VERSION})
        mlflow.log_params(
            {
                "candidates": len(candidates),
                "selected_model": selected.name,
                "dataset_version": dataset_version,
                "feature_version": FEATURE_VERSION,
                "code_version": current_code_version() or "unknown",
                "config_fingerprint": config.fingerprint(),
                "dataset_rows": dataset_rows,
                "n_series": config.sampling.n_series,
            }
        )

        for candidate in candidates:
            log_candidate(
                candidate,
                config,
                dataset_version=dataset_version,
                experiment_id=experiment_id,
                parent_run_id=parent.info.run_id,
                fva=(fva or {}).get(candidate.name),
            )

        with _artifact_dir() as directory:
            _log_frame(comparison_table, "model_comparison", directory)
            if rationale:
                path = directory / "selection_rationale.md"
                path.write_text(rationale, encoding="utf-8")
                mlflow.log_artifact(str(path))

        return parent.info.run_id


def register_selected(
    trained: TrainedForecaster,
    config: ForecastConfig,
    *,
    dataset_version: str,
    experiment_id: str,
    evaluation_report: str | None = None,
) -> tuple[str, str | None]:
    """Log and register the selected model.

    Returns ``(run_id, model_uri)``. The URI is ``None`` when the selected
    estimator has no serialisable artifact - the naive benchmarks - because a
    URI that resolves to nothing is worse than an explicit absence.
    """
    import mlflow.lightgbm
    import mlflow.xgboost

    from ml.baseline.models import LightGBMBaseline
    from ml.forecasting.xgboost_model import XGBoostForecaster

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"registered_{trained.name}"
    ) as run:
        mlflow.set_tags(
            {"stage": "registered", "model_type": trained.name, "model_version": MODEL_VERSION}
        )

        estimator = trained.estimator
        registered = False
        if isinstance(estimator, LightGBMBaseline) and estimator.booster is not None:
            mlflow.lightgbm.log_model(
                estimator.booster, name="model", registered_model_name=REGISTERED_MODEL_NAME
            )
            registered = True
        elif isinstance(estimator, XGBoostForecaster) and estimator.booster is not None:
            mlflow.xgboost.log_model(
                estimator.booster, name="model", registered_model_name=REGISTERED_MODEL_NAME
            )
            registered = True
        else:
            # Naive benchmarks have no artifact worth serialising; their
            # parameters fully describe them.
            with _artifact_dir() as directory:
                _log_json(estimator.get_params(), "estimator_params", directory)

        with _artifact_dir() as directory:
            _log_frame(estimator.feature_importance(), "feature_importance", directory)
            _log_json(
                {
                    "model_version": MODEL_VERSION,
                    "dataset_version": dataset_version,
                    "feature_version": FEATURE_VERSION,
                    "code_version": current_code_version(),
                    "config_fingerprint": config.fingerprint(),
                    "estimator": trained.name,
                    "feature_names": trained.feature_names,
                    "split": trained.split.to_dict(),
                    "calibration": (
                        trained.calibration.to_dict() if trained.calibration else None
                    ),
                    "trained_at": datetime.now(UTC).isoformat(),
                },
                "model_metadata",
                directory,
            )
            if evaluation_report:
                path = directory / "evaluation_report.md"
                path.write_text(evaluation_report, encoding="utf-8")
                mlflow.log_artifact(str(path))

        # `None` when nothing was actually logged, rather than a URI that looks
        # valid and resolves to nothing. A naive benchmark has no model artifact,
        # and handing back `runs:/<id>/model` for it would send a caller to a
        # 404 instead of telling them plainly that there is nothing to load -
        # the same "never return something fake" rule the rest of the platform
        # applies to confidence scores and forecasts.
        model_uri = f"runs:/{run.info.run_id}/model" if registered else None
        logger.info(
            "forecast.registered",
            run_id=run.info.run_id,
            model=trained.name,
            registered=registered,
        )
        return run.info.run_id, model_uri


__all__ = [
    "EXPERIMENT_NAME",
    "MODEL_VERSION",
    "REGISTERED_MODEL_NAME",
    "_configure_baseline_mlflow",
    "configure_mlflow",
    "log_candidate",
    "log_comparison",
    "register_selected",
]
