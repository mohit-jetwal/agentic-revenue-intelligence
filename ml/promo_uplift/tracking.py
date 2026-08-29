"""MLflow tracking for uplift runs (brief section 27).

Same shape as :mod:`ml.forecasting.tracking`, with one difference that matters:
**the treatment and control definitions are logged as parameters.**

For a forecasting run, the parameters describe how a number was produced. For a
causal run they describe *what the number means*. Change ``washout_days`` from 0
to 10 and the estimate stops being "effect during the promotion" and becomes
"effect net of pull-forward" - a different quantity, not a better estimate of the
same one. Two runs with different treatment fingerprints are not comparable, and
without the definition in the run there is no way to know that afterwards.

So ``treatment_definition`` goes in as a readable sentence alongside the config
hash. The hash detects that something changed; the sentence says what.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from app.config.settings import Settings, get_settings
from app.observability.logging import get_logger
from features.contracts.specs import FEATURE_VERSION, current_code_version
from ml.promo_uplift.pipeline import UpliftRun, report

logger = get_logger(__name__)

EXPERIMENT_NAME = "revenue_intelligence_promo_uplift"
REGISTERED_MODEL_NAME = "promo_uplift"
MODEL_VERSION = "v1.0"


def configure_mlflow(settings: Settings | None = None) -> str:
    """Point MLflow at the store and ensure the uplift experiment exists."""
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
    with tempfile.TemporaryDirectory(prefix="uplift_artifacts_") as directory:
        yield Path(directory)


def _log_frame(frame: pd.DataFrame | None, name: str, directory: Path) -> None:
    if frame is None or frame.empty:
        return
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_text(text: str, name: str, directory: Path) -> None:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(path))


def _log_json(payload: Any, name: str, directory: Path) -> None:
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    mlflow.log_artifact(str(path))


def track_run(
    run: UpliftRun,
    *,
    settings: Settings | None = None,
    run_name: str | None = None,
) -> str | None:
    """Log one uplift analysis. Returns the run id, or ``None`` if tracking failed.

    Wrapped in a broad exception handler on purpose. A tracking-store failure
    must never destroy a completed analysis - Step 6 lost a three-hour run to
    exactly that, and the caller writes its report to disk before this is called
    so the result survives regardless.
    """
    try:
        experiment_id = configure_mlflow(settings)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("promo_uplift.mlflow_unavailable", error=str(exc))
        return None

    try:
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=run_name or "uplift"
        ) as active:
            _log_params(run)
            _log_metrics(run)
            with _artifact_dir() as directory:
                _log_text(report(run), "uplift_report.md", directory)
                _log_frame(run.comparison(), "method_comparison", directory)
                _log_frame(run.event_impact, "event_impact", directory)
                if run.sensitivity:
                    _log_frame(run.sensitivity.to_frame(), "sensitivity", directory)
                _log_frame(run.balance.to_frame(), "covariate_balance", directory)
                for column, frame in run.segments.items():
                    _log_frame(frame, f"segments_{column}", directory)
                _log_json(_assumptions(run), "assumptions", directory)
            logger.info("promo_uplift.tracked", run_id=active.info.run_id)
            return str(active.info.run_id)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("promo_uplift.tracking_failed", error=str(exc))
        return None


def _log_params(run: UpliftRun) -> None:
    config = run.config
    mlflow.log_params(
        {
            # The definitional parameters. These are what make two runs
            # comparable or not, so they lead.
            "treatment_definition": config.treatment_definition()[:490],
            "min_discount_depth": config.treatment.min_discount_depth,
            "washout_days": config.treatment.washout_days,
            "min_duration_days": config.treatment.min_duration_days,
            "control_window_days": config.controls.same_series_window_days,
            "cross_sectional_controls": config.controls.use_cross_sectional_controls,
            "exclude_censored_rows": config.stockouts.exclude_censored_rows,
            "estimand": "ATT",
            # Method parameters.
            "propensity_model": config.propensity.model,
            "propensity_clip": str(config.propensity.clip),
            "stabilise_weights_at": config.propensity.stabilise_weights_at,
            "outcome_objective": config.outcome_model.transform,
            "cross_fitting_scheme": config.cross_fitting.scheme,
            "cross_fitting_folds": config.cross_fitting.n_folds,
            # Provenance.
            "config_fingerprint": config.fingerprint(),
            "feature_version": FEATURE_VERSION,
            "code_version": current_code_version(),
            "selected_method": run.selected or "none",
        }
    )


def _log_metrics(run: UpliftRun) -> None:
    metrics: dict[str, float] = {
        "treated_rows": float(run.pool.treated_rows),
        "control_rows": float(run.pool.control_rows),
        "events": float(len(run.analysis.events)),
        "overlap_trimmed_share": run.overlap.trimmed_share,
        "effective_sample_fraction": run.overlap.effective_sample_fraction,
        "max_standardised_difference": run.balance.max_smd(),
        "unbalanced_covariates": float(len(run.balance.unbalanced)),
        "elapsed_seconds": run.elapsed_seconds,
    }
    if run.overlap.auc is not None:
        metrics["propensity_auc"] = run.overlap.auc

    # One set of metrics per method, so the comparison survives in MLflow rather
    # than only in the report artifact.
    for name, estimate in run.estimates.items():
        metrics[f"{name}__uplift_pct"] = estimate.ate_pct
        metrics[f"{name}__uplift_units"] = estimate.ate
        if estimate.standard_error is not None:
            metrics[f"{name}__standard_error"] = estimate.standard_error
        if name in run.impacts:
            impact = run.impacts[name]
            metrics[f"{name}__incremental_units"] = impact.incremental_units
            metrics[f"{name}__incremental_profit"] = impact.incremental_profit
            if impact.roi is not None:
                metrics[f"{name}__roi"] = impact.roi

    if run.placebo:
        metrics["placebo_effect_pct"] = run.placebo.effect_pct
        metrics["placebo_passed"] = float(run.placebo.passed)
    if run.sensitivity:
        metrics["sensitivity_spread"] = run.sensitivity.spread()
        metrics["sensitivity_relative_spread"] = run.sensitivity.relative_spread()
    if run.parallel_trends:
        metrics["parallel_trends_p"] = run.parallel_trends.p_value
        metrics["parallel_trends_passed"] = float(run.parallel_trends.parallel)
    if run.ground_truth:
        metrics["ground_truth_expected_pct"] = run.ground_truth.expected_pct
        metrics["ground_truth_absolute_error"] = run.ground_truth.absolute_error

    mlflow.log_metrics(metrics)


def _assumptions(run: UpliftRun) -> dict[str, Any]:
    """The assumption record, logged as a first-class artifact.

    Section 27 asks for assumptions to be tracked. They are logged as structured
    JSON rather than prose so a later run can be diffed against this one - which
    is the only way anybody notices that an estimate's meaning changed.
    """
    headline = run.headline
    return {
        "estimand": "ATT - average effect on the promotions that ran",
        "treatment_definition": run.config.treatment_definition(),
        "selected_method": run.selected,
        "selection_reason": run.selection_reason,
        "validation_status": run.validation_status,
        "method_assumptions": {
            name: estimate.assumptions for name, estimate in run.estimates.items()
        },
        "headline_assumptions": headline.assumptions if headline else [],
        "warnings": run.warnings(),
        "estimators_unavailable": run.failures,
        "diagnostics": {
            "overlap": run.overlap.summary(),
            "balance": run.balance.summary(),
            "placebo": run.placebo.summary() if run.placebo else None,
            "sensitivity": run.sensitivity.summary() if run.sensitivity else None,
            "parallel_trends": run.parallel_trends.summary()
            if run.parallel_trends
            else None,
            "ground_truth": run.ground_truth.summary() if run.ground_truth else None,
        },
    }


__all__ = [
    "EXPERIMENT_NAME",
    "MODEL_VERSION",
    "REGISTERED_MODEL_NAME",
    "configure_mlflow",
    "track_run",
]
