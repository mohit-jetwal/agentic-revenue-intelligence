"""End-to-end baseline training run.

Orchestrates the sequence the brief lays out in section 43: build the panel,
split it temporally, train every candidate under both promotion approaches,
backtest, score against ground truth, compare, select, track, persist.

Kept separate from the individual modules so each stays independently testable -
and so the *order* of operations, which is where most of the correctness lives,
is visible in one place rather than implied across six files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.config.settings import get_settings
from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from data.repositories.sampling import sample_product_store_pairs
from features.engineering import FeatureEngineer, FeatureRequest
from ml.baseline.comparison import Candidate, ComparisonResult, score_against_latent, select_model
from ml.baseline.evaluation import (
    BaselineMetrics,
    ErrorAnalysis,
    analyse_errors,
    format_comparison,
    irreducible_error,
)
from ml.baseline.model import FittedBaselineModel
from ml.baseline.models import build_estimator, permutation_importance
from ml.baseline.training import (
    PromotionApproach,
    TemporalSplit,
    build_temporal_split,
    expanding_window_backtest,
    prepare_training_rows,
)

logger = get_logger(__name__)

#: Candidate estimators. Ordered simplest first, which is also the order the
#: comparison table reads best in.
DEFAULT_MODELS: tuple[str, ...] = ("seasonal_naive", "ridge", "lightgbm")


@dataclass
class PipelineResult:
    """Everything a training run produced."""

    comparison: ComparisonResult
    split: TemporalSplit
    panel_rows: int
    error_analysis: ErrorAnalysis | None = None
    feature_importance: pd.DataFrame | None = None
    permutation_importance: pd.DataFrame | None = None
    latent_metrics: dict[str, Any] = field(default_factory=dict)
    #: The noise floor - how well a model knowing the true conditional mean
    #: could possibly score. Every WMAPE above should be read against it.
    noise_floor: BaselineMetrics | None = None
    duration_seconds: float = 0.0
    mlflow_run_id: str | None = None
    model_path: Path | None = None

    @property
    def selected(self) -> Candidate:
        return self.comparison.selected

    def report(self) -> str:
        """Human-readable evaluation report, logged to MLflow as an artifact."""
        lines = [
            "# Baseline Sales - Evaluation Report",
            "",
            f"Panel rows: {self.panel_rows:,}",
            f"Split: {self.split.describe()}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "## Model comparison",
            "",
            self.comparison.summary(),
        ]

        if self.noise_floor is not None:
            achieved = self.selected.latent_wmape
            lines += [
                "",
                "## Noise floor",
                "",
                "A model that knew the *true* conditional mean exactly would still",
                f"score WMAPE {self.noise_floor.wmape:.1%} against realised sales, because",
                "demand is drawn from an over-dispersed negative binomial and that",
                "variance is not learnable by anything.",
                "",
            ]
            if pd.notna(achieved) and self.noise_floor.wmape > 0:
                ratio = achieved / self.noise_floor.wmape
                lines.append(
                    f"The selected model scores {achieved:.1%}, which is "
                    f"**{ratio:.2f}x the noise floor**."
                )
                if ratio < 1.0:
                    lines.append(
                        "Scoring *below* the floor is not possible on honest features - "
                        "treat it as evidence of target leakage, not of a good model."
                    )
                elif ratio < 1.25:
                    lines.append(
                        "Most of the remaining error is irreducible noise rather than "
                        "unexploited signal, so further model tuning has little headroom."
                    )
                else:
                    lines.append(
                        "There is real headroom between the model and the floor, so "
                        "additional signal remains unexploited."
                    )
            lines.append(
                "\nThis benchmark exists because a bare WMAPE is uninterpretable: "
                "without it, a near-optimal model reads as inaccurate and a leaking "
                "one reads as excellent."
            )

        if self.latent_metrics:
            lines += [
                "",
                "## Against true demand (Step 2 ground truth)",
                "",
                "The validation a real project cannot run. `latent_units` is demand",
                "before inventory censored it, so this separates a model that learned",
                "demand from one that learned what the till recorded.",
                "",
                format_comparison(self.latent_metrics),
            ]

        if self.error_analysis is not None:
            lines += ["", "## Error analysis", ""]
            lines.extend(f"- {finding}" for finding in self.error_analysis.findings)

            if not self.error_analysis.worst_products.empty:
                lines += [
                    "",
                    "### Worst products by WMAPE",
                    "",
                    self.error_analysis.worst_products.head(10).to_string(index=False),
                ]

        if self.permutation_importance is not None and not self.permutation_importance.empty:
            lines += [
                "",
                "## Feature importance (permutation)",
                "",
                "Measured as the WMAPE degradation when a feature is shuffled.",
                "Model-agnostic, and unlike split gain it is not biased toward",
                "high-cardinality features.",
                "",
                self.permutation_importance.head(15).to_string(index=False),
            ]

        return "\n".join(lines)


def build_panel(
    repository: DataRepository,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sample_pairs: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the modelling panel through the Step 3 feature layer.

    Uses a point-in-time view anchored at ``end_date``, so every feature is
    computed as it would have been known then. Nothing here reaches into the
    repository directly - the whole point of Step 3 was that a model does not
    need to.
    """
    calendar = repository.get_calendar()
    available_start = pd.to_datetime(calendar["date"]).dt.date.min()
    available_end = pd.to_datetime(calendar["date"]).dt.date.max()

    start = start_date or available_start
    end = end_date or available_end

    product_ids: list[str] | None = None
    store_ids: list[str] | None = None
    if sample_pairs:
        # Sample real listings rather than crossing independent product and
        # store samples - most of that cross product was never stocked.
        pairs = sample_product_store_pairs(
            repository, n_pairs=sample_pairs, start_date=start, end_date=end, seed=seed
        )
        product_ids = sorted(pairs["product_id"].unique().tolist())
        store_ids = sorted(pairs["store_id"].unique().tolist())
        logger.info(
            "baseline.panel_sampled",
            pairs=len(pairs), products=len(product_ids), stores=len(store_ids),
        )

    view = repository.as_of(end)
    engineer = FeatureEngineer(view)

    started = time.perf_counter()
    panel = engineer.build(
        FeatureRequest(
            start_date=start,
            end_date=end,
            product_ids=product_ids,
            store_ids=store_ids,
            # Promotion features are built once and used by Approach CONTROL;
            # Approach EXCLUDE drops them at feature-selection time. Building
            # the panel twice would double the cost for no benefit.
            promotion=True,
            include_promotion_spend=False,
        )
    )
    logger.info(
        "baseline.panel_built",
        rows=len(panel), columns=len(panel.columns),
        seconds=round(time.perf_counter() - started, 1),
    )
    return panel


def load_latent_demand(repository: DataRepository) -> pd.DataFrame:
    """Load Step 2's true-demand ground truth, if present.

    Read directly from disk, never through the repository - the whole point of
    ``ground_truth/`` is that no repository method can reach it, so a model
    cannot train on it. Loading it here, in the *evaluation* path, is the one
    legitimate use.
    """
    settings = get_settings()
    directory = (
        settings.resolve(settings.data.parquet_root).parent
        / "ground_truth"
        / "latent_demand"
    )
    if not directory.is_dir():
        logger.warning("baseline.no_ground_truth", path=str(directory))
        return pd.DataFrame()

    parts = sorted(directory.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()

    frame = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    logger.info("baseline.ground_truth_loaded", rows=len(frame))
    return frame


def train_baseline_pipeline(
    repository: DataRepository,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    approaches: tuple[PromotionApproach, ...] = (
        PromotionApproach.EXCLUDE,
        PromotionApproach.CONTROL,
    ),
    seed: int = 42,
    sample_pairs: int | None = None,
    alpha: float = 0.1,
    run_backtest: bool = True,
    n_folds: int = 4,
    track: bool = True,
    output_dir: Path | None = None,
) -> PipelineResult:
    """Run the full training and selection pipeline."""
    from ml.baseline.training import train_baseline

    started = time.perf_counter()

    panel = build_panel(repository, sample_pairs=sample_pairs, seed=seed)
    if panel.empty:
        raise ValueError("the feature panel is empty; generate a dataset first")

    split = build_temporal_split(panel)
    latent = load_latent_demand(repository)

    candidates: list[Candidate] = []
    for model_name in models:
        for approach in approaches:
            # A fresh estimator per combination. Reusing one would carry the
            # previous fit into the next and the comparison would be nonsense.
            estimator = build_estimator(model_name, seed=seed)
            trained = train_baseline(
                panel, estimator, approach=approach, split=split, alpha=alpha
            )
            candidate = Candidate(trained=trained)

            if run_backtest:
                candidate.backtest = expanding_window_backtest(
                    panel,
                    lambda name=model_name: build_estimator(name, seed=seed),
                    approach=approach,
                    n_folds=n_folds,
                )

            if not latent.empty:
                candidate.latent_metrics = score_against_latent(candidate, panel, latent)

            candidates.append(candidate)

    comparison = select_model(candidates)
    selected = comparison.selected

    # The noise floor, measured on the same test window the models are scored
    # on so the two numbers are directly comparable.
    noise_floor: BaselineMetrics | None = None
    if not latent.empty:
        latent_dates = pd.to_datetime(latent["date"]).dt.date
        noise_floor = irreducible_error(
            latent[(latent_dates >= split.test_start) & (latent_dates <= split.test_end)]
        )

    # --- diagnostics on the winner ----------------------------------------
    dates = pd.to_datetime(panel["date"]).dt.date
    test_rows = panel[(dates >= split.test_start) & (dates <= split.test_end)]

    predictions = test_rows[["date", "product_id", "store_id"]].copy()
    predictions["actual_units"] = test_rows["units"].to_numpy(dtype=float)
    predictions["baseline_units"] = selected.trained.predict_baseline(test_rows)
    for column in ("promotion_flag", "stockout_flag", "category", "brand", "region",
                   "channel", "store_type", "season", "is_new_product"):
        if column in test_rows.columns:
            predictions[column] = test_rows[column].to_numpy()

    error_analysis = analyse_errors(predictions)

    importance = selected.trained.estimator.feature_importance()

    permutation: pd.DataFrame | None = None
    clean_test, _ = prepare_training_rows(test_rows, approach=PromotionApproach.EXCLUDE)
    if not clean_test.empty and selected.trained.estimator.name != "seasonal_naive":
        features = list(selected.trained.feature_names)
        permutation = permutation_importance(
            selected.trained.estimator,
            clean_test[features],
            clean_test["units"],
            seed=seed,
        )

    result = PipelineResult(
        comparison=comparison,
        split=split,
        panel_rows=len(panel),
        error_analysis=error_analysis,
        noise_floor=noise_floor,
        feature_importance=importance,
        permutation_importance=permutation,
        latent_metrics=selected.latent_metrics,
        duration_seconds=time.perf_counter() - started,
    )

    # --- persist and track -------------------------------------------------
    settings = get_settings()
    directory = output_dir or (settings.project_root / "data" / "local" / "models" / "baseline")

    model = FittedBaselineModel(
        repository,
        estimator=selected.trained.estimator,
        approach=selected.trained.approach,
        feature_names=selected.trained.feature_names,
        calibration=selected.trained.calibration,
    )
    model.save(directory)
    result.model_path = directory

    # The evaluation report is written here, not by the caller, so that the
    # comparison table survives even if everything after this point fails. It is
    # a deliverable of the step in its own right - the justification for the
    # selection - and reconstructing it means retraining.
    report_path = directory / "evaluation_report.md"
    report_path.write_text(result.report(), encoding="utf-8")

    if track:
        from ml.baseline.tracking import configure_mlflow, log_comparison, register_selected

        # Tracking is bookkeeping, and bookkeeping must never destroy the thing
        # it is keeping books on. An experiment-tracking backend that rejects its
        # own URI, a locked database, a full disk - none of those are reasons to
        # discard a training run that already succeeded and is already on disk.
        #
        # Learned the hard way: a three-hour run was lost to an MLflow store
        # rejection raised *after* every model had been trained and selected.
        try:
            experiment_id = configure_mlflow(settings)
            dataset_version = repository.dataset_version()

            log_comparison(
                comparison,
                dataset_version=dataset_version,
                experiment_id=experiment_id,
                seed=seed,
                panel_rows=len(panel),
            )
            run_id, _ = register_selected(
                selected.trained,
                dataset_version=dataset_version,
                experiment_id=experiment_id,
                seed=seed,
                evaluation_report=result.report(),
            )
            result.mlflow_run_id = run_id
        except Exception as exc:  # noqa: BLE001 - tracking must not fail the run
            logger.warning(
                "baseline.tracking_failed",
                error=f"{type(exc).__name__}: {exc}",
                note="the model and evaluation report were still written",
                model_path=str(directory),
            )

    logger.info(
        "baseline.pipeline_completed",
        selected=selected.name,
        duration_seconds=round(result.duration_seconds, 1),
        panel_rows=len(panel),
    )
    return result
