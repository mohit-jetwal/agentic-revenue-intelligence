"""Train the baseline sales model.

    python scripts/train_baseline.py --profile dev --seed 42

Trains every candidate under both promotion approaches, backtests, scores
against Step 2's hidden ground truth, selects on the evidence, logs to MLflow
and persists the winner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.logging import configure_logging
from app.services.container import Container
from ml.baseline.pipeline import DEFAULT_MODELS, train_baseline_pipeline
from ml.baseline.training import PromotionApproach


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and select the baseline sales model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile", default="dev", help="Dataset profile the data was generated with."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Candidate estimators to compare.",
    )
    parser.add_argument(
        "--approach",
        choices=["exclude", "control", "both"],
        default="both",
        help="Promotion handling to evaluate.",
    )
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=None,
        help=(
            "Sample this many product-store listings instead of the full panel. "
            "Writes to models/baseline_sampled so it cannot overwrite the real model."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal miscoverage; 0.1 gives a 90%% interval.",
    )
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip expanding-window validation."
    )
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--folds", type=int, default=4, help="Backtest folds.")
    parser.add_argument(
        "--report", type=Path, default=None, help="Write the evaluation report here."
    )
    args = parser.parse_args()

    configure_logging()

    approaches = (
        (PromotionApproach.EXCLUDE, PromotionApproach.CONTROL)
        if args.approach == "both"
        else (PromotionApproach(args.approach),)
    )

    repository = Container().data_repository

    result = train_baseline_pipeline(
        repository,
        models=tuple(args.models),
        approaches=approaches,
        seed=args.seed,
        sample_pairs=args.sample_pairs,
        alpha=args.alpha,
        run_backtest=not args.no_backtest,
        n_folds=args.folds,
        track=not args.no_track,
    )

    print()
    print(result.comparison.summary())
    print()

    if result.latent_metrics:
        from ml.baseline.evaluation import format_comparison

        print("Against true demand (Step 2 ground truth)")
        print(format_comparison(result.latent_metrics))
        print()

    if result.error_analysis is not None:
        print("Error analysis")
        for finding in result.error_analysis.findings:
            print(f"  - {finding}")
        print()

    if result.permutation_importance is not None and not result.permutation_importance.empty:
        print("Top features (permutation importance)")
        print(result.permutation_importance.head(10).to_string(index=False))
        print()

    print(f"model saved  : {result.model_path}")
    print(f"mlflow run   : {result.mlflow_run_id or '(tracking disabled)'}")
    print(f"duration     : {result.duration_seconds:.1f}s")

    # The pipeline already wrote the report beside the model. Only write again
    # if the caller asked for it somewhere specific.
    model_dir = result.model_path or Path("data/local/models/baseline")
    default_report = model_dir / "evaluation_report.md"
    report_path = args.report or default_report
    if args.report is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(result.report(), encoding="utf-8")
    print(f"report       : {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
