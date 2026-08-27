"""Train the demand forecasting model.

    python scripts/train_forecast.py --seed 42

Samples series, builds the horizon dataset, splits with an embargo, trains every
candidate, backtests, compares, selects on the evidence, logs to MLflow and
persists the winner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.logging import configure_logging
from app.services.container import Container
from ml.forecasting.config import get_forecast_config
from ml.forecasting.evaluate import format_bucket_table
from ml.forecasting.pipeline import train_forecast_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and select the demand forecasting model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Candidate estimators to compare. Defaults to the config.",
    )
    parser.add_argument(
        "--series",
        type=int,
        default=None,
        help="Product-store series to sample. Higher is slower; see --full.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (50 series) for correctness checking. Under a minute.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Train on every series in the dataset. Materially slower - hours, "
            "not minutes - and writes to the unsampled model directory."
        ),
    )
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip walk-forward validation."
    )
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow logging.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write model artifacts here."
    )
    args = parser.parse_args()

    configure_logging()

    config = get_forecast_config()
    if args.smoke:
        config = config.smoke()
    if args.full:
        # Every real series in the dataset. sample_series returns all listings
        # when the request exceeds what exists.
        config = config.model_copy(
            update={"sampling": config.sampling.model_copy(update={"n_series": 1_000_000})}
        )
    if args.series is not None:
        config = config.model_copy(
            update={"sampling": config.sampling.model_copy(update={"n_series": args.series})}
        )
    if args.seed is not None:
        config = config.model_copy(
            update={"sampling": config.sampling.model_copy(update={"seed": args.seed})}
        )

    repository = Container().data_repository

    result = train_forecast_pipeline(
        repository,
        config=config,
        models=tuple(args.models) if args.models else None,
        run_backtest=not args.no_backtest,
        track=not args.no_track,
        output_dir=args.output,
    )

    print()
    print("Model comparison")
    print()
    print(result.comparison.to_string(index=False))
    print()
    print(f"Selected: {result.selected.name}")
    for reason in result.rationale:
        print(f"  - {reason}")

    print()
    print("Accuracy by horizon bucket")
    print(format_bucket_table(result.selected.bucket_metrics))

    if result.fva:
        print()
        print("Forecast Value Added vs the seasonal naive (WMAPE percentage points)")
        for model, buckets in result.fva.items():
            formatted = "  ".join(f"{k}: {v:+.1%}" for k, v in buckets.items())
            print(f"  {model:24s} {formatted}")

    if not result.hierarchy.empty:
        print()
        print("Accuracy by aggregation level")
        print(result.hierarchy.to_string(index=False))

    if not result.stability.empty:
        print()
        print("Backtest stability by bucket")
        print(result.stability.to_string(index=False))

    print()
    print(f"series       : {result.n_series:,}")
    print(f"dataset rows : {result.dataset_rows:,}")
    print(f"model saved  : {result.model_path}")
    print(f"mlflow run   : {result.mlflow_run_id or '(tracking disabled or failed)'}")
    print(f"duration     : {result.duration_seconds:.1f}s")
    print(f"report       : {result.model_path / 'evaluation_report.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
