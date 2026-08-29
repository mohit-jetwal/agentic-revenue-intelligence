"""Estimate promotional uplift end to end.

Three modes, and they answer different questions.

``--synthetic``   Can the estimator recover an effect that is known exactly?
                  The only test that can establish correctness, because real
                  data has no counterfactual to check against.

``--validate-ground-truth``
                  Does it work on the platform dataset, whose generator
                  recorded the true promotion response curves? Validates the
                  average effect only - two per-event terms are not persisted.

default           Run the analysis and persist it for the service to serve.

Usage
-----

.. code-block:: powershell

    uv run python scripts/estimate_uplift.py --synthetic --all-scenarios
    uv run python scripts/estimate_uplift.py --validate-ground-truth
    uv run python scripts/estimate_uplift.py --sample-pairs 400 --seed 42
    uv run python scripts/estimate_uplift.py --full --no-track
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Import before the project packages so a direct `python scripts/...` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.observability.logging import configure_logging, get_logger
from app.services.container import Container
from ml.forecasting.sampling import sample_series
from ml.promo_uplift.config import get_promo_uplift_config
from ml.promo_uplift.controls import build_control_pool
from ml.promo_uplift.data import baseline_predictions, build_uplift_panel
from ml.promo_uplift.diagnostics import (
    expected_effect_from_ground_truth,
)
from ml.promo_uplift.estimators import AIPWEstimator, fit_nuisances
from ml.promo_uplift.features import build_covariates
from ml.promo_uplift.model import FittedUpliftModel, default_output_dir
from ml.promo_uplift.pipeline import report, run_uplift
from ml.promo_uplift.synthetic import SCENARIOS, generate, scenario_config
from ml.promo_uplift.tracking import track_run
from ml.promo_uplift.treatment import build_analysis_frame

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate promotional uplift.")
    parser.add_argument("--synthetic", action="store_true", help="Run on known-truth data.")
    parser.add_argument("--all-scenarios", action="store_true", help="Every synthetic scenario.")
    parser.add_argument("--scenario", default="confounded", help="One synthetic scenario.")
    parser.add_argument(
        "--validate-ground-truth",
        action="store_true",
        help="Compare against the generator's recorded response curves.",
    )
    parser.add_argument("--sample-pairs", type=int, default=400, help="Product-store pairs.")
    parser.add_argument("--full", action="store_true", help="Every listing. Slow.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Tiny run, for correctness.")
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow.")
    parser.add_argument("--no-placebo", action="store_true", help="Skip the placebo test.")
    parser.add_argument(
        "--no-sensitivity", action="store_true", help="Skip the sensitivity sweep."
    )
    parser.add_argument("--output", type=Path, default=None, help="Artifact directory.")
    return parser.parse_args()


def run_synthetic(args: argparse.Namespace) -> int:
    """Recover known effects. Returns a process exit code."""
    base = get_promo_uplift_config()
    if args.smoke:
        base = base.smoke()

    names = list(SCENARIOS) if args.all_scenarios else [args.scenario]

    print(
        f"{'scenario':<18}{'true':>9}{'naive':>9}{'AIPW':>9}{'error':>8}"
        f"{'SE':>7}  {'covers':>6}  verdict"
    )
    print("-" * 80)

    failures = 0
    for name in names:
        config = scenario_config(name, base)
        panel = generate(name, config=config)
        analysis = build_analysis_frame(panel.observable(), config=config)
        pool = build_control_pool(analysis, config=config)
        covariates = build_covariates(
            pool.frame, analysis.events, config=config, history=analysis.frame
        )
        nuisance = fit_nuisances(covariates, config=config)
        estimate = AIPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()

        y, t = covariates.y, covariates.t
        naive = y[t].mean() / y[~t].mean() - 1.0
        error = abs(estimate.ate_pct - panel.true_att_pct)

        # The tolerance comes from the estimator's own standard error, not from
        # a round number. A flat 3 points means two different things at +65% and
        # at 0%, and picking a threshold that happens to pass is not a test. The
        # floor of 2 points exists because a standard error can be small while
        # a finite-sample nuisance bias is not.
        se_pct = (
            estimate.standard_error / estimate.baseline_units
            if estimate.standard_error and estimate.baseline_units > 0
            else 0.0
        )
        allowed = max(2.5 * se_pct, 0.02)
        ok = error <= allowed
        failures += not ok

        # Whether the interval covers the truth is reported separately. It is a
        # stricter question than recovery - a narrow interval that misses by a
        # point says the uncertainty is understated, which matters on its own.
        band = estimate.interval_pct()
        covers = (
            "yes" if band and band[0] <= panel.true_att_pct <= band[1] else "no"
        )

        print(
            f"{name:<18}{panel.true_att_pct:>+9.1%}{naive:>+9.1%}"
            f"{estimate.ate_pct:>+9.1%}{error:>8.1%}{se_pct:>7.1%}"
            f"  {covers:>6}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print(f"{len(names) - failures}/{len(names)} scenarios recovered.")
    print("Tolerance is 2.5 standard errors, floored at 2 percentage points.")
    return 0 if failures == 0 else 1


def run_ground_truth(args: argparse.Namespace) -> int:
    """Validate against the platform generator's recorded parameters."""
    settings = get_settings()
    config = get_promo_uplift_config()
    if args.smoke:
        config = config.smoke()

    repository = Container().data_repository
    pairs = None if args.full else sample_series(
        repository, n_series=args.sample_pairs, seed=args.seed
    )
    panel = build_uplift_panel(repository, pairs, config=config)

    analysis = build_analysis_frame(panel, config=config)
    pool = build_control_pool(analysis, config=config)
    covariates = build_covariates(
        pool.frame, analysis.events, config=config, history=analysis.frame
    )
    nuisance = fit_nuisances(covariates, config=config)
    estimate = AIPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()

    ground_truth_dir = settings.project_root / "data" / "local" / "ground_truth"
    expected = expected_effect_from_ground_truth(analysis.events, ground_truth_dir)
    if expected is None:
        print(f"No ground truth found at {ground_truth_dir}. Generate the dataset first.")
        return 1

    error = abs(estimate.ate_pct - expected.expected_pct)
    print(f"events analysed : {expected.n_events:,}")
    print(f"expected (true) : {expected.expected_pct:+.1%}")
    print(f"  mechanic      : {expected.mechanic_pct:+.1%}")
    print(f"  price channel : {expected.price_channel_pct:+.1%}")
    print(f"estimated (AIPW): {estimate.ate_pct:+.1%}")
    print(f"absolute error  : {error:.1%}")
    print()
    for caveat in expected.caveats:
        print(f"  ! {caveat}")
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()

    if args.synthetic:
        return run_synthetic(args)
    if args.validate_ground_truth:
        return run_ground_truth(args)

    settings = get_settings()
    config = get_promo_uplift_config()
    if args.smoke:
        config = config.smoke()

    started = time.perf_counter()
    container = Container()
    repository = container.data_repository

    sampled = not args.full
    pairs = (
        sample_series(repository, n_series=args.sample_pairs, seed=args.seed)
        if sampled
        else None
    )
    panel = build_uplift_panel(repository, pairs, config=config)

    model_root = settings.project_root / "data" / "local" / "models"
    baseline = baseline_predictions(panel, repository, model_root / "baseline")

    run = run_uplift(
        panel,
        config=config,
        baseline_units=baseline,
        ground_truth_dir=settings.project_root / "data" / "local" / "ground_truth",
        run_placebo=not args.no_placebo,
        run_sensitivity=not args.no_sensitivity,
    )

    output = args.output or default_output_dir(model_root, sampled=sampled)
    output.mkdir(parents=True, exist_ok=True)

    # The report is written BEFORE tracking. Step 6 lost a three-hour run to an
    # MLflow store that rejected the write at the end, and the fix is ordering,
    # not error handling.
    (output / "uplift_report.md").write_text(report(run), encoding="utf-8")

    model = FittedUpliftModel.from_run(run, repository)
    model.save(output)

    run_id = None
    if not args.no_track:
        run_id = track_run(run, settings=settings, run_name=f"uplift_seed{args.seed}")

    headline = run.headline
    print()
    print(f"selected   : {run.selected}")
    if headline:
        interval = headline.interval_pct()
        band = f"  [{interval[0]:+.1%}, {interval[1]:+.1%}]" if interval else ""
        print(f"uplift     : {headline.ate_pct:+.1%}{band}")
    impact = run.headline_impact
    if impact:
        print(f"impact     : {impact.summary()}")
    print(f"validation : {run.validation_status}")
    print(f"artifact   : {output}")
    if run_id:
        print(f"mlflow run : {run_id}")
    print(f"elapsed    : {time.perf_counter() - started:.1f}s")

    # A failed validation is a non-zero exit. A scheduled job should not quietly
    # publish an estimate whose causal assumptions did not hold.
    return 0 if run.validation_status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
