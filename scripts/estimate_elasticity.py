"""Estimate elasticity, and validate it against known truth.

Two modes:

``--validate-ground-truth``
    Score every estimator against ``ground_truth/elasticity.json``. This is the
    deliverable: it establishes *which* estimator to trust on data where truth
    is known, so the choice is measured rather than assumed.

default
    Estimate for one product and print the comparison.

Usage
-----

.. code-block:: powershell

    uv run python scripts/estimate_elasticity.py --validate-ground-truth
    uv run python scripts/estimate_elasticity.py --validate-ground-truth --products 40
    uv run python scripts/estimate_elasticity.py --product P00003 --cross
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.observability.logging import configure_logging, get_logger
from app.services.container import Container
from ml.price_elasticity.data import build_elasticity_panel, load_cost_index
from ml.price_elasticity.estimator import METHOD_PREFERENCE, estimate_all, prepare_panel

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate and validate elasticity.")
    parser.add_argument("--validate-ground-truth", action="store_true")
    parser.add_argument("--products", type=int, default=25, help="Products to score.")
    parser.add_argument("--product", default=None, help="Estimate one product.")
    parser.add_argument("--cross", action="store_true", help="Include cross-price.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> int:
    """Score every estimator against the recorded elasticities."""
    settings = get_settings()
    repository = Container().data_repository

    path = settings.project_root / "data" / "local" / "ground_truth" / "elasticity.json"
    if not path.is_file():
        print(f"No ground truth at {path}. Generate the dataset first.")
        return 1
    truth: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))["values"]

    products = repository.get_products()
    sample = products.sample(
        min(args.products, len(products)), random_state=args.seed
    )["product_id"].tolist()

    costs = load_cost_index(repository)
    rows: list[dict[str, float]] = []

    for product_id in sample:
        panel = build_elasticity_panel(repository, product_ids=[product_id])
        if panel.empty:
            continue
        frame = prepare_panel(panel, costs=costs if not costs.empty else None)
        if len(frame) < 200:
            continue
        estimates = estimate_all(frame)
        if not estimates:
            continue
        rows.append(
            {
                "true": truth[product_id],
                **{name: est.elasticity for name, est in estimates.items()},
            }
        )

    if not rows:
        print("No product had enough price variation to estimate.")
        return 1

    frame = pd.DataFrame(rows)
    print(f"products scored: {len(frame)}\n")
    print(f"{'method':<14}{'MAE':>8}{'signed':>9}{'est/true':>10}{'corr':>7}  verdict")
    print("-" * 62)

    for method in METHOD_PREFERENCE:
        if method not in frame:
            continue
        # Estimates beyond |20| are not near-misses, they are failures of
        # identification. Kept out of the summary statistics so one blown
        # estimate does not hide the pattern, and counted separately.
        usable = frame[method].notna() & frame[method].abs().lt(20)
        subset = frame[usable]
        if len(subset) < 3:
            continue

        error = subset[method] - subset["true"]
        ratio = float((subset[method] / subset["true"]).median())
        correlation = float(np.corrcoef(subset["true"], subset[method])[0, 1])
        blown = int((~usable).sum())

        verdict = "GOOD" if abs(ratio - 1) < 0.15 and correlation > 0.9 else "biased"
        note = f"  ({blown} unusable)" if blown else ""
        print(
            f"{method:<14}{error.abs().mean():>8.3f}{error.mean():>+9.3f}"
            f"{ratio:>10.3f}{correlation:>7.2f}  {verdict}{note}"
        )

    print()
    print("est/true is the median ratio: 1.0 means unbiased, below 1.0 means")
    print("attenuated toward zero - the product looks less price-sensitive than")
    print("it is, which encourages exactly the wrong pricing recommendation.")
    return 0


def estimate_one(args: argparse.Namespace) -> int:
    from app.schemas.elasticity import ElasticityErrorResponse, ElasticityRequest

    response = Container().elasticity_service.estimate(
        ElasticityRequest(
            product_id=args.product,
            include_comparison=True,
            include_cross_price=args.cross,
        )
    )
    if isinstance(response, ElasticityErrorResponse):
        print(f"[{response.error_code}] {response.message}")
        return 1

    print(f"product     : {response.product_id}")
    print(f"method      : {response.method}")
    print(f"elasticity  : {response.elasticity:+.3f}")
    print(f"elastic     : {response.is_elastic} - a price rise {response.revenue_direction}")
    print(f"sample      : {response.sample_size:,}")
    if response.comparison:
        print()
        for row in response.comparison:
            mark = "" if row.selectable else "  (not selectable)"
            print(f"  {row.method:<14}{row.elasticity:>9.3f}{mark}")
    if response.cross_price:
        print()
        print(f"  substitutes: {response.substitutes or 'none'}")
        print(f"  complements: {response.complements or 'none'}")
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    if args.validate_ground_truth:
        return validate(args)
    if args.product:
        return estimate_one(args)
    print("Nothing to do. Pass --validate-ground-truth or --product.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
