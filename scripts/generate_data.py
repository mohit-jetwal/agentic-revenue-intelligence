"""Dataset generation entrypoint matching the invocation in brief section 21.

    python scripts/generate_data.py \
        --products 500 --stores 1000 --customers 50000 --seed 42

A thin wrapper over ``data.generation.pipeline``. The canonical interface is the
CLI (``uv run ari generate-data``); this exists because the brief specifies this
invocation, and because a plain script is the more discoverable entrypoint for
someone reading the repository for the first time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/generate_data.py` from a clean checkout without an
# editable install having been done first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.logging import configure_logging
from data.generation.config import available_profiles
from data.generation.pipeline import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic CPG/Retail dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="dev",
        choices=available_profiles(),
        help="Dataset profile. Individual --products/--stores flags override it.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--products", type=int, default=None, help="Number of products.")
    parser.add_argument("--stores", type=int, default=None, help="Number of stores.")
    parser.add_argument("--customers", type=int, default=None, help="Number of customers.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root. Defaults to the configured DATA__PARQUET_ROOT parent.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after generating and exit non-zero if it fails.",
    )
    args = parser.parse_args()

    configure_logging()

    result = generate_dataset(
        args.profile,
        seed=args.seed,
        output_root=args.output,
        overrides={
            "scale.products": args.products,
            "scale.stores": args.stores,
            "scale.customers": args.customers,
        },
    )

    print()
    print(result.summary())
    print()
    print(f"gold content hash: {result.gold_hash[:16]}")

    if not args.validate:
        print(f"\nNext: uv run ari validate-data --profile {args.profile}")
        return 0

    from data.validation.report import validate_dataset, write_report

    report = validate_dataset(result.root)
    markdown_path, _ = write_report(report, result.root)

    summary = report.checks.summary()
    passed = sum(1 for r in report.relationships if r.passed)
    print(
        f"\ninvariants   : {summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed"
    )
    print(f"relationships: {passed}/{len(report.relationships)} passed")
    print(f"report       : {markdown_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
