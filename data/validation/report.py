"""Validation orchestration and reporting.

Loads a generated dataset, runs the business-invariant checks and the
statistical relationship tests, and writes a human-readable report plus a
machine-readable JSON summary.

The report is the artifact that answers "how do you know the data is any good?"
with evidence rather than assertion - which is the same question Steps 4-11 will
have to answer about their models, using the same ground truth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from app.observability.logging import get_logger
from data.generation.ground_truth import GroundTruth
from data.generation.writer import read_manifest
from data.validation.checks import CheckSuite, run_all_checks
from data.validation.statistical import (
    RelationshipResult,
    validate_competitor_effect,
    validate_cross_price,
    validate_own_price_elasticity,
    validate_price_demand_direction,
    validate_promotion_uplift,
    validate_seasonality_and_regions,
    validate_stockout_censoring,
)

logger = get_logger(__name__)

#: Facts sampled rather than fully loaded. Validation needs a representative
#: sample, not every row - and at stress scale the full panel will not fit.
_SAMPLED_TABLES = {"sales_daily", "inventory", "pricing", "sales_transactions"}


@dataclass
class ValidationReport:
    """Combined outcome of invariant and relationship validation."""

    checks: CheckSuite
    relationships: list[RelationshipResult]
    manifest: dict[str, Any]
    row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.checks.passed and all(r.passed for r in self.relationships)

    @property
    def failed_relationships(self) -> list[RelationshipResult]:
        return [r for r in self.relationships if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_version": self.manifest.get("dataset_version"),
            "seed": self.manifest.get("seed"),
            "config_hash": self.manifest.get("config_hash"),
            "checks": {
                "summary": self.checks.summary(),
                "results": [
                    {
                        "name": r.name,
                        "table": r.table,
                        "status": r.status,
                        "severity": r.severity.value,
                        "message": r.message,
                        "observed": r.observed,
                        "threshold": r.threshold,
                        "failing_rows": r.failing_rows,
                    }
                    for r in self.checks.results
                ],
            },
            "relationships": [
                {
                    "name": r.name,
                    "status": r.status,
                    "description": r.description,
                    "observed": r.observed,
                    "expected": r.expected,
                    "tolerance": r.tolerance,
                    "sample_size": r.sample_size,
                    "detail": r.detail,
                }
                for r in self.relationships
            ],
        }

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Dataset Validation Report",
            "",
            f"- **Dataset**: `{self.manifest.get('dataset_version')}` "
            f"(profile `{self.manifest.get('profile')}`, seed `{self.manifest.get('seed')}`)",
            f"- **Config hash**: `{self.manifest.get('config_hash')}`",
            f"- **Generated**: {self.manifest.get('written_at')}",
            f"- **Overall**: {'PASS' if self.passed else 'FAIL'}",
            "",
            "## Row counts",
            "",
            "| Table | Rows |",
            "| --- | ---: |",
        ]
        for name, count in sorted(self.manifest.get("row_counts", {}).items()):
            lines.append(f"| `{name}` | {count:,} |")

        summary = self.checks.summary()
        lines += [
            "",
            "## Business invariants",
            "",
            f"{summary['passed']}/{summary['total']} passed, "
            f"{summary['failed']} failed, {summary['warnings']} warnings.",
            "",
            "| Check | Table | Status | Observed | Threshold |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for result in self.checks.results:
            observed = "" if result.observed is None else f"{result.observed:,.4g}"
            threshold = "" if result.threshold is None else f"{result.threshold:,.4g}"
            lines.append(
                f"| `{result.name}` | `{result.table}` | {result.status} "
                f"| {observed} | {threshold} |"
            )

        lines += [
            "",
            "## Relationship recovery",
            "",
            "Does the generated data actually contain the relationships it was "
            "built to contain? These are the checks that make Steps 4-11 "
            "falsifiable rather than merely plausible.",
            "",
            "| Relationship | Status | Observed | Expected | Tolerance | n |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for relationship in self.relationships:
            observed = "" if relationship.observed is None else f"{relationship.observed:,.4g}"
            expected = "" if relationship.expected is None else f"{relationship.expected:,.4g}"
            tolerance = "" if relationship.tolerance is None else f"{relationship.tolerance:,.4g}"
            lines.append(
                f"| `{relationship.name}` | {relationship.status} | {observed} | {expected} "
                f"| {tolerance} | {relationship.sample_size:,} |"
            )

        lines += ["", "### Detail", ""]
        for relationship in self.relationships:
            lines.append(f"**`{relationship.name}`** — {relationship.description}")
            if relationship.detail:
                lines += [
                    "",
                    "```json",
                    json.dumps(relationship.detail, indent=2, default=str),
                    "```",
                ]
            lines.append("")

        injected = self.manifest.get("data_quality_injected", {})
        if injected:
            lines += [
                "## Injected data-quality issues (bronze layer)",
                "",
                "Gold is clean by construction. These were injected into bronze so "
                "the quality framework has real defects to catch.",
                "",
                "| Table | Issue | Rows |",
                "| --- | --- | ---: |",
            ]
            for table, issues in sorted(injected.items()):
                for issue, count in sorted(issues.items()):
                    lines.append(f"| `{table}` | `{issue}` | {count:,} |")

        return "\n".join(lines) + "\n"


def _select_partitions(parts: list[Path], sample_rows: int) -> list[Path]:
    """Choose partitions spread evenly across the whole history.

    Taking the *first* N partitions would be much simpler and quietly wrong.
    Regular prices only change a few times a year, so a three-month window
    contains almost no price variation and elasticity stops being identified -
    validation would then fail on a perfectly good dataset because the sampler
    starved it, which is exactly the kind of failure that sends you debugging
    the generator for hours.

    Row counts come from Parquet footers, so choosing a stride costs no I/O.
    Whole partitions are kept intact rather than sampling rows within them, so
    each retains its contiguous run of days.
    """
    if not parts:
        return []

    total = sum(pq.ParquetFile(part).metadata.num_rows for part in parts)
    if total <= sample_rows:
        return parts

    stride = max(1, math.ceil(total / sample_rows))
    selected = parts[::stride]
    # Always include the last partition so the sample spans the full range,
    # which matters for trend and lifecycle checks.
    if parts[-1] not in selected:
        selected = [*selected, parts[-1]]
    return selected


def load_gold_tables(root: Path, *, sample_rows: int | None = 1_500_000) -> dict[str, pd.DataFrame]:
    """Read the gold layer into memory, sampling the large facts.

    Sampling by whole partitions rather than randomly across rows keeps each
    partition's day sequence intact, and - via :func:`_select_partitions` -
    preserves the full time span, which the statistical checks depend on.
    """
    gold = root / "gold"
    if not gold.is_dir():
        raise FileNotFoundError(
            f"No gold layer at {gold}. Generate one first:\n"
            f"    uv run ari generate-data --profile dev"
        )

    tables: dict[str, pd.DataFrame] = {}
    for entry in sorted(gold.iterdir()):
        if entry.is_file() and entry.suffix == ".parquet":
            tables[entry.stem] = pd.read_parquet(entry)
        elif entry.is_dir():
            parts = sorted(entry.rglob("*.parquet"))
            if not parts:
                continue
            if entry.name in _SAMPLED_TABLES and sample_rows:
                parts = _select_partitions(parts, sample_rows)
            tables[entry.name] = pd.concat(
                (pd.read_parquet(p) for p in parts), ignore_index=True
            )
    return tables


def load_latent_demand(root: Path, *, sample_rows: int = 1_500_000) -> pd.DataFrame:
    """Read latent demand ground truth, for the censoring checks.

    Spread across the full history for the same reason as the gold facts: an
    injected stockout window sits at an arbitrary point in the timeline, and
    reading only the earliest partitions would often miss it entirely.
    """
    directory = root / "ground_truth" / "latent_demand"
    if not directory.is_dir():
        return pd.DataFrame()
    parts = _select_partitions(sorted(directory.rglob("*.parquet")), sample_rows)
    if not parts:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def validate_dataset(root: Path, *, sample_rows: int | None = 400_000) -> ValidationReport:
    """Run every check and relationship test against a generated dataset."""
    manifest = read_manifest(root)
    tables = load_gold_tables(root, sample_rows=sample_rows)
    ground_truth = GroundTruth.load(root)

    logger.info("validation.loaded", tables=len(tables), dataset=manifest.get("dataset_version"))

    suite = run_all_checks(tables)

    sales = tables.get("sales_daily", pd.DataFrame())
    relationships: list[RelationshipResult] = []

    if not sales.empty:
        relationships += validate_price_demand_direction(sales)
        relationships += validate_own_price_elasticity(sales, ground_truth)
        relationships += validate_promotion_uplift(sales)
        relationships += validate_stockout_censoring(sales, load_latent_demand(root))

        pricing = tables.get("pricing", pd.DataFrame())
        if not pricing.empty:
            relationships += validate_cross_price(sales, pricing, ground_truth)

        relationships += validate_competitor_effect(
            sales, tables.get("competitor_pricing", pd.DataFrame())
        )

        calendar = tables.get("calendar", pd.DataFrame())
        stores = tables.get("stores", pd.DataFrame())
        if not calendar.empty and not stores.empty:
            relationships += validate_seasonality_and_regions(sales, calendar, stores)

    return ValidationReport(
        checks=suite,
        relationships=relationships,
        manifest=manifest,
        row_counts=manifest.get("row_counts", {}),
    )


def write_report(report: ValidationReport, root: Path) -> tuple[Path, Path]:
    """Write the markdown report and JSON summary next to the dataset."""
    markdown_path = root / "validation_report.md"
    json_path = root / "validation_results.json"
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return markdown_path, json_path
