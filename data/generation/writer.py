"""Parquet output and the dataset manifest.

Large facts are written one partition per date chunk, so the full panel never
has to exist in memory at once. That is what makes the stress profile feasible
and what would make a PySpark port in Stage 2 a mechanical translation.

The manifest is the other half of the job. ``dataset_version()`` on the
repository reads it, which means every ``ToolResult`` a future agent produces can
be traced to the exact seed, config hash and row counts that generated the data
behind it. Without that, a recommendation is attributable to "some dataset".
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.observability.logging import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"


@dataclass
class DatasetWriter:
    """Writes tables to a layered Parquet dataset and tracks what it wrote."""

    root: Path
    #: Literal rather than str so the value is checked here, where the mistake
    #: is cheap, instead of surfacing as a pyarrow error mid-generation.
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "zstd"
    row_counts: dict[str, int] = field(default_factory=dict)
    _partition_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # -- layers -------------------------------------------------------------

    def layer(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reset(self) -> None:
        """Remove previous output.

        Regeneration must not leave stale partitions behind: an old partition
        from a larger profile would silently join into the new dataset and make
        row counts, and any statistic computed from them, quietly wrong.
        """
        for child in ("gold", "bronze", "ground_truth"):
            target = self.root / child
            if target.exists():
                shutil.rmtree(target)
        manifest = self.root / MANIFEST_FILENAME
        if manifest.exists():
            manifest.unlink()

    # -- writing ------------------------------------------------------------

    def write_table(self, frame: pd.DataFrame, name: str, *, layer: str = "gold") -> Path:
        """Write a complete table as a single Parquet file."""
        path = self.layer(layer) / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression=self.compression)
        if layer == "gold":
            self.row_counts[name] = len(frame)
        logger.debug("dataset.table_written", table=name, layer=layer, rows=len(frame))
        return path

    def write_partition(
        self,
        frame: pd.DataFrame,
        name: str,
        partition: str,
        *,
        layer: str = "gold",
    ) -> Path:
        """Append one partition to a partitioned table.

        Hive-style ``part=<value>`` directories, which DuckDB and Spark both
        discover automatically - so the local reader and a future Databricks
        reader see the same layout without translation.
        """
        directory = self.layer(layer) / name / f"part={partition}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "data.parquet"
        frame.to_parquet(path, index=False, compression=self.compression)

        if layer == "gold":
            self.row_counts[name] = self.row_counts.get(name, 0) + len(frame)
            self._partition_counts[name] = self._partition_counts.get(name, 0) + 1
        return path

    def write_samples(self, tables: dict[str, pd.DataFrame], destination: Path, rows: int) -> None:
        """Small CSV extracts, committed to the repo for browsing.

        Deliberately CSV and deliberately tiny: these exist so someone reading
        the repository can see the shape of every table without generating
        gigabytes first. The real dataset stays Parquet and git-ignored.
        """
        destination.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            if frame.empty:
                continue
            frame.head(rows).to_csv(destination / f"{name}.csv", index=False)

    # -- manifest -----------------------------------------------------------

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        payload = {
            **payload,
            "row_counts": dict(sorted(self.row_counts.items())),
            "partition_counts": dict(sorted(self._partition_counts.items())),
            "total_rows": int(sum(self.row_counts.values())),
            "written_at": datetime.now(UTC).isoformat(),
        }
        path = self.root / MANIFEST_FILENAME
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path

    # -- verification -------------------------------------------------------

    def content_hash(self, layer: str = "gold") -> str:
        """Stable hash over every Parquet file in a layer.

        Backs the reproducibility test: the same seed and config must produce
        byte-identical output. Hashing file contents rather than row counts
        catches the failure mode where the shape is right but the values drifted.
        """
        digest = hashlib.sha256()
        base = self.root / layer
        if not base.exists():
            return ""
        for path in sorted(base.rglob("*.parquet")):
            digest.update(str(path.relative_to(base)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


def read_manifest(root: Path) -> dict[str, Any]:
    """Load the dataset manifest, or raise a message that says what to run."""
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No dataset manifest at {path}. Generate one first:\n"
            f"    uv run ari generate-data --profile dev"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
