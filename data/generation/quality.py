"""Controlled data-quality corruption for the bronze layer.

Brief section 23 asks for realistic data problems. They are injected into
``bronze/`` only; ``gold/`` stays pristine.

That split is the important decision. Corrupting the single copy would force
every model in Steps 4-11 to defend against nulls and duplicates, and - worse -
would make a failed validation ambiguous: is the generator wrong, or is this the
injected corruption doing its job? Two layers keep the question answerable, and
they mirror the medallion flow the Stage 2 Databricks pipeline will use anyway.

Every corruption is counted and recorded in the manifest, so the data-quality
checks in ``data/validation`` can be scored against what was actually injected
rather than against a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.generation.config import DataQualityConfig
from data.generation.rng import RngFactory, Stream


@dataclass
class CorruptionReport:
    """What was injected, per table and issue type."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, table: str, issue: str, count: int) -> None:
        if count <= 0:
            return
        self.counts.setdefault(table, {})
        self.counts[table][issue] = self.counts[table].get(issue, 0) + count

    def total(self) -> int:
        return sum(sum(issues.values()) for issues in self.counts.values())

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            table: dict(sorted(issues.items())) for table, issues in sorted(self.counts.items())
        }


def _sample_rows(rng: np.random.Generator, n_rows: int, rate: float) -> np.ndarray:
    """Row positions to corrupt, at the configured rate."""
    if n_rows == 0 or rate <= 0:
        return np.array([], dtype=int)
    count = round(n_rows * rate)
    if count <= 0:
        return np.array([], dtype=int)
    return rng.choice(n_rows, size=min(count, n_rows), replace=False)


def corrupt_sales(
    frame: pd.DataFrame,
    config: DataQualityConfig,
    rngs: RngFactory,
    report: CorruptionReport,
    chunk_index: int = 0,
) -> pd.DataFrame:
    """Inject the sales-table issues from brief section 23."""
    if not config.enabled or frame.empty:
        return frame

    rng = rngs.fresh(Stream.DATA_QUALITY, chunk_index)
    dirty = frame.copy()
    n = len(dirty)

    # Bronze lands dates as text, exactly as a raw source feed would. A typed
    # date column physically cannot hold "0000-00-00", which is the whole reason
    # real landing zones keep dates as strings and let Silver do the parsing.
    dirty["date"] = dirty["date"].astype(str)

    # Missing foreign keys - the classic broken join.
    rows = _sample_rows(rng, n, config.missing_product_id)
    if rows.size:
        dirty.loc[dirty.index[rows], "product_id"] = None
        report.record("sales", "missing_product_id", rows.size)

    rows = _sample_rows(rng, n, config.missing_store_id)
    if rows.size:
        dirty.loc[dirty.index[rows], "store_id"] = None
        report.record("sales", "missing_store_id", rows.size)

    # Impossible prices: zero and negative both occur in real feeds, usually
    # from a failed currency conversion or a sentinel value.
    rows = _sample_rows(rng, n, config.invalid_price)
    if rows.size:
        half = rows.size // 2
        dirty.loc[dirty.index[rows[:half]], "selling_price"] = 0.0
        dirty.loc[dirty.index[rows[half:]], "selling_price"] = -1.0
        report.record("sales", "invalid_price", rows.size)

    # Negative quantities: returns booked into the sales feed without a flag.
    rows = _sample_rows(rng, n, config.negative_quantity)
    if rows.size:
        dirty.loc[dirty.index[rows], "units"] = (
            -np.abs(dirty.loc[dirty.index[rows], "units"].to_numpy()) - 1
        )
        report.record("sales", "negative_quantity", rows.size)

    # Out-of-range discounts.
    rows = _sample_rows(rng, n, config.invalid_discount)
    if rows.size:
        dirty.loc[dirty.index[rows], "discount_percentage"] = rng.choice(
            [-15.0, 135.0, 250.0], size=rows.size
        )
        report.record("sales", "invalid_discount", rows.size)

    # Promotion ids that reference nothing.
    rows = _sample_rows(rng, n, config.orphan_promotion_id)
    if rows.size:
        dirty.loc[dirty.index[rows], "promotion_id"] = "PR9999999"
        report.record("sales", "orphan_promotion_id", rows.size)

    # Malformed dates, into the already-textual bronze date column.
    rows = _sample_rows(rng, n, config.malformed_date)
    if rows.size:
        dirty.loc[dirty.index[rows], "date"] = "0000-00-00"
        report.record("sales", "malformed_date", rows.size)

    # Duplicate rows, appended so the originals survive.
    rows = _sample_rows(rng, n, config.duplicate_transactions)
    if rows.size:
        dirty = pd.concat([dirty, dirty.iloc[rows]], ignore_index=True)
        report.record("sales", "duplicate_rows", rows.size)

    return dirty


def corrupt_promotions(
    frame: pd.DataFrame,
    config: DataQualityConfig,
    rngs: RngFactory,
    report: CorruptionReport,
) -> pd.DataFrame:
    """Duplicate promotion records and inverted date ranges."""
    if not config.enabled or frame.empty:
        return frame

    rng = rngs.get(Stream.DATA_QUALITY)
    dirty = frame.copy()
    n = len(dirty)

    rows = _sample_rows(rng, n, config.duplicate_promotions)
    if rows.size:
        dirty = pd.concat([dirty, dirty.iloc[rows]], ignore_index=True)
        report.record("promotions", "duplicate_promotions", rows.size)

    # Start after end: a data-entry error that silently breaks any window join.
    rows = _sample_rows(rng, len(dirty), config.malformed_date)
    if rows.size:
        index = dirty.index[rows]
        starts = dirty.loc[index, "start_date"].to_numpy()
        dirty.loc[index, "start_date"] = dirty.loc[index, "end_date"].to_numpy()
        dirty.loc[index, "end_date"] = starts
        report.record("promotions", "inverted_date_range", rows.size)

    return dirty


def corrupt_inventory(
    frame: pd.DataFrame,
    config: DataQualityConfig,
    rngs: RngFactory,
    report: CorruptionReport,
    chunk_index: int = 0,
) -> pd.DataFrame:
    """Break the inventory identity on a small number of rows.

    ``opening + received - sold = closing`` holds exactly in gold. Breaking it
    here gives the reconciliation check something real to catch - a check that
    can never fail is not a check.
    """
    if not config.enabled or frame.empty:
        return frame

    rng = rngs.fresh(Stream.DATA_QUALITY, chunk_index + 500)
    dirty = frame.copy()

    # Reuses the invalid_price rate: both represent "a numeric field arrived
    # wrong", and a separate knob for each would be configuration for its own
    # sake. Recorded under its own issue name so the report stays unambiguous.
    rows = _sample_rows(rng, len(dirty), config.invalid_price)
    if rows.size:
        index = dirty.index[rows]
        # Cast first: pandas 3 refuses a silent int32 -> int64 widening on
        # assignment, and the drift would be invisible until it raised.
        dirty["closing_inventory"] = dirty["closing_inventory"].astype("int64")
        dirty.loc[index, "closing_inventory"] = dirty.loc[
            index, "closing_inventory"
        ].to_numpy() + rng.integers(5, 200, size=rows.size)
        report.record("inventory", "reconciliation_break", rows.size)

    return dirty
