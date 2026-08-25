"""Contract validation machinery.

Why Pandera here, when Step 2 deliberately rejected it: the two are solving
different problems.

``data/validation/checks.py`` verifies *cross-row business invariants* over a
whole generated dataset - the inventory identity, revenue consistency, whether
prices vary enough for elasticity to be identified. Those are statements about a
dataset as a body of evidence, and they are awkward to express in a schema
library.

These contracts verify the *shape of a frame crossing an interface* - columns
present, dtypes right, values in range, keys not null. That is exactly what
Pandera exists for, and hand-rolling it a second time would be reinventing a
wheel for no gain. Both layers stay: the Step 2 checks run against the generated
dataset, these run at the repository boundary and at the feature-builder inputs.

Validation is **opt-in per call** and off by default. Schema-checking a 6M-row
frame on every read is a real cost and the generator already guarantees the gold
layer is clean. It is switched on in tests and in the dataset builders, which is
where a contract breach would actually cause damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from app.observability.logging import get_logger
from data.repositories.base import DataAccessError

logger = get_logger(__name__)


class ContractViolationError(DataAccessError):
    """Raised when a frame fails its data contract.

    Subclasses :class:`~data.repositories.base.DataAccessError` so a caller that
    already handles data-access failures does not need a second except clause to
    stay correct - a contract breach *is* a failure to supply usable data.
    """

    def __init__(self, contract: str, failures: pd.DataFrame | None, message: str) -> None:
        super().__init__(f"contract {contract!r} violated: {message}")
        self.contract = contract
        #: Pandera's failure cases frame - column, check, failing value, index.
        self.failures = failures

    def summary(self, limit: int = 10) -> str:
        """Human-readable digest of the first few failures."""
        if self.failures is None or self.failures.empty:
            return str(self)
        lines = [str(self), "", "failing checks:"]
        for row in self.failures.head(limit).itertuples(index=False):
            column = getattr(row, "column", "?")
            check = getattr(row, "check", "?")
            value = getattr(row, "failure_case", "?")
            lines.append(f"  {column}: {check} (e.g. {value!r})")
        remaining = len(self.failures) - limit
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")
        return "\n".join(lines)


@dataclass(frozen=True)
class DataContract:
    """A named schema for one table, plus the metadata a caller needs."""

    name: str
    schema: pa.DataFrameSchema
    description: str
    #: Columns that together identify a row. Empty when no uniqueness applies.
    primary_key: tuple[str, ...] = ()
    #: Columns that must resolve against another table, for documentation and
    #: for the referential-integrity checks in ``data/validation``.
    foreign_keys: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.schema.columns)

    def validate(self, frame: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
        """Validate ``frame``, returning it (coerced) on success.

        ``lazy=True`` collects every failure rather than stopping at the first.
        That matters when a generator change breaks four columns: seeing all
        four at once is one fix, whereas seeing them one per run is four.
        """
        return validate(frame, self, lazy=lazy)


def validate(frame: pd.DataFrame, contract: DataContract, *, lazy: bool = True) -> pd.DataFrame:
    """Validate a frame against a contract.

    Returns the validated (and dtype-coerced) frame so this can be used inline:
    ``return validate(frame, SALES_CONTRACT)``.
    """
    try:
        validated: pd.DataFrame = contract.schema.validate(frame, lazy=lazy)
    except SchemaErrors as exc:
        failures = exc.failure_cases
        logger.warning(
            "contract.violated",
            contract=contract.name,
            failure_count=len(failures),
            columns=sorted({str(c) for c in failures.get("column", [])}),
        )
        raise ContractViolationError(
            contract.name,
            failures,
            f"{len(failures)} failing check(s) across "
            f"{failures['column'].nunique() if 'column' in failures else 0} column(s)",
        ) from exc
    except SchemaError as exc:
        logger.warning("contract.violated", contract=contract.name, error=str(exc))
        raise ContractViolationError(contract.name, exc.failure_cases, str(exc)) from exc

    return validated


def check_primary_key(frame: pd.DataFrame, contract: DataContract) -> None:
    """Assert the contract's primary key is unique and complete.

    Kept separate from the schema because Pandera's multi-column uniqueness is
    awkward to express alongside per-column checks, and because a caller reading
    a filtered slice legitimately may not want the cost.
    """
    key = list(contract.primary_key)
    if not key or frame.empty:
        return

    missing = [column for column in key if column not in frame.columns]
    if missing:
        raise ContractViolationError(
            contract.name, None, f"primary key columns absent from frame: {missing}"
        )

    duplicates = int(frame.duplicated(subset=key).sum())
    if duplicates:
        raise ContractViolationError(
            contract.name,
            frame.loc[frame.duplicated(subset=key, keep=False), key].head(20),
            f"{duplicates} duplicate rows on primary key {key}",
        )

    nulls = int(frame[key].isna().any(axis=1).sum())
    if nulls:
        raise ContractViolationError(
            contract.name, None, f"{nulls} rows with a null primary key component {key}"
        )


# ---------------------------------------------------------------------------
# Shared column builders
# ---------------------------------------------------------------------------
# Defined once so "what is a product_id" has a single answer. A column redefined
# per table drifts, and the drift shows up as a join that silently returns
# nothing.


def id_column(description: str, *, nullable: bool = False) -> pa.Column:
    return pa.Column(
        str,
        nullable=nullable,
        description=description,
        checks=pa.Check.str_length(min_value=1),
    )


def date_column(description: str) -> pa.Column:
    # `coerce` because Parquet round-trips dates as objects or datetimes
    # depending on the writer, and every consumer wants one type.
    return pa.Column("datetime64[ns]", nullable=False, coerce=True, description=description)


def money_column(description: str, *, nullable: bool = False, positive: bool = False) -> pa.Column:
    checks = [pa.Check.gt(0)] if positive else [pa.Check.ge(0)]
    return pa.Column(float, nullable=nullable, coerce=True, description=description, checks=checks)


def count_column(description: str, *, nullable: bool = False) -> pa.Column:
    return pa.Column(
        int, nullable=nullable, coerce=True, description=description, checks=[pa.Check.ge(0)]
    )


def percentage_column(description: str, *, nullable: bool = False) -> pa.Column:
    return pa.Column(
        float,
        nullable=nullable,
        coerce=True,
        description=description,
        checks=[pa.Check.in_range(0.0, 100.0)],
    )


def flag_column(description: str, *, nullable: bool = False) -> pa.Column:
    return pa.Column(bool, nullable=nullable, coerce=True, description=description)


def category_column(
    description: str, *, allowed: list[str] | None = None, nullable: bool = False
) -> pa.Column:
    checks: list[Any] = []
    if allowed:
        checks.append(pa.Check.isin(allowed))
    return pa.Column(str, nullable=nullable, description=description, checks=checks)
