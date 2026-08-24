"""Narrowing helpers for values read out of pandas.

``DataFrame.itertuples`` yields fields typed as a wide union in pandas-stubs -
anything a column could hold, including dates and bytes. At runtime these are
plain numbers, but the type checker cannot know that, and calling ``float()``
directly is rejected.

One small helper is better than a scatter of ``# type: ignore`` comments: the
ignores would suppress genuine type errors alongside the spurious ones, whereas
this makes the narrowing explicit and gives a clear failure if a column really
does contain something unexpected.
"""

from __future__ import annotations

from typing import Any


def as_float(value: Any) -> float:
    """Coerce a pandas cell to ``float``, failing loudly if it cannot be."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected a numeric value, got {value!r} ({type(value).__name__})"
        ) from exc


def as_int(value: Any) -> int:
    """Coerce a pandas cell to ``int``, failing loudly if it cannot be."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected an integer value, got {value!r} ({type(value).__name__})"
        ) from exc
