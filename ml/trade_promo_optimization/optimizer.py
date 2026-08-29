"""Trade promotion budget allocation.

Allocates a fixed budget across candidate promotions to maximise incremental
profit, subject to business constraints.

**Diminishing returns are the whole problem.** The generator builds promotional
response as ``a·(1 - e^(-b·d))`` - the tenth point of discount buys far less than
the second. A linear program over a constant ROI-per-rupee ignores that and
pours the entire budget into the single highest-ROI cell, which is both wrong and
obviously wrong to any category manager.

So the response is approximated as a **piecewise-linear concave** function and
solved as a mixed-integer program with OR-Tools. Concavity is what makes the
greedy intuition correct: each successive segment of spend on a cell returns
less than the last, so the optimum spreads.

**Constraint infeasibility is a result, not an error.** "Your minimum regional
spends already exceed the budget" is a genuine business finding the agent should
surface. The solver reports which constraints bind rather than raising.

**A caveat this module cannot fix.** Step 7 measured promotional spend in the
generated data at roughly 20x the achievable margin at product-store grain, so
absolute ROI is not interpretable. The allocation is therefore validated on
*ranking* and *constraint satisfaction* - does it prefer the genuinely better
promotions, and does it respect its bounds - not on the profit number it reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from app.observability.logging import get_logger

logger = get_logger(__name__)

#: Segments in the piecewise-linear approximation of each cell's response curve.
#: Five is enough to make the concavity bite without turning a 2,000-cell problem
#: into 10,000 variables the solver labours over.
DEFAULT_SEGMENTS = 5

#: Saturation curvature. Spend beyond `saturation_point x this` returns very
#: little, matching the generator's `a(1 - exp(-b*d))` shape.
DEFAULT_CURVATURE = 3.0


@dataclass
class Candidate:
    """One promotion the optimiser may fund."""

    candidate_id: str
    product_id: str
    region: str | None = None
    retailer: str | None = None
    category: str | None = None

    #: Incremental profit if funded at ``reference_spend``. From Step 7's
    #: per-event table, which is exactly what it was built to supply.
    reference_profit: float = 0.0
    reference_spend: float = 0.0
    reference_units: float = 0.0
    reference_revenue: float = 0.0

    #: Hard bounds on what this cell may receive.
    min_spend: float = 0.0
    max_spend: float | None = None

    @property
    def profit_rate(self) -> float:
        """Incremental profit per rupee at the reference point."""
        return self.reference_profit / self.reference_spend if self.reference_spend > 0 else 0.0


@dataclass
class AllocationOutcome:
    """What the solver decided, and what limited it."""

    lines: pd.DataFrame
    total_budget: float
    allocated: float
    incremental_units: float
    incremental_revenue: float
    incremental_profit: float
    status: str
    binding_constraints: list[str] = field(default_factory=list)
    solver: str = "CBC"
    solve_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def roi(self) -> float | None:
        return self.incremental_profit / self.allocated if self.allocated > 0 else None

    def summary(self) -> str:
        roi = f"{self.roi:.2f}" if self.roi is not None else "n/a"
        return (
            f"{self.status}: allocated {self.allocated:,.0f} of "
            f"{self.total_budget:,.0f} across {len(self.lines):,} cells, "
            f"incremental profit {self.incremental_profit:,.0f} (ROI {roi})"
        )


def candidates_from_events(
    events: pd.DataFrame,
    *,
    max_candidates: int = 500,
    min_spend_floor: float = 0.0,
) -> list[Candidate]:
    """Build the candidate set from Step 7's per-event impact table.

    Ranked by profit rate rather than absolute profit: with a fixed budget the
    question is what each rupee buys, not what each promotion produced. A large
    promotion with a poor rate should lose to two small efficient ones.

    Value-destroying events are **kept**. The optimiser's job is partly to
    allocate away from them, and it cannot do that if they are filtered out
    before it sees them.
    """
    if events.empty:
        return []

    required = {"promotion_id", "product_id", "incremental_profit", "promotion_spend"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"event table is missing {sorted(missing)}")

    working = events[events["promotion_spend"].fillna(0) > 0].copy()
    if working.empty:
        return []

    working["_rate"] = working["incremental_profit"] / working["promotion_spend"]
    working = working.nlargest(max_candidates, "_rate")

    return [
        Candidate(
            candidate_id=str(row["promotion_id"]),
            product_id=str(row["product_id"]),
            region=str(row["region"]) if pd.notna(row.get("region")) else None,
            retailer=str(row["retailer"]) if pd.notna(row.get("retailer")) else None,
            category=str(row["category"]) if pd.notna(row.get("category")) else None,
            reference_profit=float(row["incremental_profit"]),
            reference_spend=float(row["promotion_spend"]),
            reference_units=float(row.get("incremental_units", 0.0) or 0.0),
            reference_revenue=float(row.get("incremental_revenue", 0.0) or 0.0),
            min_spend=min_spend_floor,
        )
        for row in working.to_dict("records")
    ]


def _segments(
    candidate: Candidate,
    *,
    n_segments: int,
    curvature: float,
    max_cell_spend: float,
) -> list[tuple[float, float]]:
    """Piecewise-linear breakdown of one cell's concave response.

    Returns ``(width, profit_per_rupee)`` per segment, with the rate strictly
    decreasing. The rates follow the derivative of ``1 - e^(-k·x)``, evaluated at
    each segment's midpoint and normalised so that spending the full reference
    amount reproduces the reference profit.

    Decreasing rates are what make the linear program's optimum correct without
    integer variables: because the cheapest-first ordering is optimal for a
    concave objective, the solver naturally fills high-return segments across
    many cells before topping any one up.
    """
    ceiling = candidate.max_spend or max_cell_spend
    ceiling = max(ceiling, candidate.min_spend)
    if ceiling <= 0:
        return []

    edges = np.linspace(0.0, ceiling, n_segments + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    scale = max(candidate.reference_spend, 1e-9)

    # Marginal return of a saturating curve at each midpoint.
    marginal = np.exp(-curvature * midpoints / scale)
    # Normalise so total profit at reference_spend equals reference_profit.
    at_reference = (1.0 - np.exp(-curvature)) / curvature
    rate = candidate.profit_rate / max(at_reference, 1e-9)

    return [
        (float(edges[i + 1] - edges[i]), float(rate * marginal[i]))
        for i in range(n_segments)
    ]


def allocate(
    candidates: list[Candidate],
    total_budget: float,
    *,
    dimension_limits: dict[tuple[str, str], tuple[float | None, float | None]] | None = None,
    n_segments: int = DEFAULT_SEGMENTS,
    curvature: float = DEFAULT_CURVATURE,
    max_cell_multiple: float = 2.0,
) -> AllocationOutcome:
    """Allocate ``total_budget`` across candidates to maximise incremental profit.

    ``dimension_limits`` maps ``(dimension, value)`` to ``(min_spend, max_spend)``
    for constraints like "at least 20% in the North" or "no more than 3M with
    this retailer".

    ``max_cell_multiple`` caps any one cell at that multiple of its reference
    spend. Without it the optimiser would push spend far beyond the range where
    the response curve was ever observed - extrapolating a saturating curve into
    territory no promotion has visited is how an optimiser produces a confident
    recommendation nobody should act on.
    """
    started = time.perf_counter()

    if not candidates:
        return AllocationOutcome(
            lines=pd.DataFrame(),
            total_budget=total_budget,
            allocated=0.0,
            incremental_units=0.0,
            incremental_revenue=0.0,
            incremental_profit=0.0,
            status="infeasible",
            warnings=["no fundable candidates: every event had zero recorded spend"],
        )

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools GLOP solver unavailable")

    warnings: list[str] = []
    segment_vars: dict[str, list] = {}
    objective = solver.Objective()

    for candidate in candidates:
        ceiling = candidate.max_spend or candidate.reference_spend * max_cell_multiple
        pieces = _segments(
            candidate,
            n_segments=n_segments,
            curvature=curvature,
            max_cell_spend=ceiling,
        )
        variables = []
        for index, (width, rate) in enumerate(pieces):
            variable = solver.NumVar(0.0, width, f"{candidate.candidate_id}_s{index}")
            objective.SetCoefficient(variable, rate)
            variables.append(variable)
        segment_vars[candidate.candidate_id] = variables

        if candidate.min_spend > 0 and variables:
            floor = solver.Constraint(candidate.min_spend, solver.infinity())
            for variable in variables:
                floor.SetCoefficient(variable, 1.0)

    objective.SetMaximization()

    budget = solver.Constraint(0.0, total_budget, "budget")
    for variables in segment_vars.values():
        for variable in variables:
            budget.SetCoefficient(variable, 1.0)

    warnings.extend(
        _apply_dimension_limits(solver, candidates, segment_vars, dimension_limits or {})
    )
    applied_limits = {
        key: bounds
        for key, bounds in (dimension_limits or {}).items()
        if any(str(getattr(c, key[0], None)) == key[1] for c in candidates)
    }

    status_code = solver.Solve()
    status = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
    }.get(status_code, "unknown")

    if status in {"infeasible", "unbounded", "unknown"}:
        warnings.append(
            f"solver returned {status}. The usual cause is conflicting "
            f"constraints - minimum spends that already exceed the budget - "
            f"which is a business finding rather than a failure"
        )
        return AllocationOutcome(
            lines=pd.DataFrame(),
            total_budget=total_budget,
            allocated=0.0,
            incremental_units=0.0,
            incremental_revenue=0.0,
            incremental_profit=0.0,
            status=status,
            binding_constraints=_infeasible_hints(candidates, total_budget, dimension_limits),
            solve_time_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    lines = _build_lines(candidates, segment_vars, n_segments=n_segments, curvature=curvature)
    allocated = float(lines["allocated_spend"].sum()) if not lines.empty else 0.0

    binding = []
    if allocated >= total_budget * 0.999:
        binding.append("budget")
    # Only constraints that were actually applied can bind. Reporting an
    # unmatched one as "at its minimum" because its spend is zero would be a
    # false claim that the constraint was honoured.
    binding.extend(_binding_dimension_limits(lines, applied_limits))

    if allocated < total_budget * 0.95:
        warnings.append(
            f"only {allocated:,.0f} of {total_budget:,.0f} allocated. The "
            f"remaining budget had no positive-return home: every candidate's "
            f"marginal return had fallen below zero, or per-cell caps bound first"
        )

    outcome = AllocationOutcome(
        lines=lines,
        total_budget=total_budget,
        allocated=allocated,
        incremental_units=float(lines["expected_incremental_units"].sum()),
        incremental_revenue=float(lines["expected_incremental_revenue"].sum()),
        incremental_profit=float(lines["expected_incremental_profit"].sum()),
        status=status,
        binding_constraints=binding,
        solver="GLOP",
        solve_time_ms=int((time.perf_counter() - started) * 1000),
        warnings=warnings,
    )
    logger.info(
        "trade_promo.allocated",
        status=status,
        cells=len(lines),
        allocated=round(allocated, 2),
        profit=round(outcome.incremental_profit, 2),
    )
    return outcome


def _apply_dimension_limits(
    solver: pywraplp.Solver,
    candidates: list[Candidate],
    segment_vars: dict[str, list],
    limits: dict[tuple[str, str], tuple[float | None, float | None]],
) -> list[str]:
    """Add min/max spend constraints on region, retailer, category or product.

    Returns warnings for constraints that matched nothing. **Those must not be
    silently dropped.** A caller who asks for "at least 900,000 in the North"
    and receives an allocation with none in the North has been misled twice: the
    constraint was ignored, and the result looked like it was honoured. It
    happens for a mundane reason - the event table carries no ``region`` column,
    so every candidate's region is ``None`` and nothing matches.
    """
    unmatched: list[str] = []

    for (dimension, value), (minimum, maximum) in limits.items():
        members = [c for c in candidates if str(getattr(c, dimension, None)) == value]
        if not members:
            unmatched.append(
                f"constraint on {dimension}='{value}' matched no candidate and "
                f"was NOT applied. Either no promotion carries that value, or "
                f"the input table has no '{dimension}' column"
            )
            continue
        constraint = solver.Constraint(
            minimum if minimum is not None else 0.0,
            maximum if maximum is not None else solver.infinity(),
            f"{dimension}:{value}",
        )
        for candidate in members:
            for variable in segment_vars[candidate.candidate_id]:
                constraint.SetCoefficient(variable, 1.0)

    return unmatched


def _build_lines(
    candidates: list[Candidate],
    segment_vars: dict[str, list],
    *,
    n_segments: int,
    curvature: float,
) -> pd.DataFrame:
    """Turn solved segment variables into one row per funded cell."""
    rows = []
    for candidate in candidates:
        spend = sum(v.solution_value() for v in segment_vars[candidate.candidate_id])
        if spend <= 1e-6:
            continue

        # Realised profit on the concave curve at this spend, not spend x the
        # reference rate - the latter would credit the cell with linear returns
        # the optimiser explicitly declined to assume.
        scale = max(candidate.reference_spend, 1e-9)
        at_reference = (1.0 - np.exp(-curvature)) / curvature
        realised = (
            candidate.profit_rate
            / max(at_reference, 1e-9)
            * scale
            / curvature
            * (1.0 - np.exp(-curvature * spend / scale))
        )
        share = spend / scale

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "product_id": candidate.product_id,
                "region": candidate.region,
                "retailer": candidate.retailer,
                "category": candidate.category,
                "allocated_spend": spend,
                "reference_spend": candidate.reference_spend,
                "spend_vs_reference": share,
                "expected_incremental_profit": realised,
                "expected_incremental_units": candidate.reference_units * min(share, 1.0),
                "expected_incremental_revenue": candidate.reference_revenue * min(share, 1.0),
                "expected_roi": realised / spend if spend > 0 else 0.0,
                # Return on the next rupee, which is where a budget increase
                # should go.
                "marginal_roi": float(
                    candidate.profit_rate
                    / max(at_reference, 1e-9)
                    * np.exp(-curvature * spend / scale)
                ),
            }
        )

    frame = pd.DataFrame(rows)
    _ = n_segments
    return (
        frame.sort_values("expected_incremental_profit", ascending=False).reset_index(drop=True)
        if not frame.empty
        else frame
    )


def _binding_dimension_limits(
    lines: pd.DataFrame,
    limits: dict[tuple[str, str], tuple[float | None, float | None]],
) -> list[str]:
    """Which dimension constraints are active at the optimum."""
    binding: list[str] = []
    if lines.empty:
        return binding

    for (dimension, value), (minimum, maximum) in limits.items():
        if dimension not in lines.columns:
            continue
        spend = float(lines.loc[lines[dimension] == value, "allocated_spend"].sum())
        if maximum is not None and spend >= maximum * 0.999:
            binding.append(f"{dimension}:{value} at its maximum ({maximum:,.0f})")
        if minimum is not None and spend <= minimum * 1.001:
            binding.append(f"{dimension}:{value} at its minimum ({minimum:,.0f})")
    return binding


def _infeasible_hints(
    candidates: list[Candidate],
    total_budget: float,
    limits: dict[tuple[str, str], tuple[float | None, float | None]] | None,
) -> list[str]:
    """Name the conflict, so the caller learns something from the refusal."""
    hints: list[str] = []

    floors = sum(c.min_spend for c in candidates)
    if floors > total_budget:
        hints.append(
            f"per-candidate minimum spends total {floors:,.0f}, which already "
            f"exceeds the budget of {total_budget:,.0f}"
        )

    if limits:
        dimension_floors = sum(
            minimum for (minimum, _) in limits.values() if minimum is not None
        )
        if dimension_floors > total_budget:
            hints.append(
                f"dimension minimum spends total {dimension_floors:,.0f}, which "
                f"already exceeds the budget of {total_budget:,.0f}"
            )
    return hints or ["constraints conflict; no allocation satisfies all of them"]


__all__ = [
    "DEFAULT_CURVATURE",
    "DEFAULT_SEGMENTS",
    "AllocationOutcome",
    "Candidate",
    "allocate",
    "candidates_from_events",
]
