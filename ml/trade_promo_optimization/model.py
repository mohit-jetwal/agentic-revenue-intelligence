"""The trade promotion optimiser, wired to Step 7's event table."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from ml.base import InsufficientDataError, ModelMetadata
from ml.trade_promo_optimization.interface import (
    AllocationConstraint,
    AllocationLine,
    OptimizationResult,
    TradePromoOptimizationModel,
)
from ml.trade_promo_optimization.optimizer import allocate, candidates_from_events

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"


class FittedTradePromoOptimizer(TradePromoOptimizationModel):
    """Allocates a promotional budget across analysed promotions."""

    name = "trade_promo_optimization"

    def __init__(
        self,
        repository: DataRepository,
        *,
        event_impact: pd.DataFrame | None = None,
        model_dir: Path | None = None,
        model_version: str = MODEL_VERSION,
    ) -> None:
        super().__init__(repository)
        self.version = model_version
        self._events = event_impact
        self._model_dir = model_dir
        self._metadata = ModelMetadata(
            name=self.name, version=model_version, trained_at=datetime.now(UTC), approved=False
        )

    @property
    def events(self) -> pd.DataFrame:
        """Step 7's per-event impact table, loaded on first use.

        This optimiser has no model of its own - it allocates against measured
        incremental profit. Without that table there is nothing to allocate
        against, which is why the failure names the command that produces it.
        """
        if self._events is None:
            path = self._resolve_events_path()
            if path is None:
                raise InsufficientDataError(
                    "no promo uplift event table found. This optimiser allocates "
                    "against measured incremental profit per promotion; run "
                    "scripts/estimate_uplift.py to produce it"
                )
            self._events = pd.read_parquet(path)
            logger.info("trade_promo.events_loaded", path=str(path), rows=len(self._events))
        return self._events

    def _resolve_events_path(self) -> Path | None:
        if self._model_dir is not None:
            candidate = self._model_dir / "event_impact.parquet"
            return candidate if candidate.is_file() else None

        from app.config.settings import get_settings

        root = get_settings().project_root / "data" / "local" / "models"
        for directory in ("promo_uplift", "promo_uplift_sampled"):
            candidate = root / directory / "event_impact.parquet"
            if candidate.is_file():
                return candidate
        return None

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """No-op: the optimiser solves fresh on every call.

        There is no artifact because there is nothing learned - the inputs are
        Step 7's estimates and the output is a solve over them.
        """
        return self.metadata

    def predict(  # type: ignore[override]
        self,
        *,
        total_budget: float,
        product_ids: list[str] | None = None,
        regions: list[str] | None = None,
        retailers: list[str] | None = None,
        constraints: list[AllocationConstraint] | None = None,
        objective: str = "incremental_profit",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> OptimizationResult:
        """Allocate the budget under the given constraints."""
        events = self.events
        if product_ids:
            events = events[events["product_id"].isin(product_ids)]
        if regions and "region" in events.columns:
            events = events[events["region"].isin(regions)]
        if retailers and "retailer" in events.columns:
            events = events[events["retailer"].isin(retailers)]

        if events.empty:
            raise InsufficientDataError(
                "no analysed promotions match these filters, so there is nothing "
                "to allocate a budget across"
            )

        candidates = candidates_from_events(events)
        limits = {
            (c.dimension, c.value): (c.min_spend, c.max_spend)
            for c in (constraints or [])
        }

        outcome = allocate(candidates, total_budget, dimension_limits=limits)
        return self._to_result(outcome, objective=objective)

    def _to_result(self, outcome: Any, *, objective: str) -> OptimizationResult:
        lines = [
            AllocationLine(
                product_id=str(row["product_id"]),
                region=str(row["region"]) if pd.notna(row.get("region")) else None,
                retailer=str(row["retailer"]) if pd.notna(row.get("retailer")) else None,
                allocated_spend=float(row["allocated_spend"]),
                expected_incremental_units=float(row["expected_incremental_units"]),
                expected_incremental_revenue=float(row["expected_incremental_revenue"]),
                expected_incremental_profit=float(row["expected_incremental_profit"]),
                expected_roi=float(row["expected_roi"]),
                marginal_roi=float(row["marginal_roi"]),
            )
            for row in (
                outcome.lines.to_dict("records") if not outcome.lines.empty else []
            )
        ]

        return OptimizationResult(
            total_budget=outcome.total_budget,
            allocated_budget=outcome.allocated,
            objective=objective,
            allocations=lines,
            expected_incremental_units=outcome.incremental_units,
            expected_incremental_revenue=outcome.incremental_revenue,
            expected_incremental_profit=outcome.incremental_profit,
            overall_roi=outcome.roi,
            optimization_status=outcome.status,
            binding_constraints=outcome.binding_constraints,
            solver=outcome.solver,
            solve_time_ms=outcome.solve_time_ms,
            assumptions=[
                "Promotional response saturates: each cell's return is modelled "
                "as a concave piecewise-linear curve, so the budget spreads "
                "rather than pouring into the single highest-ROI cell.",
                "Incremental profit per promotion comes from Step 7's causal "
                "estimates, and inherits their assumptions - including that "
                "cannibalisation is not deducted, so profit is an upper bound.",
                "No cell is funded beyond twice its observed spend. Extrapolating "
                "a saturating curve past the range any promotion has visited "
                "would produce a confident recommendation nobody should act on.",
                "ROI on the current dataset is not interpretable: generated "
                "promotional spend runs about 20x the achievable margin at "
                "product-store grain. The allocation's RANKING is meaningful; "
                "its absolute profit is not.",
            ],
            warnings=outcome.warnings,
        )


__all__ = ["MODEL_VERSION", "FittedTradePromoOptimizer"]
