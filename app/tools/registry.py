"""Tool registry.

The Supervisor plans over tool *names*. This registry is what turns a planned
name back into an executable object, and it is the single place where the set of
capabilities available to an agent is defined.

Keeping it explicit (rather than auto-discovering every ``AnalyticalTool``
subclass on the import path) is deliberate: a tool becoming callable by an agent
should be a decision someone made, not a side effect of a file existing.
"""

from __future__ import annotations

from typing import Any

from app.observability.logging import get_logger
from app.schemas.tool_contract import ToolSpec
from app.tools.base import AnalyticalTool

logger = get_logger(__name__)

#: A tool of unknown concrete input/output types. The registry is deliberately
#: heterogeneous - it holds forecasters next to optimisers - so its element type
#: has to erase the parameters. Type safety is recovered at the call site, where
#: ``run()`` validates against the tool's own input schema.
type AnyTool = AnalyticalTool[Any, Any]


class ToolNotFoundError(KeyError):
    """Raised when a plan references a tool that is not registered."""


class ToolRegistry:
    """Name -> tool instance, with the specs Claude needs for tool calling."""

    def __init__(self) -> None:
        self._tools: dict[str, AnyTool] = {}

    def register(self, tool: AnyTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool
        logger.debug("tool.registered", tool=tool.name, permission=tool.permission)

    def get(self, name: str) -> AnyTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"unknown tool {name!r}; registered: {sorted(self._tools)}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, *, permissions: set[str] | None = None) -> list[ToolSpec]:
        """Tool specs for Claude, optionally filtered by permission.

        The filter is how least privilege reaches the LLM: a tool the caller may
        not use is not merely blocked at execution, it is never advertised, so
        the model does not plan around a capability it cannot have.
        """
        return [
            tool.spec()
            for _, tool in sorted(self._tools.items())
            if permissions is None or tool.permission in permissions
        ]

    def __len__(self) -> int:
        return len(self._tools)


def build_default_registry(
    *,
    forecasting_service: object | None = None,
    promo_uplift_service: object | None = None,
    elasticity_service: object | None = None,
    optimization_service: object | None = None,
) -> ToolRegistry:
    """Construct the registry with the platform's analytical tools.

    Empty in Step 1; tools arrive as the models behind them are built. Step 5
    registers the first one - the brief for that step asks for a working
    ``ForecastingTool`` contract, not a placeholder, so it is wired here rather
    than deferred to Step 13 along with the rest.

    Services are injected rather than constructed. A tool that builds its own
    service would load a model at registry-construction time, and the registry is
    built during container startup - where a missing model artifact should not be
    fatal.
    """
    registry = ToolRegistry()

    if forecasting_service is not None:
        from app.tools.forecasting_tool import ForecastingTool

        registry.register(ForecastingTool(forecasting_service))  # type: ignore[arg-type]

    if promo_uplift_service is not None:
        from app.tools.promo_uplift_tool import PromoUpliftTool

        registry.register(PromoUpliftTool(promo_uplift_service))  # type: ignore[arg-type]

    if elasticity_service is not None:
        from app.tools.elasticity_tool import ElasticityTool

        registry.register(ElasticityTool(elasticity_service))  # type: ignore[arg-type]

    if optimization_service is not None:
        from app.tools.optimization_tools import (
            AllocateBudgetTool,
            OptimizePriceTool,
            SimulateScenarioTool,
        )

        registry.register(AllocateBudgetTool(optimization_service))
        registry.register(OptimizePriceTool(optimization_service))
        registry.register(SimulateScenarioTool(optimization_service))

    return registry
