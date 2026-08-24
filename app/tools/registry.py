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


def build_default_registry() -> ToolRegistry:
    """Construct the registry with the platform's analytical tools.

    Empty in Step 1. Tools are registered here as they are implemented in
    Stage 1 Step 13, once the models from Steps 4-11 exist to back them.
    """
    return ToolRegistry()
