"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config.settings import Settings, get_settings
from app.services.container import Container, get_container


def container_dependency(request: Request) -> Container:
    """Resolve the container.

    Prefers the instance attached to application state at startup so that a test
    can override it per-app; falls back to the process-wide singleton for
    callers outside a request cycle.
    """
    container = getattr(request.app.state, "container", None)
    if isinstance(container, Container):
        return container
    return get_container()


def settings_dependency() -> Settings:
    return get_settings()


ContainerDep = Annotated[Container, Depends(container_dependency)]
SettingsDep = Annotated[Settings, Depends(settings_dependency)]
