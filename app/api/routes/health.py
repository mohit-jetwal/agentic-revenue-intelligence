"""Health and metrics endpoints. Live from Step 1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ContainerDep, SettingsDep
from app.observability.metrics import METRICS
from app.schemas.api import (
    DependencyCheck,
    DependencyStatus,
    HealthResponse,
    MetricsResponse,
)

router = APIRouter(tags=["operations"])

#: Dependencies that are legitimately absent in early Stage 1 steps. Their
#: failure degrades the service rather than breaking it, so the endpoint reports
#: "degraded" instead of "unavailable" - an honest signal, not a green light.
_OPTIONAL_UNTIL_LATER = {"data_repository", "model_registry", "vector_store", "llm_provider"}


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(container: ContainerDep, settings: SettingsDep) -> HealthResponse:
    """Report service status and per-dependency detail.

    Always returns 200: the body carries the verdict. A health check that 503s
    on a dependency that is *expected* to be missing at this stage would make
    the endpoint useless for the thing it is for - telling you what is wrong.
    """
    checks: list[DependencyCheck] = []
    degraded = False

    for name, ok, detail in container.health_checks():
        if ok:
            dep_status = DependencyStatus.OK
        elif name in _OPTIONAL_UNTIL_LATER:
            dep_status = DependencyStatus.NOT_CONFIGURED
            degraded = True
        else:
            dep_status = DependencyStatus.UNAVAILABLE
            degraded = True
        checks.append(DependencyCheck(name=name, status=dep_status, detail=detail))

    return HealthResponse(
        status=DependencyStatus.DEGRADED if degraded else DependencyStatus.OK,
        name=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment.value,
        dependencies=checks,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Process counters")
def metrics() -> MetricsResponse:
    """In-process counters.

    Process-local and reset on restart - adequate for a single-process dev
    server. Production monitoring is Databricks-native; see the README.
    """
    return MetricsResponse(**METRICS.snapshot())  # type: ignore[arg-type]
