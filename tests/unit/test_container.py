"""Dependency container - the dev/prod seam.

The property under test is that switching ``APP__ENVIRONMENT`` switches
implementations cleanly, and that an unconfigured Databricks environment fails
with an actionable message rather than an ImportError or a silent fallback to
local data. A silent fallback would be the dangerous outcome: an agent happily
answering from stale local Parquet while believing it queried the warehouse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import (
    AppSettings,
    DatabricksSettings,
    DataSettings,
    Environment,
)
from app.llm.claude import ClaudeProvider
from app.memory.vector_store import ChromaVectorStore
from app.services.container import (
    ConfigurationError,
    Container,
    get_container,
    set_container,
)
from app.services.model_registry import MLflowModelRegistry
from data.repositories.base import DatasetNotFoundError
from data.repositories.databricks import DatabricksDataRepository
from data.repositories.local import LocalDataRepository
from tests.conftest import build_settings

pytestmark = pytest.mark.unit


def test_local_environment_selects_local_implementations(container: Container) -> None:
    assert container.environment is Environment.LOCAL
    assert isinstance(container.data_repository, LocalDataRepository)
    assert isinstance(container.model_registry, MLflowModelRegistry)
    assert isinstance(container.vector_store, ChromaVectorStore)
    assert isinstance(container.llm_provider, ClaudeProvider)


def test_components_are_cached(container: Container) -> None:
    """Repeated access returns the same instance, not a new connection each time."""
    assert container.data_repository is container.data_repository
    assert container.llm_provider is container.llm_provider


def test_local_repository_resolves_paths_absolutely(container: Container) -> None:
    repo = container.data_repository
    assert isinstance(repo, LocalDataRepository)
    assert repo.parquet_root.is_absolute()


def test_databricks_environment_without_credentials_fails_clearly() -> None:
    settings = build_settings(app=AppSettings(_env_file=None, environment="databricks"))
    container = Container(settings)

    with pytest.raises(ConfigurationError) as exc_info:
        _ = container.data_repository

    message = str(exc_info.value)
    assert "DATABRICKS__HOST" in message
    assert "DATABRICKS__TOKEN" in message


def test_databricks_environment_without_warehouse_fails_clearly() -> None:
    settings = build_settings(
        app=AppSettings(_env_file=None, environment="databricks"),
        databricks=DatabricksSettings(
            _env_file=None,
            host="https://example.cloud.databricks.com",
            token="dapi-x",
        ),
    )
    with pytest.raises(ConfigurationError, match="DATABRICKS__WAREHOUSE_ID"):
        _ = Container(settings).data_repository


def test_databricks_environment_with_credentials_selects_databricks_repository() -> None:
    settings = build_settings(
        app=AppSettings(_env_file=None, environment="databricks"),
        databricks=DatabricksSettings(
            _env_file=None,
            host="https://example.cloud.databricks.com",
            token="dapi-x",
            warehouse_id="wh-123",
        ),
    )
    repo = Container(settings).data_repository
    assert isinstance(repo, DatabricksDataRepository)


def test_databricks_repository_methods_name_the_stage() -> None:
    """Stage 2 stubs must say what is missing, not raise a bare NotImplementedError."""
    settings = build_settings(
        app=AppSettings(_env_file=None, environment="databricks"),
        databricks=DatabricksSettings(
            _env_file=None,
            host="https://example.cloud.databricks.com",
            token="dapi-x",
            warehouse_id="wh-123",
        ),
    )
    repo = Container(settings).data_repository
    with pytest.raises(NotImplementedError, match="Stage 2"):
        repo.get_sales()


def test_local_repository_reports_missing_dataset_clearly(tmp_path: Path) -> None:
    """With no generated data, reads must say what to run rather than crash.

    Pointed at a temporary root rather than the configured one: whether a
    developer happens to have generated a dataset must not decide the outcome.
    """
    settings = build_settings(
        data=DataSettings(_env_file=None, parquet_root=tmp_path / "gold")
    )
    repository = Container(settings).data_repository

    with pytest.raises(DatasetNotFoundError, match="generate-data"):
        repository.get_sales()

    healthy, detail = repository.health_check()
    assert healthy is False
    assert "Step 2" in detail


def test_budget_trackers_are_not_shared(container: Container) -> None:
    """Per-investigation state must not leak between requests."""
    first = container.new_budget_tracker()
    second = container.new_budget_tracker()
    assert first is not second

    first.record_tool_call()
    assert second.tool_calls == 0


def test_health_checks_isolate_failures() -> None:
    """One broken dependency must not prevent the others from reporting."""
    settings = build_settings(app=AppSettings(_env_file=None, environment="databricks"))
    results = Container(settings).health_checks()

    names = {name for name, _, _ in results}
    assert {"data_repository", "model_registry", "vector_store", "llm_provider"} <= names

    data_ok = next(ok for name, ok, _ in results if name == "data_repository")
    assert data_ok is False

    detail = next(detail for name, _, detail in results if name == "data_repository")
    assert "DATABRICKS__HOST" in detail


def test_baseline_service_is_reachable_through_the_container(
    container: Container,
) -> None:
    """Step 4's service must be wired, not merely written.

    A service nothing can reach is not a seam - it is dead code that reads like
    a seam. Every later step gets its baseline through this property.
    """
    assert container.baseline_service is not None
    # Cached like the other components, so a request does not rebuild it.
    assert container.baseline_service is container.baseline_service


def test_baseline_service_construction_does_not_require_a_trained_model(
    container: Container,
) -> None:
    """Building the container must never depend on a training run having
    happened. On a clean checkout there is no model, and a container that
    refused to start would make the failure look like a configuration problem
    rather than a missing artifact."""
    service = container.baseline_service

    # Reaching `is_available` must answer rather than raise, whatever the state.
    assert isinstance(service.is_available, bool)


def test_baseline_model_is_reported_in_health_checks(container: Container) -> None:
    """A missing baseline should be visible on /health, not discovered when a
    user asks a question the platform cannot answer."""
    names = {name for name, _, _ in container.health_checks()}

    assert "baseline_model" in names


def test_forecasting_service_is_reachable_through_the_container(
    container: Container,
) -> None:
    """Step 5's service must be wired, not merely written."""
    assert container.forecasting_service is not None
    assert container.forecasting_service is container.forecasting_service


def test_forecast_model_is_reported_in_health_checks(container: Container) -> None:
    names = {name for name, _, _ in container.health_checks()}

    assert "forecast_model" in names


def test_tool_registry_carries_the_forecasting_tool(container: Container) -> None:
    """Step 1 left the registry empty and deferred tools to Step 13.

    Step 5 changes that for one tool: its brief asks for a working
    ``ForecastingTool`` contract rather than a placeholder, so the tool is
    registered as soon as the model behind it exists. The remaining tools still
    arrive with their own steps.
    """
    assert container.tool_registry.has("forecast_demand")

    spec = container.tool_registry.get("forecast_demand").spec()
    assert spec.permission == "run_model"
    assert "forecast" in spec.description.lower()


def test_registering_a_tool_does_not_load_its_model(container: Container) -> None:
    """Building the registry must not touch the filesystem for a model.

    The registry is constructed during container startup, where a missing model
    artifact should surface as an unhealthy check rather than a boot failure.
    """
    registry = container.tool_registry

    assert len(registry) >= 1
    # Reaching `is_available` answers without raising whether or not a model
    # has ever been trained.
    assert isinstance(container.forecasting_service.is_available, bool)


def test_tool_registry_is_stable(container: Container) -> None:
    """Cached, so repeated access returns the same registry.

    Matters because tools hold injected services: rebuilding the registry per
    access would create a fresh service - and a fresh model load - on every
    agent turn.
    """
    assert container.tool_registry is container.tool_registry
    assert container.tool_registry.names() == sorted(container.tool_registry.names())


def test_process_container_singleton_can_be_replaced(container: Container) -> None:
    set_container(container)
    assert get_container() is container
