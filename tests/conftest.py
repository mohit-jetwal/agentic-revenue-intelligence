"""Shared pytest fixtures.

Note on isolation: settings sections read a ``.env`` file when one exists, and
``get_settings()`` is cached. Without care, a test would pass on a clean
checkout and fail on a developer machine that has run ``Copy-Item .env.example
.env`` - the worst kind of flake, because it only appears on someone else's
computer.

Two measures handle it: every ``SECTION__`` environment variable is stripped,
and the fixtures construct sections with ``_env_file=None``, which disables
dotenv loading for that instance. (Patching the module-level config dict does
*not* work - pydantic binds ``env_file`` when the class is created, long before
a test could reach it.)
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.config.settings import (
    AgentSettings,
    AppSettings,
    DatabricksSettings,
    DataSettings,
    LLMSettings,
    MLSettings,
    ObservabilitySettings,
    Settings,
    VectorStoreSettings,
    reset_settings_cache,
)
from app.observability.metrics import METRICS
from app.services.container import Container, set_container

_ENV_PREFIXES = (
    "APP__",
    "LLM__",
    "DATA__",
    "ML__",
    "AGENT__",
    "VECTORSTORE__",
    "OBSERVABILITY__",
    "DATABRICKS__",
)


def build_settings(**overrides: object) -> Settings:
    """Settings with dotenv disabled on every section.

    Exposed as a helper (not just a fixture) so individual tests can build a
    variant - e.g. a Databricks-environment settings object - with the same
    isolation guarantees.
    """
    sections: dict[str, object] = {
        "app": AppSettings(_env_file=None),
        "llm": LLMSettings(_env_file=None),
        "data": DataSettings(_env_file=None),
        "ml": MLSettings(_env_file=None),
        "agent": AgentSettings(_env_file=None),
        "vectorstore": VectorStoreSettings(_env_file=None),
        "observability": ObservabilitySettings(_env_file=None),
        "databricks": DatabricksSettings(_env_file=None),
    }
    sections.update(overrides)
    return Settings(**sections)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip project environment variables and reset global state per test."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    reset_settings_cache()
    set_container(None)
    METRICS.reset()
    yield
    reset_settings_cache()
    set_container(None)


@pytest.fixture
def settings() -> Settings:
    """Default settings, uncontaminated by the local environment."""
    return build_settings()


@pytest.fixture
def container(settings: Settings) -> Container:
    return Container(settings)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """API client with lifespan executed, so ``app.state.container`` exists."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
