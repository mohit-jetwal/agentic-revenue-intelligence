"""Configuration loading and secret handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import (
    AgentSettings,
    AppSettings,
    DatabricksSettings,
    Environment,
    LLMSettings,
    ObservabilitySettings,
    Settings,
    get_settings,
)

pytestmark = pytest.mark.unit


def test_defaults_are_local(settings: Settings) -> None:
    assert settings.app.environment is Environment.LOCAL
    assert settings.app.debug is False
    assert settings.llm.provider.value == "claude"
    assert settings.llm.temperature == 0.0


def test_nested_env_var_overrides_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT__MAX_TOOL_CALLS", "7")
    assert AgentSettings(_env_file=None).max_tool_calls == 7


def test_environment_switch_parses_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__ENVIRONMENT", "databricks")
    assert AppSettings(_env_file=None).environment is Environment.DATABRICKS


def test_unknown_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__ENVIRONMENT", "gcp")
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_api_key_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret must not leak through repr, str, or a model dump."""
    monkeypatch.setenv("LLM__API_KEY", "sk-ant-supersecret")
    llm = LLMSettings(_env_file=None)

    assert "supersecret" not in repr(llm)
    assert "supersecret" not in str(llm)
    assert "supersecret" not in str(llm.model_dump())
    # ...but is retrievable deliberately.
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "sk-ant-supersecret"


def test_databricks_token_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS__TOKEN", "dapi-supersecret")
    assert "supersecret" not in repr(DatabricksSettings(_env_file=None))


def test_llm_is_configured_reflects_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert LLMSettings(_env_file=None).is_configured is False
    monkeypatch.setenv("LLM__API_KEY", "sk-ant-x")
    assert LLMSettings(_env_file=None).is_configured is True


def test_empty_api_key_counts_as_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """.env.example ships LLM__API_KEY= with no value; that must not read as set."""
    monkeypatch.setenv("LLM__API_KEY", "")
    assert LLMSettings(_env_file=None).is_configured is False


def test_databricks_is_configured_requires_host_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DatabricksSettings(_env_file=None).is_configured is False
    monkeypatch.setenv("DATABRICKS__HOST", "https://example.cloud.databricks.com")
    assert DatabricksSettings(_env_file=None).is_configured is False
    monkeypatch.setenv("DATABRICKS__TOKEN", "dapi-x")
    assert DatabricksSettings(_env_file=None).is_configured is True


def test_log_level_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSERVABILITY__LOG_LEVEL", "debug")
    assert ObservabilitySettings(_env_file=None).log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSERVABILITY__LOG_LEVEL", "chatty")
    with pytest.raises(ValidationError):
        ObservabilitySettings(_env_file=None)


def test_budget_limits_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero tool-call budget would deadlock every investigation."""
    monkeypatch.setenv("AGENT__MAX_TOOL_CALLS", "0")
    with pytest.raises(ValidationError):
        AgentSettings(_env_file=None)


def test_relative_paths_resolve_against_project_root(settings: Settings) -> None:
    resolved = settings.resolve(settings.data.parquet_root)
    assert resolved.is_absolute()
    assert resolved.is_relative_to(settings.project_root)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_settings_are_immutable(settings: Settings) -> None:
    """Frozen config prevents a component mutating shared state at runtime."""
    with pytest.raises(ValidationError):
        settings.app.debug = True  # type: ignore[misc]
