"""Application configuration.

All settings are loaded from environment variables (and a local ``.env`` file),
using a double-underscore delimiter for nested sections::

    APP__ENVIRONMENT=local
    LLM__MODEL=claude-sonnet-5
    AGENT__MAX_TOOL_CALLS=25

The single most important field is :attr:`AppSettings.environment`. It is the
seam between Stage 1 (local) and Stage 2 (Databricks): the DI container reads
it to decide which implementation of every interface to construct. No other
module should branch on the environment.

Secrets are typed as :class:`~pydantic.SecretStr` so they cannot leak through
``repr()``, structured log rendering, or an accidental settings dump.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_FILE = PROJECT_ROOT / ".env"

# Shared section configuration, spread into each section's SettingsConfigDict
# alongside its own env_prefix. Declared as a SettingsConfigDict rather than a
# plain dict so ``**`` expansion stays type-checkable, and defined standalone
# rather than inherited from a base class because pydantic merges parent config
# into the child, which would collide with an explicit ``env_prefix`` keyword.
_SECTION_CONFIG = SettingsConfigDict(
    env_file=_ENV_FILE,
    env_file_encoding="utf-8",
    extra="ignore",
    frozen=True,
)


class Environment(StrEnum):
    """Deployment target. Drives implementation selection in the container."""

    LOCAL = "local"
    DATABRICKS = "databricks"


class LLMProviderName(StrEnum):
    CLAUDE = "claude"


class VectorStoreBackend(StrEnum):
    CHROMA = "chroma"
    DATABRICKS = "databricks"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP__", **_SECTION_CONFIG)

    environment: Environment = Environment.LOCAL
    debug: bool = False
    name: str = "agentic-revenue-intelligence"
    version: str = "0.1.0"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM__", **_SECTION_CONFIG)

    provider: LLMProviderName = LLMProviderName.CLAUDE
    api_key: SecretStr | None = None
    #: Model used for interpretation, synthesis and most agent turns.
    model: str = "claude-sonnet-5"
    #: Model used for planning / re-planning, where reasoning quality matters most.
    planner_model: str = "claude-opus-5"
    #: 0.0 by default: planning and tool selection should be reproducible.
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @property
    def is_configured(self) -> bool:
        """True when an API key is present. Checked at call time, not import time."""
        return self.api_key is not None and bool(self.api_key.get_secret_value())


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA__", **_SECTION_CONFIG)

    #: Root of the analytical (Gold) Parquet datasets, read via DuckDB.
    parquet_root: Path = Path("data/local/gold")
    #: DuckDB database file used as the local analytical query engine.
    duckdb_path: Path = Path("data/local/analytics.duckdb")
    #: Application state only - investigations, traces, feedback. Not analytics.
    app_database_url: str = "sqlite:///data/local/app_state.sqlite"
    query_timeout_seconds: int = Field(default=30, gt=0)
    #: Hard cap on rows returned to a tool, so one bad filter cannot exhaust memory.
    max_result_rows: int = Field(default=100_000, gt=0)


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML__", **_SECTION_CONFIG)

    #: SQLite locally; "databricks" in Stage 2.
    #:
    #: Not the bare ``file:./mlruns`` store that MLflow used to default to -
    #: current versions refuse it outright and tell you to migrate. SQLite is the
    #: supported local backend, it is what the model registry requires (the file
    #: store never supported registration), and it matches the choice already
    #: made for application state elsewhere in the project.
    tracking_uri: str = "sqlite:///data/local/mlflow.db"
    registry_uri: str | None = None
    experiment_name: str = "agentic-revenue-intelligence"
    #: Agents must only ever load models at an approved stage (brief section 30).
    model_stage: str = "Production"
    artifact_root: Path = Path("mlartifacts")


class AgentSettings(BaseSettings):
    """Execution budget. Every limit here exists to stop a runaway agent loop."""

    model_config = SettingsConfigDict(env_prefix="AGENT__", **_SECTION_CONFIG)

    max_iterations: int = Field(default=12, gt=0)
    max_tool_calls: int = Field(default=25, gt=0)
    max_execution_seconds: float = Field(default=300.0, gt=0)
    max_token_budget: int = Field(default=200_000, gt=0)


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTORSTORE__", **_SECTION_CONFIG)

    backend: VectorStoreBackend = VectorStoreBackend.CHROMA
    path: Path = Path("data/local/chroma")
    collection: str = "commercial_policies"
    top_k: int = Field(default=5, gt=0)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__", **_SECTION_CONFIG)

    log_level: str = "INFO"
    json_logs: bool = True
    trace_enabled: bool = True

    @field_validator("log_level")
    @classmethod
    def _normalise_level(cls, v: str) -> str:
        level = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return level


class DatabricksSettings(BaseSettings):
    """Stage 2 configuration. Present but unused while ``environment=local``."""

    model_config = SettingsConfigDict(env_prefix="DATABRICKS__", **_SECTION_CONFIG)

    host: str | None = None
    token: SecretStr | None = None
    warehouse_id: str | None = None
    catalog: str = "cpg_revenue_intelligence"
    gold_schema: str = "gold"
    features_schema: str = "features"
    ml_schema: str = "ml"
    vector_search_endpoint: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.token and self.token.get_secret_value())


class Settings(BaseSettings):
    """Root settings object. Obtain via :func:`get_settings`."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, path: Path) -> Path:
        """Resolve a configured relative path against the project root."""
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that every module observes identical configuration. Tests that
    need different values should call :func:`reset_settings_cache` after
    patching the environment.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Intended for tests only."""
    get_settings.cache_clear()
