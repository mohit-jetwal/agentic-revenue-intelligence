"""Dependency container - seam 3 of the dev/prod boundary.

Every environment-dependent choice in the platform is made here and nowhere
else. Business logic receives an interface and never asks which implementation
it got. If a ``if settings.app.environment == ...`` appears outside this module,
the abstraction has leaked and the Stage 2 migration has started to become a
rewrite.

Not a DI framework. ``dependency-injector`` and friends buy scoping,
auto-wiring and declarative overrides; with four factories and one switch, they
would add a dependency and a mental model in exchange for nothing. A class with
cached properties is the right size for the problem, and stays readable to a
reviewer who has never seen it before.

Components are constructed lazily and cached: building a repository should not
require a Claude key, and building an LLM provider should not require generated
data. That laziness is what lets ``GET /health`` report on each dependency
independently instead of failing wholesale on the first missing one.
"""

from __future__ import annotations

from functools import cached_property

from app.config.settings import Environment, Settings, VectorStoreBackend, get_settings
from app.guardrails.budget import BudgetTracker
from app.llm.base import LLMProvider
from app.llm.claude import ClaudeProvider
from app.memory.base import VectorStore
from app.memory.vector_store import ChromaVectorStore, DatabricksVectorSearchStore
from app.observability.logging import get_logger
from app.services.baseline_service import BaselineSalesService
from app.services.elasticity_service import ElasticityService
from app.services.forecast_service import ForecastingService
from app.services.model_registry import (
    DatabricksModelRegistry,
    MLflowModelRegistry,
    ModelRegistry,
)
from app.services.promo_uplift_service import PromoUpliftService
from app.tools.registry import ToolRegistry, build_default_registry
from data.repositories.base import DataRepository
from data.repositories.databricks import DatabricksDataRepository
from data.repositories.local import LocalDataRepository

logger = get_logger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when settings cannot produce a working component."""


class Container:
    """Constructs and caches the platform's components for one environment."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        logger.info(
            "container.initialised",
            environment=self.settings.app.environment.value,
            version=self.settings.app.version,
        )

    @property
    def environment(self) -> Environment:
        return self.settings.app.environment

    # -- data ---------------------------------------------------------------

    @cached_property
    def data_repository(self) -> DataRepository:
        """Analytical data access. DuckDB/Parquet locally, Databricks SQL in prod."""
        data = self.settings.data
        if self.environment is Environment.LOCAL:
            return LocalDataRepository(
                parquet_root=self.settings.resolve(data.parquet_root),
                duckdb_path=self.settings.resolve(data.duckdb_path),
                max_result_rows=data.max_result_rows,
                query_timeout_seconds=data.query_timeout_seconds,
            )

        db = self.settings.databricks
        # Bind to locals so the emptiness checks below also narrow the types,
        # without an `assert` that would vanish under `python -O`.
        host = db.host
        token = db.token.get_secret_value() if db.token is not None else ""
        warehouse_id = db.warehouse_id

        if not host or not token:
            raise ConfigurationError(
                "APP__ENVIRONMENT=databricks requires DATABRICKS__HOST and "
                "DATABRICKS__TOKEN. See .env.example."
            )
        if not warehouse_id:
            raise ConfigurationError(
                "APP__ENVIRONMENT=databricks requires DATABRICKS__WAREHOUSE_ID."
            )
        return DatabricksDataRepository(
            host=host,
            token=token,
            warehouse_id=warehouse_id,
            catalog=db.catalog,
            schema=db.gold_schema,
            max_result_rows=data.max_result_rows,
            query_timeout_seconds=data.query_timeout_seconds,
        )

    # -- models -------------------------------------------------------------

    @cached_property
    def model_registry(self) -> ModelRegistry:
        """Model discovery and loading. Local MLflow, or Unity Catalog in prod."""
        if self.environment is Environment.LOCAL:
            return MLflowModelRegistry(self.settings.ml)
        db = self.settings.databricks
        return DatabricksModelRegistry(
            self.settings.ml, catalog=db.catalog, schema=db.ml_schema
        )

    @cached_property
    def baseline_service(self) -> BaselineSalesService:
        """Baseline sales estimation (Step 4).

        Environment-independent: the service takes a ``DataRepository``, so it
        gets DuckDB locally and Databricks SQL in production without knowing
        which. That indifference is the whole point of seam 1 - the first real
        demonstration that a model built locally moves to Stage 2 untouched.

        Constructing this never loads the model. The service resolves it lazily
        on first prediction, so a container built on a clean checkout - where no
        model has been trained yet - still starts, and the missing artifact
        surfaces as a readable error at the point someone asks for a baseline.
        """
        return BaselineSalesService(self.data_repository, settings=self.settings)

    @cached_property
    def forecasting_service(self) -> ForecastingService:
        """Demand forecasting (Step 5).

        Environment-independent for the same reason the baseline service is: it
        takes a ``DataRepository`` and never asks which implementation it got.
        """
        return ForecastingService(self.data_repository, settings=self.settings)

    @cached_property
    def promo_uplift_service(self) -> PromoUpliftService:
        """Promotional uplift (Step 7).

        Environment-independent for the same reason as the two above. Lazy for a
        sharper reason: this service serves a *completed analysis*, and on a
        clean checkout none exists. Loading eagerly would make container startup
        depend on someone having run a causal study.
        """
        return PromoUpliftService(self.data_repository, settings=self.settings)

    @cached_property
    def elasticity_service(self) -> ElasticityService:
        """Own-price and cross-price elasticity (Step 8).

        No artifact to load: elasticity is estimated on demand, because the
        answer depends on the slice being asked about. An elasticity for one
        region is a different regression, not a filter over a stored one.
        """
        return ElasticityService(self.data_repository, settings=self.settings)

    # -- retrieval ----------------------------------------------------------

    @cached_property
    def vector_store(self) -> VectorStore:
        """Enterprise document retrieval. Chroma locally, Vector Search in prod."""
        vs = self.settings.vectorstore
        if vs.backend is VectorStoreBackend.CHROMA:
            return ChromaVectorStore(
                path=self.settings.resolve(vs.path),
                collection=vs.collection,
            )

        db = self.settings.databricks
        if not db.vector_search_endpoint:
            raise ConfigurationError(
                "VECTORSTORE__BACKEND=databricks requires "
                "DATABRICKS__VECTOR_SEARCH_ENDPOINT."
            )
        return DatabricksVectorSearchStore(
            endpoint=db.vector_search_endpoint,
            index_name=f"{db.catalog}.{db.ml_schema}.{vs.collection}",
        )

    # -- reasoning ----------------------------------------------------------

    @cached_property
    def llm_provider(self) -> LLMProvider:
        """The reasoning model.

        Identical in both environments - Claude is reached over the Anthropic
        API either way. Kept in the container so evaluation runs can substitute
        a recorded provider without touching agent code.
        """
        return ClaudeProvider(self.settings.llm)

    # -- tools --------------------------------------------------------------

    @cached_property
    def tool_registry(self) -> ToolRegistry:
        """The analytical capabilities available to agents.

        Populated as the models behind each tool are built. Step 5 adds
        ``forecast_demand``; the rest arrive with their models.

        The service is passed in rather than constructed inside the registry, so
        building the registry never loads a model - a missing artifact must not
        make container startup fail.
        """
        return build_default_registry(
            forecasting_service=self.forecasting_service,
            promo_uplift_service=self.promo_uplift_service,
            elasticity_service=self.elasticity_service,
        )

    # -- per-request objects ------------------------------------------------

    def new_budget_tracker(self) -> BudgetTracker:
        """A fresh budget for one investigation.

        Not cached: budgets are per-investigation state, and sharing one across
        requests would let an earlier investigation exhaust a later one.
        """
        return BudgetTracker.from_settings(self.settings.agent)

    # -- diagnostics --------------------------------------------------------

    def health_checks(self) -> list[tuple[str, bool, str]]:
        """Probe every dependency, isolating failures.

        Each component is constructed inside its own try block so one
        misconfigured dependency reports as unavailable rather than making the
        whole health endpoint fail - which would tell you nothing about the rest.
        """
        results: list[tuple[str, bool, str]] = []
        probes: list[tuple[str, object]] = [
            ("data_repository", lambda: self.data_repository.health_check()),
            ("model_registry", lambda: self.model_registry.health_check()),
            ("baseline_model", lambda: self.baseline_service.health_check()),
            ("forecast_model", lambda: self.forecasting_service.health_check()),
            ("vector_store", lambda: self.vector_store.health_check()),
            ("llm_provider", lambda: self.llm_provider.health_check()),
            # Inside the guarded loop, not appended after it. Once the registry
            # started injecting services into tools, building it could raise on a
            # misconfigured repository - and an unguarded probe would take the
            # whole health endpoint down, which is precisely what this method
            # exists to prevent.
            ("tool_registry", lambda: (True, f"{len(self.tool_registry)} tools registered")),
        ]
        for name, probe in probes:
            try:
                ok, detail = probe()  # type: ignore[operator]
            except Exception as exc:  # noqa: BLE001 - health checks must not raise
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results.append((name, ok, detail))
        return results


_container: Container | None = None


def get_container() -> Container:
    """Return the process-wide container, creating it on first use."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container | None) -> None:
    """Replace the process-wide container. Intended for tests and startup."""
    global _container
    _container = container
