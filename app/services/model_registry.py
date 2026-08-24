"""Model registry abstraction.

Local MLflow in Stage 1, Databricks MLflow / Model Serving in Stage 2. The
tracking URI alone differs for experiment logging, but *loading* differs enough
(local artifact path versus a served endpoint) to justify an interface.

The governance rule this interface enforces is section 30 of the brief: an agent
must never invoke an unapproved model. :meth:`ModelRegistry.load` checks the
stage before returning, so "only approved models reach production reasoning" is
a property of the code path rather than a matter of discipline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.config.settings import MLSettings
from app.schemas.api import ModelInfo

_STEP = "Stage 1 Step 12 (MLflow)"
_STAGE = "Stage 2 (Databricks MLflow / Model Serving)"


class ModelNotApprovedError(RuntimeError):
    """Raised when a model exists but is not at an approved stage."""


class ModelRegistry(ABC):
    """Discovery and loading of versioned analytical models."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Registered models with version, stage and evaluation metrics."""

    @abstractmethod
    def get_model_info(self, name: str, version: str | None = None) -> ModelInfo:
        """Metadata for one model. Latest approved version when unspecified."""

    @abstractmethod
    def load(self, name: str, version: str | None = None) -> Any:
        """Load a model artifact.

        Implementations must refuse to return a model that is not at the
        configured approved stage, raising :class:`ModelNotApprovedError`.
        """

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Probe registry availability."""


class MLflowModelRegistry(ModelRegistry):
    """Local MLflow tracking store and registry (Stage 1)."""

    def __init__(self, settings: MLSettings) -> None:
        self._settings = settings

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"MLflowModelRegistry.{method}() is implemented in {_STEP}")

    def list_models(self) -> list[ModelInfo]:
        # Returns empty rather than raising: GET /models should answer "none
        # registered yet" honestly instead of failing before any model exists.
        return []

    def get_model_info(self, name: str, version: str | None = None) -> ModelInfo:
        raise self._not_yet("get_model_info")

    def load(self, name: str, version: str | None = None) -> Any:
        raise self._not_yet("load")

    def health_check(self) -> tuple[bool, str]:
        return True, f"mlflow tracking at {self._settings.tracking_uri} (no models yet)"


class DatabricksModelRegistry(ModelRegistry):
    """Unity Catalog model registry with Databricks Model Serving (Stage 2).

    Stage 2 notes: register models under ``<catalog>.<ml_schema>.<name>`` so
    model access is governed by the same Unity Catalog grants as the data.
    Prefer a served endpoint over loading artifacts into the API process -
    it keeps model dependencies out of the application image and lets model and
    application scale independently.
    """

    def __init__(self, settings: MLSettings, *, catalog: str, schema: str) -> None:
        self._settings = settings
        self.catalog = catalog
        self.schema = schema

    def _not_yet(self, method: str) -> NotImplementedError:
        return NotImplementedError(f"DatabricksModelRegistry.{method}() belongs to {_STAGE}")

    def list_models(self) -> list[ModelInfo]:
        raise self._not_yet("list_models")

    def get_model_info(self, name: str, version: str | None = None) -> ModelInfo:
        raise self._not_yet("get_model_info")

    def load(self, name: str, version: str | None = None) -> Any:
        raise self._not_yet("load")

    def health_check(self) -> tuple[bool, str]:
        return False, f"Databricks model registry not implemented. {_STAGE}"
