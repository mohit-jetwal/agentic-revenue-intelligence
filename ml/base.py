"""Analytical model interfaces.

Every deterministic model in the platform implements :class:`AnalyticalModel`.
The point of a shared base is not code reuse - a LightGBM forecaster and an
OR-Tools allocator share almost nothing - it is a shared *contract*:

* Models are constructed with a :class:`~data.repositories.base.DataRepository`
  and never touch storage directly.
* Every model carries :class:`ModelMetadata`, so any number it produces can be
  attributed to a version, a dataset and an MLflow run.
* ``predict`` returns a validated Pydantic model, not a bare float or an
  untyped dict. This is what lets the tool layer wrap results without knowing
  what kind of model produced them.

Concrete implementations arrive in Stage 1 Steps 4-11. The subclass interfaces
below fix the signatures now so the tool layer (Step 13) can be written against
them without waiting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from data.repositories.base import DataRepository


@dataclass(frozen=True)
class ModelMetadata:
    """Provenance carried by every fitted model."""

    name: str
    version: str
    dataset_version: str | None = None
    feature_version: str | None = None
    trained_at: datetime | None = None
    mlflow_run_id: str | None = None
    stage: str | None = None
    #: Evaluation metrics recorded at training time.
    metrics: dict[str, float] = field(default_factory=dict)
    #: An unapproved model must never be invoked by an agent (brief section 30).
    approved: bool = False


class ModelNotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``/``load``."""


class InsufficientDataError(ValueError):
    """Raised when the requested slice has too little history to model.

    Distinct from a generic error because it is *recoverable*: the Supervisor
    should widen the window or aggregate to a coarser grain rather than abandon
    the investigation.
    """


class AnalyticalModel[TResult: BaseModel](ABC):
    """Base class for every deterministic analytical model."""

    #: Registered model name. Must match the MLflow registry entry.
    name: str
    version: str

    def __init__(self, repository: DataRepository) -> None:
        self._repository = repository
        self._metadata: ModelMetadata | None = None

    @property
    def repository(self) -> DataRepository:
        return self._repository

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise ModelNotFittedError(
                f"{type(self).__name__} has no metadata; call fit() or load() first"
            )
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        return self._metadata is not None

    @abstractmethod
    def fit(self, **kwargs: Any) -> ModelMetadata:
        """Train the model and return its metadata.

        Implementations log parameters, metrics and artifacts to MLflow.
        """

    @abstractmethod
    def predict(self, **kwargs: Any) -> TResult:
        """Produce a validated result for the requested slice."""

    def load(self, version: str | None = None) -> ModelMetadata:
        """Load a fitted model from the registry.

        Default raises: not every model is registry-backed (the optimisers are
        solved fresh each call and have nothing to load).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support registry loading")
