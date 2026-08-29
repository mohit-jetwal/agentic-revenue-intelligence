"""Persisted uplift analysis (brief sections 27, 29).

Uplift is retrospective. Unlike a forecaster, there is nothing to "serve" for a
future date - the question is always about promotions that already ran. So what
gets persisted is not a predictor but an **analysis**: the effect estimates, the
per-event table, the diagnostics that justify them, and the CATE model that can
score a hypothetical promotion for Step 8.

That distinction shapes the class. :meth:`FittedUpliftModel.for_promotion` is a
lookup into a computed table, not an inference call. The only genuine prediction
here is :meth:`predict_uplift`, which asks the CATE model what a promotion on
covariates it has not seen would do - and that is exactly the operation a future
optimiser needs and the one most likely to be extrapolating, so it is flagged
rather than presented as equivalent.

**The treatment definition is persisted with the artifact.** An uplift number
means nothing without it, and two artifacts built under different definitions are
not comparable. Loading checks the fingerprint and warns when it has moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.base import ModelMetadata
from ml.promo_uplift.config import PromoUpliftConfig, PromoUpliftConfigError
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.exceptions import UnknownPromotionError, UpliftModelUnavailableError
from ml.promo_uplift.interface import PromoUpliftModel, UpliftResult
from ml.promo_uplift.pipeline import UpliftRun

logger = get_logger(__name__)

ARTIFACT_NAME = "uplift.joblib"
METADATA_NAME = "metadata.json"
REPORT_NAME = "uplift_report.md"
EVENTS_NAME = "event_impact.parquet"


@dataclass
class UpliftArtifact:
    """What is written to disk."""

    config: PromoUpliftConfig
    estimates: dict[str, EffectEstimate]
    selected: str | None
    selection_reason: str
    validation_status: str
    warnings: list[str]
    segments: dict[str, pd.DataFrame]
    cate_model: Any | None
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    treatment_definition: str
    config_fingerprint: str
    dataset_version: str | None
    trained_at: str


class FittedUpliftModel(PromoUpliftModel):
    """A completed uplift analysis, ready to answer questions about it."""

    name = "promo_uplift"

    def __init__(
        self,
        repository: Any,
        *,
        artifact: UpliftArtifact,
        event_impact: pd.DataFrame | None = None,
        model_version: str = "v1.0",
        metadata: ModelMetadata | None = None,
    ) -> None:
        super().__init__(repository)
        self._artifact = artifact
        self._events = (
            event_impact if event_impact is not None else pd.DataFrame()
        )
        self.version = model_version
        self._metadata = metadata

    # -- properties ---------------------------------------------------------

    @property
    def artifact(self) -> UpliftArtifact:
        return self._artifact

    @property
    def event_impact(self) -> pd.DataFrame:
        return self._events

    @property
    def headline(self) -> EffectEstimate | None:
        selected = self._artifact.selected
        return self._artifact.estimates.get(selected) if selected else None

    @property
    def validation_status(self) -> str:
        return self._artifact.validation_status

    # -- ABC surface --------------------------------------------------------

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """Not the training entry point.

        Estimating uplift needs a treatment definition, a control pool and a
        diagnostic suite - concerns owned by :mod:`ml.promo_uplift.pipeline`.
        Pointing at it beats a partial implementation that quietly skips the
        validation.
        """
        raise NotImplementedError(
            "Estimate through ml.promo_uplift.pipeline.run_uplift, which owns "
            "the treatment definition, control construction and causal "
            "diagnostics. Build a FittedUpliftModel from its result with "
            "FittedUpliftModel.from_run, or load a saved one."
        )

    def predict(  # type: ignore[override]
        self,
        *,
        promotion_id: str | None = None,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> UpliftResult:
        """The Step 1 contract: one aggregate result for a slice."""
        rows = self._select(promotion_id, product_ids, store_ids, start_date, end_date)
        if rows.empty:
            raise UnknownPromotionError(
                "no analysed promotion matches this request",
                promotion_ids=[promotion_id] if promotion_id else None,
                product_ids=product_ids,
                store_ids=store_ids,
            )
        return self._to_result(rows, promotion_id=promotion_id)

    def for_promotion(self, promotion_id: str) -> UpliftResult:
        """The result for one event.

        A table lookup, not a model call - the effect was estimated when the
        analysis ran.
        """
        return self.predict(promotion_id=promotion_id)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Score hypothetical promotions with the CATE model.

        This is the one genuinely predictive operation here, and it is what a
        future optimiser calls to evaluate candidate promotions that have not
        run. Treat the output accordingly: it is an extrapolation from the
        promotions that *did* run, and a candidate outside their covariate range
        is a guess rather than an estimate.
        """
        model = self._artifact.cate_model
        if model is None:
            raise UpliftModelUnavailableError(
                "this artifact has no CATE model, so conditional effects cannot "
                "be predicted; re-run with estimators.dr_learner enabled",
                model="dr_learner",
            )
        missing = [c for c in self._artifact.feature_names if c not in X.columns]
        if missing:
            raise ValueError(
                f"X is missing {len(missing)} covariates the CATE model was "
                f"fitted on, e.g. {missing[:5]}"
            )
        return np.asarray(model.predict(X[list(self._artifact.feature_names)]), dtype=float)

    # -- internals ----------------------------------------------------------

    def _select(
        self,
        promotion_id: str | None,
        product_ids: list[str] | None,
        store_ids: list[str] | None,
        start_date: Any,
        end_date: Any,
    ) -> pd.DataFrame:
        rows = self._events
        if rows.empty:
            return rows
        if promotion_id:
            rows = rows[rows["promotion_id"] == promotion_id]
        if product_ids:
            rows = rows[rows["product_id"].isin(product_ids)]
        if store_ids:
            rows = rows[rows["store_id"].isin(store_ids)]
        if start_date is not None and "start_date" in rows.columns:
            rows = rows[pd.to_datetime(rows["start_date"]) >= pd.to_datetime(start_date)]
        if end_date is not None and "end_date" in rows.columns:
            rows = rows[pd.to_datetime(rows["end_date"]) <= pd.to_datetime(end_date)]
        return rows

    def _to_result(self, rows: pd.DataFrame, *, promotion_id: str | None) -> UpliftResult:
        headline = self.headline
        incremental = float(rows["incremental_units"].sum())
        observed = float(rows["observed_units"].sum())
        baseline = observed - incremental
        spend = float(rows["promotion_spend"].fillna(0.0).sum())
        profit = float(rows["incremental_profit"].sum())

        single = rows.iloc[0] if len(rows) == 1 else None
        interval: tuple[float, float] | None = None
        if headline is not None and len(rows) > 1:
            # The aggregate interval is only meaningful when it comes from the
            # aggregate estimator. Scaling a single event's point estimate by a
            # panel-level relative interval would invent precision that was
            # never measured for that event.
            interval = headline.interval_pct()

        single_id = str(single["promotion_id"]) if single is not None else None
        return UpliftResult(
            promotion_id=promotion_id or single_id,
            product_id=str(single["product_id"]) if single is not None else None,
            store_id=str(single["store_id"]) if single is not None else None,
            start_date=pd.to_datetime(rows["start_date"]).min().date()
            if "start_date" in rows.columns
            else datetime.now(UTC).date(),
            end_date=pd.to_datetime(rows["end_date"]).max().date()
            if "end_date" in rows.columns
            else datetime.now(UTC).date(),
            baseline_units=max(baseline, 0.0),
            actual_units=max(observed, 0.0),
            incremental_units=incremental,
            uplift_pct=incremental / baseline if baseline > 0 else 0.0,
            incremental_revenue=float(rows["incremental_revenue"].sum()),
            incremental_profit=profit,
            promotion_spend=spend,
            roi=profit / spend if spend > 0 else None,
            confidence_interval=interval,
            method=self._artifact.selected,
            treatment_group_size=int(rows["treated_days"].sum()),
            assumptions=list(headline.assumptions) if headline else [],
        )

    # -- persistence --------------------------------------------------------

    @classmethod
    def from_run(cls, run: UpliftRun, repository: Any) -> FittedUpliftModel:
        """Build an artifact from a completed pipeline run."""
        artifact = UpliftArtifact(
            config=run.config,
            estimates=run.estimates,
            selected=run.selected,
            selection_reason=run.selection_reason,
            validation_status=run.validation_status,
            warnings=run.warnings(),
            segments=run.segments,
            cate_model=run.learner._model if run.learner else None,
            feature_names=run.covariates.feature_names,
            categorical_names=run.covariates.categorical_names,
            treatment_definition=run.config.treatment_definition(),
            config_fingerprint=run.config.fingerprint(),
            dataset_version=None,
            trained_at=datetime.now(UTC).isoformat(),
        )
        return cls(repository, artifact=artifact, event_impact=run.event_impact)

    def save(self, directory: Path) -> Path:
        """Write the artifact, the event table and a human-readable summary."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._artifact, directory / ARTIFACT_NAME)

        if not self._events.empty:
            self._events.to_parquet(directory / EVENTS_NAME, index=False)

        headline = self.headline
        metadata = {
            "model_name": self.name,
            "model_version": self.version,
            "selected_method": self._artifact.selected,
            "selection_reason": self._artifact.selection_reason,
            "validation_status": self._artifact.validation_status,
            "treatment_definition": self._artifact.treatment_definition,
            "config_fingerprint": self._artifact.config_fingerprint,
            "trained_at": self._artifact.trained_at,
            "uplift_pct": headline.ate_pct if headline else None,
            "events_analysed": len(self._events),
            "warnings": self._artifact.warnings,
        }
        (directory / METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        logger.info("promo_uplift.artifact_saved", directory=str(directory))
        return directory / ARTIFACT_NAME

    @classmethod
    def load_from(
        cls, directory: Path, repository: Any, *, config: PromoUpliftConfig | None = None
    ) -> FittedUpliftModel:
        """Load a saved analysis, checking it was built under today's definition."""
        path = directory / ARTIFACT_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no promo uplift artifact at {path}; run "
                f"scripts/estimate_uplift.py to produce one"
            )
        artifact: UpliftArtifact = joblib.load(path)

        events = pd.DataFrame()
        events_path = directory / EVENTS_NAME
        if events_path.is_file():
            events = pd.read_parquet(events_path)

        if config is not None and config.fingerprint() != artifact.config_fingerprint:
            # A warning, not a refusal. The stored analysis is still valid for
            # the definition it was built under, and that definition travels
            # with it. Refusing would make a config edit silently break every
            # existing result.
            logger.warning(
                "promo_uplift.config_fingerprint_changed",
                stored=artifact.config_fingerprint,
                current=config.fingerprint(),
            )

        model = cls(repository, artifact=artifact, event_impact=events)
        model._metadata = ModelMetadata(
            name=model.name,
            version=model.version,
            trained_at=datetime.fromisoformat(artifact.trained_at),
            metrics={"uplift_pct": model.headline.ate_pct} if model.headline else {},
        )
        return model


def default_output_dir(root: Path, *, sampled: bool) -> Path:
    """Where a run writes.

    Sampled runs write elsewhere, so a quick development pass cannot overwrite
    the full-panel artifact the service is serving. Step 6 learned this the hard
    way when a smoke run replaced a three-hour model.
    """
    return root / ("promo_uplift_sampled" if sampled else "promo_uplift")


def load_config_or_none(path: Path) -> PromoUpliftConfig | None:
    """Best-effort config load, for callers that can proceed without one."""
    try:
        from ml.promo_uplift.config import load_promo_uplift_config

        return load_promo_uplift_config(path)
    except (PromoUpliftConfigError, OSError):
        return None


__all__ = [
    "ARTIFACT_NAME",
    "FittedUpliftModel",
    "UpliftArtifact",
    "default_output_dir",
]
