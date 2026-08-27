"""The fitted forecaster: what implements ``ml.forecasting.interface``.

Separate from the estimators for the same reason Step 4 separated them.
``ml/forecasting/train.py`` holds sklearn-shaped objects that take a frame and a
target and know nothing about repositories; this class knows how to fetch its own
features, apply the fallback chain, attach calibrated intervals and answer the
question the interface actually asks - *"forecast this product for 30 days"*.

The practical reason is testability: an estimator that needs a dataset on disk to
be exercised is an estimator nobody unit-tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.observability.logging import get_logger
from app.schemas.domain import ForecastHorizon
from data.repositories.base import DataRepository
from features.contracts.specs import FEATURE_VERSION, current_code_version
from ml.base import InsufficientDataError, ModelMetadata, ModelNotFittedError
from ml.forecasting.baselines import HorizonSeasonalNaive
from ml.forecasting.config import ForecastConfig
from ml.forecasting.conformal import HorizonCalibration
from ml.forecasting.dataset import KEYS, TARGET_DATE, build_history
from ml.forecasting.interface import ForecastingModel, ForecastPoint, ForecastResult
from ml.forecasting.predict import (
    PREDICTED,
    ForecastFrame,
    generate_forecast,
    latest_supported_as_of,
    summarise_series,
)
from ml.forecasting.sampling import SeriesSample
from ml.forecasting.train import TrainedForecaster

logger = get_logger(__name__)

MODEL_VERSION = "v1.0"
ARTIFACT_NAME = "model.joblib"
METADATA_NAME = "metadata.json"


@dataclass
class _LoadedArtifacts:
    trained: TrainedForecaster
    config: ForecastConfig
    pairs: pd.DataFrame
    fallback: HorizonSeasonalNaive | None


class FittedForecastModel(ForecastingModel):
    """A trained demand forecaster, ready to serve.

    ``fit`` deliberately raises: training is a pipeline concern with a dataset,
    a split, several candidates and a comparison, and pretending it fits behind
    a single method call would hide the parts that make the result trustworthy.
    Use ``scripts/train_forecast.py`` and then :meth:`load_from`.
    """

    version = MODEL_VERSION

    def __init__(
        self,
        repository: DataRepository,
        *,
        trained: TrainedForecaster | None = None,
        config: ForecastConfig | None = None,
        pairs: pd.DataFrame | None = None,
        fallback: HorizonSeasonalNaive | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        super().__init__(repository)
        self.trained = trained
        self.config = config
        self.pairs = pairs if pairs is not None else pd.DataFrame(columns=list(KEYS))
        self.fallback = fallback
        self._metrics = metrics or {}
        self._history: pd.DataFrame | None = None

    # -- interface ----------------------------------------------------------

    def fit(self, **kwargs: Any) -> ModelMetadata:
        raise NotImplementedError(
            "training runs through the pipeline, not this class - it needs a "
            "horizon dataset, an embargoed split, several candidates and a "
            "comparison. Run scripts/train_forecast.py, then load_from()."
        )

    @property
    def is_fitted(self) -> bool:
        return self.trained is not None and self.trained.estimator.is_fitted

    @property
    def calibration(self) -> HorizonCalibration | None:
        return self.trained.calibration if self.trained else None

    @property
    def estimator_name(self) -> str:
        return self.trained.name if self.trained else "unfitted"

    def predict(  # type: ignore[override]
        self,
        *,
        horizon: ForecastHorizon,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        as_of: date | None = None,
    ) -> ForecastResult:
        """Forecast demand over the requested horizon.

        Aggregates across the requested slice: the daily ``points`` are summed
        over the matching series, which is the number a category manager asks
        for. Per-series detail is available through :meth:`predict_detail`.
        """
        detail = self.predict_detail(
            horizon=horizon,
            product_ids=product_ids,
            store_ids=store_ids,
            region=region,
            as_of=as_of,
        )
        frame = detail.frame

        has_interval = "lower_bound" in frame.columns and "upper_bound" in frame.columns
        aggregation: dict[str, tuple[str, str]] = {"predicted": (PREDICTED, "sum")}
        if has_interval:
            # Daily bounds summed across *series* - which is legitimate, unlike
            # summing across days. Independent series errors do partially cancel,
            # so this is conservative; the horizon total uses its own calibration.
            aggregation["lower"] = ("lower_bound", "sum")
            aggregation["upper"] = ("upper_bound", "sum")

        daily = (
            frame.groupby(TARGET_DATE, observed=True)
            .agg(**aggregation)
            .reset_index()
            .sort_values(TARGET_DATE)
        )

        points = [
            ForecastPoint(
                date=row[TARGET_DATE].date()
                if hasattr(row[TARGET_DATE], "date")
                else row[TARGET_DATE],
                predicted_units=max(float(row["predicted"]), 0.0),
                lower_bound=max(float(row["lower"]), 0.0) if has_interval else None,
                upper_bound=float(row["upper"]) if has_interval else None,
            )
            for _, row in daily.iterrows()
        ]

        return ForecastResult(
            product_id=product_ids[0] if product_ids and len(product_ids) == 1 else None,
            store_id=store_ids[0] if store_ids and len(store_ids) == 1 else None,
            region=region,
            horizon=horizon,
            points=points,
            total_predicted_units=max(detail.total_units, 0.0),
            total_predicted_revenue=detail.total_revenue(),
            backtest_metrics=dict(self._metrics),
            model_used=self.estimator_name,
        )

    def backtest(
        self,
        *,
        horizon: ForecastHorizon,
        n_splits: int = 3,
        product_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Recorded backtest metrics from training.

        Returns what was measured during training rather than refitting on
        demand. A backtest triggered from a serving path would take minutes and
        would tempt a caller to treat "run it again" as a way to get a number
        they preferred.
        """
        if not self._metrics:
            raise ModelNotFittedError(
                "no backtest metrics recorded; retrain with backtesting enabled"
            )
        return dict(self._metrics)

    # -- detail -------------------------------------------------------------

    def predict_detail(
        self,
        *,
        horizon: ForecastHorizon,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        as_of: date | None = None,
    ) -> ForecastFrame:
        """Per-series, per-day forecasts."""
        if self.trained is None or self.config is None:
            raise ModelNotFittedError("no trained forecaster loaded")

        pairs = self._select_pairs(product_ids, store_ids, region)
        if pairs.empty:
            raise InsufficientDataError(
                "no trained series match the requested product/store/region filters"
            )

        history = self._load_history()
        effective_as_of = as_of or latest_supported_as_of(
            self._repository.as_of(pd.to_datetime(history["date"]).dt.date.max()),
            horizon.days,
        )
        view = self._repository.as_of(effective_as_of)

        return generate_forecast(
            view,
            history,
            pairs,
            self.trained,
            self.config,
            as_of=effective_as_of,
            horizon_days=horizon.days,
            calibration=self.trained.calibration,
            fallback=self.fallback,
        )

    def series_totals(
        self,
        *,
        horizon: ForecastHorizon,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        detail = self.predict_detail(
            horizon=horizon,
            product_ids=product_ids,
            store_ids=store_ids,
            region=region,
            as_of=as_of,
        )
        calibration = self.trained.calibration if self.trained else None
        return summarise_series(detail, calibration=calibration)

    def _select_pairs(
        self,
        product_ids: list[str] | None,
        store_ids: list[str] | None,
        region: str | None,
    ) -> pd.DataFrame:
        """Filter the trained series to those the caller asked for.

        A model can only forecast series it was trained on. Silently returning
        an empty forecast for an unknown product would look like "no demand
        expected", which is a very different claim from "this product is not in
        the model".
        """
        pairs = self.pairs
        if product_ids:
            pairs = pairs[pairs["product_id"].isin(product_ids)]
        if store_ids:
            pairs = pairs[pairs["store_id"].isin(store_ids)]
        if region:
            stores = self._repository.get_stores(region=region)
            pairs = pairs[pairs["store_id"].isin(stores["store_id"].tolist())]
        return pairs

    def _load_history(self) -> pd.DataFrame:
        """Feature history for the trained series, built once and cached."""
        if self._history is None:
            if self.config is None:
                raise ModelNotFittedError("no configuration loaded")
            sample = SeriesSample(
                pairs=self.pairs,
                product_ids=sorted(self.pairs["product_id"].unique().tolist()),
                store_ids=sorted(self.pairs["store_id"].unique().tolist()),
            )
            self._history = build_history(self._repository, self.config, sample)
        return self._history

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Persist the estimator plus a readable JSON sidecar.

        The sidecar is not redundant with the joblib blob: it is the only part a
        human or a Databricks job can read without unpickling, and unpickling
        requires the exact environment that wrote it.
        """
        if self.trained is None or self.config is None:
            raise ModelNotFittedError("nothing to save")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "trained": self.trained,
                "config": self.config,
                "pairs": self.pairs,
                "fallback": self.fallback,
                "metrics": self._metrics,
            },
            directory / ARTIFACT_NAME,
        )

        metadata = {
            "model_name": self.name,
            "model_version": self.version,
            "estimator": self.trained.name,
            "feature_version": FEATURE_VERSION,
            "code_version": current_code_version(),
            "config_fingerprint": self.config.fingerprint(),
            "n_features": len(self.trained.feature_names),
            "n_series": len(self.pairs),
            "max_horizon": self.config.max_horizon,
            "split": self.trained.split.to_dict(),
            "calibration": (
                self.trained.calibration.to_dict() if self.trained.calibration else None
            ),
            "metrics": self._metrics,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        (directory / METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        logger.info(
            "forecast.saved", directory=str(directory), estimator=self.trained.name
        )
        return directory

    @classmethod
    def load_from(
        cls, directory: Path, repository: DataRepository
    ) -> FittedForecastModel:
        artifact = directory / ARTIFACT_NAME
        if not artifact.is_file():
            raise FileNotFoundError(
                f"no trained forecaster at {artifact}. Train one with "
                f"`uv run python scripts/train_forecast.py`."
            )

        payload = joblib.load(artifact)
        model = cls(
            repository,
            trained=payload["trained"],
            config=payload["config"],
            pairs=payload["pairs"],
            fallback=payload.get("fallback"),
            metrics=payload.get("metrics", {}),
        )
        logger.info("forecast.loaded", directory=str(directory))
        return model
