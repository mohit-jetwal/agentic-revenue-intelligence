"""The business-facing baseline model (brief sections 19-20).

Implements the Step 1 ABC :class:`~ml.baseline.interface.BaselineSalesModel`,
which speaks in dates and product ids rather than feature matrices.

**Why this is separate from the estimators.** ``ml/baseline/models.py`` holds
sklearn-shaped estimators that take a frame and a target and know nothing about
repositories. This class composes one of those with a repository and a feature
engineer. Section 19 forbids putting data loading inside the model class, and
the practical reason is testability: an estimator that needs a dataset on disk to
exercise cannot be unit-tested, and the arithmetic in a lag is exactly what you
want to test in isolation.

The persistence format is deliberately plain - a joblib pickle plus a JSON
sidecar - so the artifact is inspectable without loading it. MLflow holds the
authoritative copy; this is the local convenience path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository
from data.repositories.point_in_time import PointInTimeView
from features.contracts.specs import FEATURE_VERSION, current_code_version
from features.engineering import FeatureEngineer, FeatureRequest
from ml.base import InsufficientDataError, ModelMetadata, ModelNotFittedError
from ml.baseline.conformal import ConformalCalibration, add_intervals
from ml.baseline.interface import BaselineResult, BaselineSalesModel
from ml.baseline.models import BaselineEstimator
from ml.baseline.training import PromotionApproach, neutralise_promotions

logger = get_logger(__name__)

#: Below this many observed days, a product-store series cannot support lag and
#: rolling features and the model falls back (brief section 24).
COLD_START_DAYS = 60


@dataclass
class BaselinePrediction:
    """Per-row baseline output (brief section 17).

    The Step 1 ``BaselineResult`` is an *aggregate* over a slice, which is what
    a tool returns to an agent. Downstream models need the rows. Both exist
    rather than one replacing the other, because changing the Step 1 contract
    would break the interfaces later steps already reference.
    """

    frame: pd.DataFrame
    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str
    #: Rows that used the cold-start fallback rather than the model.
    fallback_rows: int = 0

    def to_records(self) -> list[dict[str, Any]]:
        # pandas types the keys as `Hashable`; every column here is a string, so
        # the cast states what is already true rather than papering over a risk.
        return cast("list[dict[str, Any]]", self.frame.to_dict(orient="records"))

    def aggregate(
        self,
        *,
        start_date: date,
        end_date: date,
        product_id: str | None = None,
        store_id: str | None = None,
        region: str | None = None,
    ) -> BaselineResult:
        """Collapse to the Step 1 slice-level result."""
        frame = self.frame
        baseline_units = float(frame["baseline_units"].sum())
        actual_units = float(frame["actual_units"].sum())
        baseline_revenue = float(frame.get("baseline_revenue", pd.Series(dtype=float)).sum())
        actual_revenue = float(frame.get("actual_revenue", pd.Series(dtype=float)).sum())

        revenue_gap = actual_revenue - baseline_revenue
        return BaselineResult(
            product_id=product_id,
            store_id=store_id,
            region=region,
            start_date=start_date,
            end_date=end_date,
            baseline_units=max(baseline_units, 0.0),
            baseline_revenue=max(baseline_revenue, 0.0),
            actual_units=max(actual_units, 0.0),
            actual_revenue=max(actual_revenue, 0.0),
            units_gap=actual_units - baseline_units,
            revenue_gap=revenue_gap,
            revenue_gap_pct=revenue_gap / baseline_revenue if baseline_revenue > 1e-9 else 0.0,
            baseline_lower=float(frame["baseline_lower"].sum())
            if "baseline_lower" in frame else None,
            baseline_upper=float(frame["baseline_upper"].sum())
            if "baseline_upper" in frame else None,
            # Significant when the aggregate actual falls outside the summed
            # interval. Conservative - summing per-row intervals overstates the
            # aggregate width, since errors partly cancel - so this errs toward
            # *not* declaring significance, which is the right direction for a
            # claim that will be shown to a business user.
            is_significant=bool(
                "baseline_lower" in frame
                and (
                    actual_units < float(frame["baseline_lower"].sum())
                    or actual_units > float(frame["baseline_upper"].sum())
                )
            ),
        )


class FittedBaselineModel(BaselineSalesModel):
    """A trained baseline, ready to answer business questions.

    Constructed by the training pipeline or loaded from disk. ``fit`` is not the
    entry point here - training happens through
    :mod:`ml.baseline.training`, which owns the temporal splits and the
    promotion-approach logic. This class exists to *use* the result.
    """

    name = "baseline_sales"

    def __init__(
        self,
        repository: DataRepository,
        *,
        estimator: BaselineEstimator | None = None,
        approach: PromotionApproach = PromotionApproach.EXCLUDE,
        feature_names: tuple[str, ...] = (),
        calibration: ConformalCalibration | None = None,
        model_version: str = "v1.0",
        metadata: ModelMetadata | None = None,
    ) -> None:
        super().__init__(repository)
        self.estimator = estimator
        self.approach = approach
        self.feature_names = feature_names
        self.calibration = calibration
        self.version = model_version
        self._metadata = metadata

    # -- ABC surface --------------------------------------------------------

    def fit(self, **kwargs: Any) -> ModelMetadata:
        """Not the training entry point.

        Training needs temporal splits, a calibration fold and a promotion
        approach - concerns that belong to the pipeline, not to a model object
        holding a repository. Pointing at the right place beats a partial
        implementation that quietly trains on a random split.
        """
        raise NotImplementedError(
            "Train through ml.baseline.pipeline.train_baseline_pipeline, which owns "
            "the temporal split, promotion approach and conformal calibration. "
            "Construct FittedBaselineModel from its result, or load a saved one."
        )

    def predict(  # type: ignore[override]
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
    ) -> BaselineResult:
        """Aggregate baseline for a slice - the Step 1 contract."""
        detail = self.predict_detail(
            start_date=start_date,
            end_date=end_date,
            product_ids=product_ids,
            store_ids=store_ids,
            region=region,
        )
        return detail.aggregate(
            start_date=start_date,
            end_date=end_date,
            product_id=product_ids[0] if product_ids and len(product_ids) == 1 else None,
            store_id=store_ids[0] if store_ids and len(store_ids) == 1 else None,
            region=region,
        )

    # -- per-row prediction -------------------------------------------------

    def predict_detail(
        self,
        *,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None = None,
        store_ids: list[str] | None = None,
        region: str | None = None,
        as_of_date: date | None = None,
    ) -> BaselinePrediction:
        """Per-row baseline over a slice (brief section 17).

        ``as_of_date`` defaults to ``end_date``: a baseline for a past window is
        normally computed with everything known by the end of it. Passing an
        earlier date reproduces what the model would have said at that time,
        which is what a backtest needs.
        """
        if self.estimator is None or not self.estimator.is_fitted:
            raise ModelNotFittedError("no fitted estimator; train or load one first")
        if start_date > end_date:
            raise ValueError(f"start_date {start_date} is after end_date {end_date}")

        view = self._view(as_of_date or end_date)
        panel = self._build_panel(
            view, start_date, end_date, product_ids, store_ids, region
        )
        if panel.empty:
            raise InsufficientDataError(
                f"no rows for {start_date}..{end_date} with the given filters; the "
                f"product and store may never have been co-listed"
            )

        return self._predict_panel(panel)

    def predict_panel(self, panel: pd.DataFrame) -> BaselinePrediction:
        """Predict over a pre-built feature panel.

        For the training pipeline and for tests, which already hold the panel
        and should not pay to rebuild it.
        """
        if self.estimator is None or not self.estimator.is_fitted:
            raise ModelNotFittedError("no fitted estimator")
        return self._predict_panel(panel)

    # -- internals ----------------------------------------------------------

    def _view(self, as_of_date: date) -> PointInTimeView:
        if isinstance(self._repository, PointInTimeView):
            return self._repository.as_of(as_of_date)
        return self._repository.as_of(as_of_date)

    def _build_panel(
        self,
        view: PointInTimeView,
        start_date: date,
        end_date: date,
        product_ids: list[str] | None,
        store_ids: list[str] | None,
        region: str | None,
    ) -> pd.DataFrame:
        engineer = FeatureEngineer(view)
        return engineer.build(
            FeatureRequest(
                start_date=start_date,
                end_date=end_date,
                product_ids=product_ids,
                store_ids=store_ids,
                region=region,
                # Approach EXCLUDE never saw promotion features, so building them
                # would produce columns the estimator rejects.
                promotion=self.approach is PromotionApproach.CONTROL,
                include_promotion_spend=False,
            )
        )

    def _predict_panel(self, panel: pd.DataFrame) -> BaselinePrediction:
        # An explicit check rather than `assert`: asserts are stripped under
        # `python -O`, so the narrowing would vanish in exactly the deployment
        # where an unfitted model would fail most obscurely.
        estimator = self.estimator
        if estimator is None:
            raise ModelNotFittedError(
                "no estimator is loaded; train one with scripts/train_baseline.py"
            )

        missing = [c for c in self.feature_names if c not in panel.columns]
        if missing:
            raise ValueError(
                f"panel is missing {len(missing)} training features, e.g. {missing[:5]}. "
                f"The feature version may have moved since the model was trained "
                f"(model expects {FEATURE_VERSION})."
            )

        X = panel[list(self.feature_names)]
        if self.approach is PromotionApproach.CONTROL:
            X = neutralise_promotions(X)

        predictions = estimator.predict(X)

        result = panel[["date", "product_id", "store_id"]].copy()
        result["actual_units"] = panel["units"].to_numpy(dtype=float)
        result["baseline_units"] = predictions

        # Cold start: a series too young for lags gets a coarser estimate, and
        # says so. A caller must be able to tell an estimate from a guess.
        fallback_mask = self._cold_start_mask(panel)
        fallback_rows = int(fallback_mask.sum())
        if fallback_rows:
            result.loc[fallback_mask, "baseline_units"] = self._cold_start_baseline(
                panel[fallback_mask], panel
            )
        result["fallback_used"] = fallback_mask.to_numpy()

        result["sales_gap"] = result["actual_units"] - result["baseline_units"]
        result["sales_gap_pct"] = np.where(
            result["baseline_units"] > 1e-9,
            result["sales_gap"] / result["baseline_units"] * 100.0,
            np.nan,
        )

        for column in ("promotion_flag", "stockout_flag", "category", "brand",
                       "region", "channel", "selling_price"):
            if column in panel.columns:
                result[column] = panel[column].to_numpy()

        if "selling_price" in result.columns:
            price = result["selling_price"].to_numpy(dtype=float)
            result["actual_revenue"] = result["actual_units"].to_numpy() * price
            result["baseline_revenue"] = result["baseline_units"].to_numpy() * price

        if self.calibration is not None:
            result = add_intervals(result, self.calibration)

        result["model_name"] = estimator.name
        result["model_version"] = self.version

        return BaselinePrediction(
            frame=result,
            model_name=estimator.name,
            model_version=self.version,
            dataset_version=self._dataset_version(),
            feature_version=FEATURE_VERSION,
            fallback_rows=fallback_rows,
        )

    def _cold_start_mask(self, panel: pd.DataFrame) -> pd.Series:
        """Rows whose series is too young for the model's features.

        Detected via ``product_age_days`` where available, otherwise by a null
        long-lag - both mean the same thing: the history the model relies on is
        not there yet.
        """
        if "product_age_days" in panel.columns:
            young = panel["product_age_days"].fillna(0) < COLD_START_DAYS
        else:
            young = pd.Series(False, index=panel.index)

        if "rolling_28_units" in panel.columns:
            young = young | panel["rolling_28_units"].isna()

        return young.fillna(False)

    def _cold_start_baseline(
        self, cold_rows: pd.DataFrame, panel: pd.DataFrame
    ) -> np.ndarray:
        """Category x channel mean for rows with no usable history.

        Coarse on purpose. With no history the honest estimate is "what does
        this kind of product sell in this kind of store", and pretending to more
        precision than that would be false confidence at exactly the moment a
        caller most needs to distrust the number.
        """
        group_keys = [k for k in ("category", "channel") if k in panel.columns]
        if not group_keys or "units" not in panel.columns:
            fallback = float(panel["units"].mean()) if "units" in panel else 0.0
            return np.full(len(cold_rows), fallback)

        established = panel
        if "product_age_days" in panel.columns:
            established = panel[panel["product_age_days"].fillna(0) >= COLD_START_DAYS]
        if established.empty:
            established = panel

        means = established.groupby(group_keys, observed=True)["units"].mean()
        overall = float(established["units"].mean())

        if len(group_keys) > 1:
            keys = list(zip(*[cold_rows[k] for k in group_keys], strict=True))
        else:
            keys = list(cold_rows[group_keys[0]])
        return np.array([float(means.get(key, overall)) for key in keys])

    def _dataset_version(self) -> str:
        try:
            return self._repository.dataset_version()
        except Exception:  # noqa: BLE001 - metadata must never break a prediction
            return "unknown"

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Persist the model and a readable metadata sidecar.

        MLflow holds the authoritative artifact; this is the local path used by
        the service. The JSON sidecar means the model's provenance is
        inspectable without unpickling anything.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "estimator": self.estimator,
                "approach": self.approach.value,
                "feature_names": self.feature_names,
                "calibration": self.calibration,
                "model_version": self.version,
            },
            directory / "model.joblib",
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": self.estimator.name if self.estimator else None,
                    "model_version": self.version,
                    "approach": self.approach.value,
                    "feature_version": FEATURE_VERSION,
                    "code_version": current_code_version(),
                    "n_features": len(self.feature_names),
                    "calibration": self.calibration.to_dict() if self.calibration else None,
                    "saved_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info(
            "baseline.saved",
            directory=str(directory),
            model=self.estimator.name if self.estimator else None,
        )
        return directory / "model.joblib"

    @classmethod
    def load_from(cls, directory: Path, repository: DataRepository) -> FittedBaselineModel:
        """Load a saved model."""
        directory = Path(directory)
        path = directory / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(
                f"no baseline model at {path}. Train one first:\n"
                f"    uv run python scripts/train_baseline.py --profile dev"
            )

        payload = joblib.load(path)
        return cls(
            repository,
            estimator=payload["estimator"],
            approach=PromotionApproach(payload["approach"]),
            feature_names=tuple(payload["feature_names"]),
            calibration=payload.get("calibration"),
            model_version=payload.get("model_version", "v1.0"),
        )
