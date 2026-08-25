"""Candidate baseline estimators (brief sections 7-8, 20).

Three of them, deliberately spanning the complexity range: a naive benchmark, a
linear model, and a gradient-boosted one. Section 41 is explicit that the most
complex model does not win by default, and on a well-behaved panel the seasonal
naive is a genuine contender - so it is built to be beaten rather than to be
dismissed.

**Two layers, and the separation matters.** :class:`BaselineEstimator` here is
sklearn-shaped: it takes a feature frame and a target, and knows nothing about
repositories, dates or products. :class:`~ml.baseline.model.BaselineSalesModel`
(the Step 1 ABC) is business-shaped: it takes a date range and product ids, and
composes an estimator with a repository. Mixing the two would put data loading
inside the model class, which section 19 explicitly forbids and which makes the
estimator impossible to unit-test without a dataset on disk.

**On the target.** LightGBM uses ``objective="poisson"`` rather than a log
transform. Sales are over-dispersed counts, and the ``log1p`` / ``expm1``
round-trip introduces retransformation bias - by Jensen's inequality
``E[exp(X)] != exp(E[X])``, so the back-transformed mean is systematically low.
For a baseline that is the worst possible failure mode: a baseline biased low
manufactures phantom uplift on every promotion measured against it. Poisson's
log link handles the skew without the back-transform.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.observability.logging import get_logger

logger = get_logger(__name__)

#: Columns that identify a row rather than describe it. Never features - a model
#: given a raw date learns the training window and cannot extrapolate past it.
IDENTIFIER_COLUMNS: tuple[str, ...] = ("date", "product_id", "store_id")


@dataclass
class FitResult:
    """What a fit produced, beyond the model itself."""

    train_rows: int
    train_seconds: float
    feature_names: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    #: Best iteration when early stopping ran; None otherwise.
    best_iteration: int | None = None


class BaselineEstimator(ABC):
    """Sklearn-shaped estimator over a prepared feature frame.

    Implementations must be deterministic given a seed: two fits on identical
    inputs produce identical predictions, or the model comparison in
    :mod:`ml.baseline.comparison` is measuring noise.
    """

    #: Stable identifier, recorded in MLflow and in every prediction's metadata.
    name: str

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self._fitted = False
        self._fit_result: FitResult | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fit_result(self) -> FitResult:
        if self._fit_result is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        return self._fit_result

    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
    ) -> None:
        """Train. Called by :meth:`fit` once inputs are validated."""

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict. Called by :meth:`predict` once inputs are aligned."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> BaselineEstimator:
        """Fit, timing the run and recording what it saw."""
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if X.empty:
            raise ValueError(f"{self.name}: refusing to fit on an empty frame")

        started = time.perf_counter()
        self._fit(X, y, X_valid, y_valid)
        elapsed = time.perf_counter() - started

        self._fitted = True
        self._fit_result = FitResult(
            train_rows=len(X),
            train_seconds=elapsed,
            feature_names=tuple(X.columns),
            params=self.get_params(),
            best_iteration=getattr(self, "_best_iteration", None),
        )
        logger.info(
            "baseline.fitted",
            model=self.name,
            rows=len(X),
            features=len(X.columns),
            seconds=round(elapsed, 2),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected units. Never negative - demand cannot be.

        Clipping at zero is a modelling statement, not a cosmetic fix: a
        baseline of -3 units is not a defensible estimate of anything, and
        letting it through would make the uplift arithmetic downstream produce
        nonsense.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        if X.empty:
            return np.array([], dtype=float)

        predictions = np.asarray(self._predict(X), dtype=float)
        return np.clip(predictions, 0.0, None)

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def feature_importance(self) -> pd.DataFrame | None:
        """Feature importance, where the model provides it."""
        return None


# ---------------------------------------------------------------------------
# 1. Seasonal naive - the benchmark
# ---------------------------------------------------------------------------


class SeasonalNaiveBaseline(BaselineEstimator):
    """The benchmark every other model must beat (brief section 8).

    Blends two signals that a demand planner would reach for unprompted:

    * ``lag_364`` - the same weekday one year ago. 364 rather than 365 so the
      comparison lands on the same day of week; retail demand is far more
      sensitive to weekday than to calendar date, and a 365-day lag compares a
      Saturday to a Friday.
    * ``rolling_28`` - the recent four-week level, which carries trend that a
      year-old value cannot.

    The blend weight is a parameter rather than a fitted quantity, because the
    point of a benchmark is to be simple and stable. If a gradient-boosted model
    cannot beat this, the complexity is not earning its place - and on a
    well-behaved panel that is a real possibility, not a rhetorical one.

    Falls back through progressively coarser signals when a series is too young
    to have a year of history, which is also the cold-start path.
    """

    name = "seasonal_naive"

    #: Tried in order; the first with a usable value wins.
    FALLBACK_CHAIN: tuple[str, ...] = (
        "lag_364_units",
        "rolling_28_units",
        "rolling_14_units",
        "rolling_7_units",
        "lag_7_units",
        "lag_1_units",
    )

    def __init__(self, *, seed: int = 42, seasonal_weight: float = 0.4) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= seasonal_weight <= 1.0:
            raise ValueError(f"seasonal_weight must be in [0, 1], got {seasonal_weight}")
        self.seasonal_weight = seasonal_weight
        self._global_mean: float = 0.0

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> None:
        # The only fitted quantity: a last-resort constant for a series with no
        # history at all. Deliberately the training mean rather than zero -
        # predicting zero for a brand-new product is confidently wrong.
        self._global_mean = float(y.mean())

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        seasonal = self._first_available(X, ("lag_364_units",))
        recent = self._first_available(
            X, ("rolling_28_units", "rolling_14_units", "rolling_7_units")
        )

        # Where both signals exist, blend. Where only one does, use it whole -
        # blending against a NaN would propagate the NaN and lose the row.
        both = np.isfinite(seasonal) & np.isfinite(recent)
        blended = np.where(
            both,
            self.seasonal_weight * seasonal + (1.0 - self.seasonal_weight) * recent,
            np.where(np.isfinite(recent), recent, seasonal),
        )

        # Anything still missing walks the remaining fallback chain.
        if not np.isfinite(blended).all():
            fallback = self._first_available(X, self.FALLBACK_CHAIN)
            blended = np.where(np.isfinite(blended), blended, fallback)

        return np.where(np.isfinite(blended), blended, self._global_mean)

    @staticmethod
    def _first_available(X: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        """Coalesce across columns in preference order."""
        result = np.full(len(X), np.nan)
        for column in columns:
            if column not in X.columns:
                continue
            values = X[column].to_numpy(dtype=float)
            result = np.where(np.isfinite(result), result, values)
            if np.isfinite(result).all():
                break
        return result

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "seasonal_weight": self.seasonal_weight}


# ---------------------------------------------------------------------------
# 2. Ridge - is the relationship mostly linear?
# ---------------------------------------------------------------------------


class RidgeBaseline(BaselineEstimator):
    """Regularised linear regression (brief section 7, candidate 2).

    Worth building for what it tells you rather than for winning: if Ridge lands
    close to LightGBM, the relationship is largely linear and the extra
    machinery is buying little. If it lands far behind, the interactions are
    real and the boosting is justified.

    Ridge rather than plain OLS because the feature set is collinear by
    construction - ``lag_7``, ``lag_14`` and ``rolling_7`` all measure recent
    demand. OLS on collinear inputs produces unstable, uninterpretable
    coefficients that swing between refits.

    Categoricals are one-hot encoded with ``min_frequency``, which folds rare
    levels into an "infrequent" bucket. Without it, a store seen twice in
    training gets its own coefficient fitted on two observations.
    """

    name = "ridge"

    #: Cap on rows used to *fit* the coefficients.
    #:
    #: Unlike LightGBM, Ridge has no sparse or chunked path here: one-hot
    #: encoding produces a dense design matrix, so fitting on the full panel
    #: means materialising roughly ``rows x 200`` float64s at once. On the dev
    #: panel that is over 6 GB, which pushes a 16 GB machine into swap and turns
    #: a two-minute fit into a twenty-minute one.
    #:
    #: Subsampling costs almost nothing statistically. Ridge estimates a few
    #: hundred coefficients, and their standard errors scale as
    #: ``1/sqrt(n)`` - at 750,000 rows they are already far tighter than the
    #: model's own specification error, so the extra rows buy precision that
    #: rounds away. This is a memory decision, not an accuracy trade.
    MAX_FIT_ROWS = 750_000

    #: Rows per prediction chunk, for the same densification reason.
    PREDICT_CHUNK_ROWS = 250_000

    def __init__(self, *, seed: int = 42, alpha: float = 1.0, max_categories: int = 40) -> None:
        super().__init__(seed=seed)
        self.alpha = alpha
        self.max_categories = max_categories
        self._pipeline: Pipeline | None = None
        self._numeric: list[str] = []
        self._categorical: list[str] = []

    def _split_columns(self, X: pd.DataFrame) -> None:
        self._numeric = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
        ]
        self._categorical = [
            c for c in X.columns
            if c not in self._numeric and X[c].nunique(dropna=True) <= self.max_categories
        ]

    def _design(self, X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        numeric = X[self._numeric].astype(float)
        # Median rather than zero: a missing lag means "no history", and zero
        # would assert "sold nothing", which is a different and wrong claim.
        if fit:
            self._medians = numeric.median()
        numeric = numeric.fillna(self._medians).replace([np.inf, -np.inf], 0.0)

        if not self._categorical:
            return numeric.to_numpy()

        categories = X[self._categorical].astype(str)
        if fit:
            self._encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                sparse_output=False,
            )
            encoded = self._encoder.fit_transform(categories)
        else:
            encoded = self._encoder.transform(categories)

        return np.hstack([numeric.to_numpy(), encoded])

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> None:
        # Subsample before encoding, not after - the point is to never build the
        # full dense matrix at all. Seeded, so the fit stays reproducible.
        if len(X) > self.MAX_FIT_ROWS:
            sample = np.random.default_rng(self.seed).choice(
                len(X), size=self.MAX_FIT_ROWS, replace=False
            )
            sample.sort()  # preserve chronological order for readability
            X, y = X.iloc[sample], y.iloc[sample]
            logger.info(
                "baseline.ridge_subsampled",
                fit_rows=self.MAX_FIT_ROWS,
                reason="dense one-hot design matrix would not fit in memory",
            )

        self._split_columns(X)
        design = self._design(X, fit=True)

        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.seed)),
            ]
        )
        self._pipeline.fit(design, y.to_numpy(dtype=float))

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("ridge pipeline is missing")

        # Chunked for the same reason the fit is subsampled: encoding a million
        # rows in one call densifies to gigabytes, and prediction is trivially
        # row-independent so there is nothing to lose by splitting it.
        if len(X) <= self.PREDICT_CHUNK_ROWS:
            return np.asarray(self._pipeline.predict(self._design(X, fit=False)))

        parts = [
            np.asarray(
                self._pipeline.predict(
                    self._design(X.iloc[start : start + self.PREDICT_CHUNK_ROWS], fit=False)
                )
            )
            for start in range(0, len(X), self.PREDICT_CHUNK_ROWS)
        ]
        return np.concatenate(parts)

    def get_params(self) -> dict[str, Any]:
        return {"seed": self.seed, "alpha": self.alpha, "max_categories": self.max_categories}

    def feature_importance(self) -> pd.DataFrame | None:
        """Absolute standardised coefficients.

        Comparable across features only because the inputs were standardised
        first - raw coefficients on unscaled features measure units, not
        importance.
        """
        if self._pipeline is None:
            return None
        coefficients = self._pipeline.named_steps["ridge"].coef_

        names = list(self._numeric)
        if self._categorical:
            names.extend(self._encoder.get_feature_names_out(self._categorical).tolist())
        if len(names) != len(coefficients):
            return None

        return (
            pd.DataFrame({"feature": names, "importance": np.abs(coefficients)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 3. LightGBM - the expected winner
# ---------------------------------------------------------------------------


class LightGBMBaseline(BaselineEstimator):
    """Gradient-boosted trees with a Poisson objective (brief section 7).

    Poisson, not squared error, and not a log transform:

    * **Squared error** treats a 5-unit miss on a 10-unit SKU the same as on a
      500-unit one, so the fit is dominated by hero products and the long tail
      is ignored.
    * **log1p + expm1** introduces retransformation bias. ``E[exp(X)]`` exceeds
      ``exp(E[X])``, so back-transforming the mean prediction is systematically
      low - which for a baseline manufactures phantom uplift everywhere.
    * **Poisson** has a log link, so it handles the skew and multiplicative
      structure natively, and predicts on the count scale directly. It matches
      how the data was generated in Step 2, where demand is log-additive and
      drawn as a negative binomial.

    Native categorical handling, so ``category`` dtype columns pass through
    without one-hot expansion. At 300 products x 200 stores, one-hot would add
    500 columns for no gain.
    """

    name = "lightgbm"

    # A class-level mapping is the natural home for these, and `MappingProxyType`
    # makes the immutability RUF012 asks for real rather than merely annotated -
    # a shared mutable default that one instance edits would silently change
    # every subsequent fit in the same process.
    DEFAULT_PARAMS: Mapping[str, Any] = MappingProxyType({
        "objective": "poisson",
        "metric": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        # Mild L2. The feature set is collinear by construction, and unregularised
        # boosting on collinear inputs produces importances that shuffle between
        # refits even when accuracy is stable.
        "lambda_l2": 1.0,
        "verbosity": -1,
    })

    def __init__(
        self,
        *,
        seed: int = 42,
        num_boost_round: int = 800,
        early_stopping_rounds: int = 50,
        objective: str = "poisson",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.params = {**self.DEFAULT_PARAMS, "objective": objective, "seed": seed}
        if objective == "tweedie":
            self.params["metric"] = "tweedie"
            self.params.setdefault("tweedie_variance_power", 1.3)
        if params:
            self.params.update(params)
        self._booster: lgb.Booster | None = None
        self._best_iteration: int | None = None
        self._categorical: list[str] = []

    def _prepare(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        frame = X.copy()
        if fit:
            self._categorical = [
                c for c in frame.columns
                if isinstance(frame[c].dtype, pd.CategoricalDtype) or frame[c].dtype == object
            ]
        for column in self._categorical:
            if column in frame.columns:
                frame[column] = frame[column].astype("category")
        # Booleans as integers: LightGBM handles them, but the dtype round-trip
        # between fit and predict is a common source of silent misalignment.
        for column in frame.columns:
            if pd.api.types.is_bool_dtype(frame[column]):
                frame[column] = frame[column].astype("int8")
        return frame

    def _fit(
        self, X: pd.DataFrame, y: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> None:
        train_frame = self._prepare(X, fit=True)
        target = y.to_numpy(dtype=float)
        if (target < 0).any():
            raise ValueError("Poisson objective requires a non-negative target")

        train_set = lgb.Dataset(
            train_frame, label=target, categorical_feature=self._categorical or "auto"
        )

        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]

        if X_valid is not None and y_valid is not None and not X_valid.empty:
            valid_frame = self._prepare(X_valid, fit=False)[train_frame.columns]
            valid_set = lgb.Dataset(
                valid_frame, label=y_valid.to_numpy(dtype=float), reference=train_set
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")
            # Early stopping against a *temporally later* validation fold, so the
            # stopping point reflects generalisation forward in time rather than
            # to a random holdout.
            callbacks.append(
                lgb.early_stopping(
                    self.early_stopping_rounds, verbose=False, first_metric_only=True
                )
            )

        self._booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._best_iteration = self._booster.best_iteration or self.num_boost_round

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("booster is missing")
        frame = self._prepare(X, fit=False)
        expected = list(self._booster.feature_name())
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"prediction frame is missing training features: {missing[:10]}")
        return np.asarray(
            self._booster.predict(frame[expected], num_iteration=self._best_iteration)
        )

    def get_params(self) -> dict[str, Any]:
        return {
            **self.params,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame | None:
        """Split-gain importance.

        Gain answers "how much did this feature reduce loss", which is the right
        question, but it is biased toward high-cardinality continuous features
        that offer more split points. Permutation importance in
        :func:`permutation_importance` is the honest cross-check and disagreeing
        with gain is informative rather than alarming.
        """
        if self._booster is None:
            return None
        return (
            pd.DataFrame(
                {
                    "feature": self._booster.feature_name(),
                    "importance": self._booster.feature_importance(importance_type=importance_type),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def booster(self) -> lgb.Booster:
        if self._booster is None:
            raise RuntimeError("booster is missing; fit first")
        return self._booster


# ---------------------------------------------------------------------------
# Model-agnostic importance
# ---------------------------------------------------------------------------


def permutation_importance(
    estimator: BaselineEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 3,
    sample_size: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Model-agnostic importance by shuffling one feature at a time.

    Chosen over SHAP deliberately. SHAP gives per-row attributions, which is
    valuable when explaining an individual prediction to a user - but this model
    is consumed by other models, not by people, and the question here is the
    aggregate one: what drives baseline demand? Permutation answers that
    without a dependency, and unlike gain it is not biased toward
    high-cardinality features.

    Measured on a *sample*: shuffling a feature and re-predicting 60 times over
    5M rows costs minutes for a number that is stable at 50k.
    """
    from ml.baseline.evaluation import compute_metrics

    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)

    baseline_wmape = compute_metrics(y, estimator.predict(X)).wmape

    rows: list[dict[str, Any]] = []
    for feature in X.columns:
        deltas: list[float] = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[feature] = (
                shuffled[feature]
                .sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
                .to_numpy()
            )
            deltas.append(compute_metrics(y, estimator.predict(shuffled)).wmape - baseline_wmape)
        rows.append(
            {
                "feature": feature,
                # How much WMAPE worsens when the feature is destroyed.
                "importance": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    )


#: Registry for the comparison harness and the CLI.
ESTIMATORS: dict[str, type[BaselineEstimator]] = {
    SeasonalNaiveBaseline.name: SeasonalNaiveBaseline,
    RidgeBaseline.name: RidgeBaseline,
    LightGBMBaseline.name: LightGBMBaseline,
}


def build_estimator(name: str, *, seed: int = 42, **kwargs: Any) -> BaselineEstimator:
    """Construct an estimator by name."""
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](seed=seed, **kwargs)
