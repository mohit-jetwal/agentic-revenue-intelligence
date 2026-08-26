"""XGBoost candidate (brief section 8, model 5).

Included because the brief asks for the comparison, and the expected outcome is
worth stating in advance so the result is not oversold either way: on tabular
count data with the same features and the same objective, XGBoost and LightGBM
usually land within noise of each other. **"No material difference" is a
legitimate finding**, and reporting it as one is more useful than manufacturing
a distinction from a gap smaller than the fold-to-fold variance.

Where they genuinely differ is handling of categoricals. LightGBM splits on them
natively; XGBoost needs them encoded, and the encoding is a modelling choice
that has to be made somewhere. This uses XGBoost's own categorical support
(``enable_categorical=True``) rather than one-hot, because one-hot on
``product_id``-adjacent columns would produce a very wide sparse matrix and
reintroduce exactly the memory problem that Step 4's Ridge candidate hit.

Objective is ``count:poisson``, matching :class:`~ml.baseline.models.LightGBMBaseline`,
so the comparison isolates the implementation rather than confounding it with a
different loss.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from app.observability.logging import get_logger
from ml.baseline.models import BaselineEstimator

logger = get_logger(__name__)


class XGBoostForecaster(BaselineEstimator):
    """Gradient boosting with a Poisson objective, as the LightGBM comparator."""

    name = "xgboost"

    #: Mirrors LightGBM's settings where the two have equivalents, so a
    #: difference in the comparison table reflects the library rather than a
    #: difference in how hard each was tuned.
    #:
    #: ``min_child_weight`` is deliberately absent - see
    #: :attr:`MIN_CHILD_SAMPLES_EQUIVALENT`, because the naive translation of
    #: LightGBM's setting is wrong by a factor of the target mean.
    DEFAULT_PARAMS: dict[str, Any] = MappingProxyType(  # type: ignore[assignment]
        {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "max_cat_to_onehot": 1,
            "verbosity": 0,
        }
    )

    #: Rows per leaf, translated into XGBoost's units at fit time.
    #:
    #: **The trap this exists to avoid.** LightGBM's ``min_child_samples``
    #: counts *rows*. XGBoost's ``min_child_weight`` sums *Hessians*, and under
    #: ``count:poisson`` the Hessian is approximately ``mu`` - so the parameter
    #: silently scales with the level of the target. Setting both to 50 does not
    #: give the two models comparable regularisation; on this data, where mean
    #: demand is ~38 units, it gives XGBoost roughly **38x less**.
    #:
    #: Measured, not theorised. With ``min_child_weight=50`` XGBoost scored
    #: 82.9% WMAPE at +58.2% bias - predicting a test mean of 57.9 against an
    #: actual 36.6, while fitting its own training fold perfectly. Scaling to
    #: ``50 x mean(y)`` moved it to 46.2% WMAPE at -3.6% bias.
    #:
    #: This matters beyond one hyper-parameter: reporting the unscaled run would
    #: have produced a confident and completely false "XGBoost is much worse than
    #: LightGBM" line in the comparison table.
    MIN_CHILD_SAMPLES_EQUIVALENT = 50

    def __init__(
        self,
        *,
        seed: int = 42,
        num_boost_round: int = 800,
        early_stopping_rounds: int = 50,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.params: dict[str, Any] = {**self.DEFAULT_PARAMS, "seed": seed}
        if params:
            self.params.update(params)
        #: True when the caller pinned it explicitly, in which case the
        #: Hessian-scaling below is skipped.
        self._min_child_weight_pinned = bool(params and "min_child_weight" in params)
        self._booster: xgb.Booster | None = None
        self._best_iteration: int | None = None
        self._feature_names: list[str] = []

    def _to_dmatrix(self, X: pd.DataFrame, y: pd.Series | None = None) -> xgb.DMatrix:
        return xgb.DMatrix(X, label=y, enable_categorical=True, missing=np.nan)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
    ) -> None:
        self._feature_names = list(X.columns)

        # Translate "rows per leaf" into XGBoost's Hessian units, using the
        # actual target level rather than assuming one. Skipped when the caller
        # pinned the value, so a deliberate override is still honoured.
        if not self._min_child_weight_pinned:
            mean_target = float(y.mean()) if len(y) else 1.0
            self.params["min_child_weight"] = max(
                1.0, self.MIN_CHILD_SAMPLES_EQUIVALENT * mean_target
            )
            logger.info(
                "forecast.xgboost_min_child_weight_scaled",
                rows_per_leaf=self.MIN_CHILD_SAMPLES_EQUIVALENT,
                mean_target=round(mean_target, 2),
                min_child_weight=round(self.params["min_child_weight"], 1),
            )

        train = self._to_dmatrix(X, y)
        watchlist: list[tuple[xgb.DMatrix, str]] = [(train, "train")]
        # Early stopping against a *temporally later* fold, never a random one.
        # Stopping on a random split would pick the round that best fits data
        # interleaved with training, which is not the question being asked.
        if X_valid is not None and y_valid is not None and not X_valid.empty:
            watchlist.append((self._to_dmatrix(X_valid[self._feature_names], y_valid), "valid"))

        self._booster = xgb.train(
            self.params,
            train,
            num_boost_round=self.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=(
                self.early_stopping_rounds if len(watchlist) > 1 else None
            ),
            verbose_eval=False,
        )
        self._best_iteration = getattr(self._booster, "best_iteration", None)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("xgboost booster is missing")

        matrix = self._to_dmatrix(X[self._feature_names])
        if self._best_iteration is not None:
            return np.asarray(
                self._booster.predict(
                    matrix, iteration_range=(0, self._best_iteration + 1)
                )
            )
        return np.asarray(self._booster.predict(matrix))

    def get_params(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            **{k: v for k, v in self.params.items() if k != "seed"},
        }

    def feature_importance(self) -> pd.DataFrame | None:
        """Gain importance.

        Gain rather than weight: weight counts how often a feature was split on,
        which rewards high-cardinality numerics for being easy to split rather
        than for being informative.
        """
        if self._booster is None:
            return None

        scores = self._booster.get_score(importance_type="gain")
        if not scores:
            return None

        return (
            pd.DataFrame(
                {"feature": list(scores.keys()), "importance": list(scores.values())}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def booster(self) -> xgb.Booster | None:
        return self._booster
