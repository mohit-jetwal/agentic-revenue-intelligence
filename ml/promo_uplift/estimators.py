"""Causal estimators: IPW, AIPW and the DR-learner (brief sections 11, 13, 14).

Every estimator here targets the same quantity - the **ATT**, the average effect
on the promotions that actually ran - and they differ only in what has to be true
for the answer to be right. That is the whole point of running several: agreement
is evidence, and disagreement localises which assumption is doing the work.

.. code-block:: text

    tau_ATT = E[ Y(1) - Y(0) | T = 1 ]

``Y(1)`` is observed for treated rows. ``Y(0)`` for those same rows never is, so
every method below is a different way of constructing it.

**IPW** reweights the control group to look like the treated one, using
``e/(1-e)``. Right if the propensity model is right. Fragile where scores
approach 1, since the weight diverges.

**AIPW** adds an outcome model and a residual correction:

.. code-block:: text

    tau = (1/n1) * sum_i [ T_i*(Y_i - mu0(X_i))
                           - (1-T_i) * e_i/(1-e_i) * (Y_i - mu0(X_i)) ]

The doubly robust property: this is consistent if **either** ``mu0`` **or**
``e`` is correctly specified, not both. Two chances to be right instead of one.
It is not magic - if both are wrong the estimate is wrong, and the property says
nothing about which of the two is more likely to be right on your data.

**DR-learner** regresses a doubly robust pseudo-outcome on covariates to get
``CATE(x)``, which is what segment-level uplift and Step 8's allocation need.

**Cross-fitting** is not optional. Nuisance models fitted on the same rows they
predict have optimistically small residuals, and the influence-function standard
error inherits that optimism - producing intervals that are too narrow in exactly
the direction that makes a null result look like a finding. Folds are assigned on
the **anchor date**, which is constant within a promotion, so no event is split
across folds and no event's idiosyncratic shock informs its own counterfactual.

**On the outcome scale.** The models predict units directly under a Poisson
objective rather than fitting ``log1p`` and exponentiating back. Step 6 hit
retransformation bias doing the latter: ``E[exp(X)] != exp(E[X])``, and the gap
grows with residual variance, so the correction is not a constant. A Poisson
objective keeps the multiplicative structure that suits demand data while
returning a conditional mean on the scale the residuals are taken on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import EstimationError
from ml.promo_uplift.features import CovariateFrame
from ml.promo_uplift.propensity import PropensityModel, att_weights, effective_sample_size

logger = get_logger(__name__)


@dataclass
class EffectEstimate:
    """One estimate of the ATT, with everything needed to judge it."""

    method: str
    #: Average incremental units per treated row.
    ate: float
    #: Effect as a fraction of the counterfactual baseline on treated rows. This
    #: is the "uplift %" a business reader wants.
    ate_pct: float
    #: Mean counterfactual units per treated row - the denominator above, exposed
    #: so a reader can reconstruct the percentage rather than trust it.
    baseline_units: float
    n_treated: int
    n_control: int

    #: Standard error, when the estimator has one. ``None`` is a real answer and
    #: is never replaced by a guess.
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    confidence_level: float | None = None

    #: What must be true for this number to be the causal effect.
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def has_interval(self) -> bool:
        return self.ci_lower is not None and self.ci_upper is not None

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero. False when there is no interval.

        Deliberately conservative: an estimate with no interval is not
        "significant by default". It is an estimate whose uncertainty was never
        established, and treating that as a positive finding is how a noisy
        number becomes a budget decision.
        """
        lower, upper = self.ci_lower, self.ci_upper
        if lower is None or upper is None:
            return False
        return lower > 0 or upper < 0

    def interval_pct(self) -> tuple[float, float] | None:
        """The interval expressed as uplift percentages."""
        lower, upper = self.ci_lower, self.ci_upper
        if lower is None or upper is None or self.baseline_units <= 0:
            return None
        return (lower / self.baseline_units, upper / self.baseline_units)

    def summary(self) -> str:
        interval = self.interval_pct()
        band = f" [{interval[0]:+.1%}, {interval[1]:+.1%}]" if interval else " (no interval)"
        return f"{self.method}: {self.ate_pct:+.1%}{band} on {self.n_treated:,} treated rows"


class UpliftEstimator(Protocol):
    """The contract every estimator satisfies.

    Deliberately not tied to pandas or to any one library: ``fit`` takes arrays
    and a covariate frame, so a future estimator backed by something else drops
    in without the pipeline knowing.
    """

    name: str

    def fit(self, data: CovariateFrame) -> UpliftEstimator: ...

    def estimate_ate(self) -> EffectEstimate: ...

    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class NuisanceFit:
    """Cross-fitted nuisance predictions.

    ``mu0`` and ``mu1`` are conditional means of the outcome under control and
    treatment; ``propensity`` is ``P(T=1|X)``. Each value is predicted by a model
    that never saw its own row.
    """

    mu0: np.ndarray
    mu1: np.ndarray
    propensity: np.ndarray
    folds: np.ndarray
    #: Out-of-fold outcome-model fit, for the report. Not a causal quantity - a
    #: badly fitting outcome model is a warning, not a verdict, because AIPW can
    #: still be consistent through the propensity side.
    mu0_r2: float | None = None


def assign_folds(
    series: pd.Series,
    anchor: pd.Series,
    n_folds: int,
    *,
    scheme: str = "series",
    seed: int = 42,
) -> np.ndarray:
    """Cross-fitting fold labels.

    Two schemes, and the choice is not cosmetic.

    ``"series"`` (default) holds out **whole product-store listings**. Every row
    of a listing - and therefore every row of every promotion on it - lands in
    one fold, so nothing about a listing informs its own counterfactual. It also
    respects serial correlation: rows within a listing are far from independent,
    and a scheme that splits them treats one listing as hundreds of observations.
    It matches the bootstrap's resampling unit, so the two uncertainty
    calculations rest on the same independence assumption.

    ``"time_blocks"`` cuts contiguous date ranges instead. It sounds safer - it
    is what a forecasting split would do - and here it is actively harmful.
    Every fold is then predicted by a model trained only on *other* time periods,
    so any covariate with a time trend is extrapolated rather than interpolated.
    Measured on the synthetic panel, a linear ``time_index`` under time-block
    folds drove propensity scores to the clip boundaries and left the control
    weights summing to 43x the treated count; the AIPW estimate came back at
    -424% against a true +65%. Kept as an option because it is the right choice
    if the analysis period spans a genuine regime change, but it is not the
    default and the reason is measured rather than theoretical.

    ``anchor`` is retained for the time-block scheme, where fold boundaries must
    fall on the event anchor so a promotion is never split.
    """
    if scheme == "series":
        codes = pd.factorize(series)[0]
        rng = np.random.default_rng(seed)
        n_series = int(codes.max()) + 1 if len(codes) else 0
        labels = rng.permutation(n_series) % n_folds
        return labels[codes].astype(int)

    if scheme == "time_blocks":
        values = pd.to_datetime(anchor)
        ranks = values.rank(method="dense").to_numpy()
        return np.floor((ranks - 1) / ranks.max() * n_folds).astype(int).clip(0, n_folds - 1)

    raise EstimationError(
        f"unknown cross-fitting scheme {scheme!r}; expected 'series' or 'time_blocks'",
        method="cross_fitting",
    )


def fit_nuisances(
    data: CovariateFrame,
    *,
    config: PromoUpliftConfig | None = None,
    anchor: pd.Series | None = None,
) -> NuisanceFit:
    """Cross-fitted outcome and propensity models."""
    settings = config or get_promo_uplift_config()
    X = data.X
    t = data.t
    y = data.y

    if anchor is None:
        anchor = data.frame.get("_anchor_date", data.frame["date"])

    listing = data.frame[["product_id", "store_id"]].agg("|".join, axis=1)
    folds = assign_folds(
        listing,
        anchor,
        settings.cross_fitting.n_folds,
        scheme=settings.cross_fitting.scheme,
        seed=settings.outcome_model.seed,
    )

    mu0 = np.zeros(len(X), dtype=float)
    mu1 = np.zeros(len(X), dtype=float)
    propensity = np.zeros(len(X), dtype=float)

    numeric = data.numeric_names()
    categorical = data.categorical_names

    for fold in np.unique(folds):
        train = folds != fold
        predict = folds == fold
        if not train.any() or not predict.any():
            continue

        X_train = X[train]
        t_train = t[train]
        y_train = y[train]

        # Outcome models are fitted per arm - a T-learner structure. Fitting one
        # model with treatment as a feature lets a tree ignore it wherever the
        # other splits are more profitable, which shrinks the estimated effect
        # for reasons that are about the loss function, not the data.
        mu0[predict] = _fit_outcome(
            X_train[~t_train], y_train[~t_train], settings, categorical
        ).predict(X[predict])
        if t_train.any():
            mu1[predict] = _fit_outcome(
                X_train[t_train], y_train[t_train], settings, categorical
            ).predict(X[predict])

        model = PropensityModel(config=settings).fit(
            X_train, t_train, numeric=numeric, categorical=categorical
        )
        propensity[predict] = model.predict(X[predict])

    low, high = settings.propensity.clip
    propensity = np.clip(propensity, low, high)

    control = ~t
    residual_ss = float(np.sum((y[control] - mu0[control]) ** 2))
    total_ss = float(np.sum((y[control] - y[control].mean()) ** 2))
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else None

    logger.info(
        "promo_uplift.nuisances_fitted",
        folds=len(np.unique(folds)),
        rows=len(X),
        mu0_r2=round(r2, 4) if r2 is not None else None,
    )
    return NuisanceFit(mu0=mu0, mu1=mu1, propensity=propensity, folds=folds, mu0_r2=r2)


def _fit_outcome(
    X: pd.DataFrame,
    y: np.ndarray,
    config: PromoUpliftConfig,
    categorical: tuple[str, ...],
) -> LGBMRegressor:
    """One arm's outcome model.

    Poisson objective, not a log transform of the target. The two are close in
    spirit - both assume multiplicative structure - but the Poisson objective
    returns a conditional *mean* on the units scale, which is what the AIPW
    residuals require. Fitting ``log1p(y)`` and inverting gives a conditional
    median-like quantity that is biased low by an amount which varies with the
    residual variance, so the bias does not cancel between arms.
    """
    if len(y) == 0:
        raise EstimationError(
            "an outcome model was asked to fit on an empty arm; one of the "
            "cross-fitting folds contains no treated or no control rows",
            method="outcome_model",
        )

    objective = "poisson" if config.outcome_model.transform in {"poisson", "log1p"} else "l2"
    model = LGBMRegressor(
        objective=objective,
        verbose=-1,
        **config.outcome_model.params(),
    )
    model.fit(X, y, categorical_feature=list(categorical) or "auto")
    return model


def _clustered_standard_error(psi: np.ndarray, frame: pd.DataFrame) -> float:
    """Cluster-robust standard error, clustering on the product-store listing.

    The textbook influence-function standard error is ``sd(psi)/sqrt(n)``, which
    assumes the rows are independent. They are emphatically not: a listing
    running hot stays hot for weeks, so its residuals are strongly correlated
    and the effective number of independent observations is closer to the number
    of *listings* than the number of rows.

    This was measured, not anticipated. With the i.i.d. formula the intervals on
    the synthetic panels came out at 0.5-1.5 percentage points and failed to
    cover the known truth in four of six scenarios, while the point estimates
    were within 2-5 points throughout. The problem was never the estimate; it
    was an interval several times too narrow.

    The correction sums the influence function within each cluster before taking
    the variance, which is the standard sandwich form:

    .. code-block:: text

        SE = sqrt( sum_g ( sum_{i in g} psi_i )^2 ) / n

    It also matches what the bootstrap does - resampling whole series - so the
    analytic and resampled intervals now rest on the same independence
    assumption rather than two different ones.
    """
    n = len(psi)
    if n == 0:
        return 0.0

    keys = [c for c in ("product_id", "store_id") if c in frame.columns]
    if not keys:
        return float(psi.std(ddof=1) / np.sqrt(n))

    cluster = pd.factorize(frame[keys].astype(str).agg("|".join, axis=1))[0]
    totals = np.bincount(cluster, weights=psi)
    n_clusters = len(totals)
    if n_clusters < 2:
        return float(psi.std(ddof=1) / np.sqrt(n))

    # Finite-cluster correction, as used by standard cluster-robust estimators.
    # With few clusters the raw sandwich is biased downward, which is the same
    # failure this whole function exists to fix.
    correction = n_clusters / max(n_clusters - 1, 1)
    return float(np.sqrt(correction * np.sum(totals**2)) / n)


class IPWEstimator:
    """Inverse probability weighting, ATT form."""

    name = "inverse_probability_weighting"

    def __init__(self, *, config: PromoUpliftConfig | None = None) -> None:
        self._config = config or get_promo_uplift_config()
        self._data: CovariateFrame | None = None
        self._nuisance: NuisanceFit | None = None

    def fit(self, data: CovariateFrame, nuisance: NuisanceFit | None = None) -> IPWEstimator:
        self._data = data
        self._nuisance = nuisance or fit_nuisances(data, config=self._config)
        return self

    def estimate_ate(self) -> EffectEstimate:
        data, nuisance = self._require()
        t = data.t
        y = data.y
        weights = att_weights(
            nuisance.propensity,
            t,
            stabilise_at=self._config.propensity.stabilise_weights_at,
        )

        treated_mean = float(y[t].mean())
        control_weights = weights[~t]
        if control_weights.sum() <= 0:
            raise EstimationError(
                "all control weights are zero; no control row resembles the "
                "treated group closely enough to contribute",
                method=self.name,
            )
        # Hajek (self-normalised) rather than Horvitz-Thompson. The normalised
        # form is biased in finite samples but far less variable, and the
        # unnormalised version can produce a weighted mean outside the range of
        # the data when a few weights dominate.
        control_mean = float(np.dot(y[~t], control_weights) / control_weights.sum())

        effect = treated_mean - control_mean
        ess = effective_sample_size(control_weights)

        # A weighted-mean difference has no simple closed-form variance once the
        # propensity is estimated, so the interval comes from the bootstrap in
        # the pipeline rather than from an approximation here. Reporting an
        # analytic SE that ignores propensity estimation would understate the
        # uncertainty and look more precise than AIPW, which is backwards.
        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / control_mean if control_mean > 0 else 0.0,
            baseline_units=control_mean,
            n_treated=int(t.sum()),
            n_control=int((~t).sum()),
            assumptions=[
                "Treatment is ignorable given the pre-treatment covariates.",
                "Every treated unit had a non-zero chance of not being promoted.",
                "The propensity model is correctly specified - IPW has no second "
                "line of defence if it is not.",
            ],
            diagnostics={
                "effective_sample_size": ess,
                "effective_sample_fraction": ess / max(int((~t).sum()), 1),
                "max_weight": float(control_weights.max()),
            },
        )

    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray:
        raise EstimationError(
            "IPW estimates an average, not a conditional effect; use the "
            "DR-learner for CATE",
            method=self.name,
        )

    def _require(self) -> tuple[CovariateFrame, NuisanceFit]:
        if self._data is None or self._nuisance is None:
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        return self._data, self._nuisance


class AIPWEstimator:
    """Augmented IPW - the doubly robust ATT estimator."""

    name = "augmented_ipw"

    def __init__(self, *, config: PromoUpliftConfig | None = None) -> None:
        self._config = config or get_promo_uplift_config()
        self._data: CovariateFrame | None = None
        self._nuisance: NuisanceFit | None = None

    def fit(self, data: CovariateFrame, nuisance: NuisanceFit | None = None) -> AIPWEstimator:
        self._data = data
        self._nuisance = nuisance or fit_nuisances(data, config=self._config)
        return self

    def influence(self) -> np.ndarray:
        """Per-row influence function values.

        Exposed because the standard error is ``sd(psi)/sqrt(n)`` and a reader
        should be able to check that rather than accept it. It is also what makes
        the interval analytic instead of bootstrapped, which matters at panel
        scale where 200 bootstrap refits is minutes rather than seconds.
        """
        data, nuisance = self._require()
        treated = data.t
        t = treated.astype(float)
        y = data.y
        e = nuisance.propensity
        mu0 = nuisance.mu0

        share_treated = float(t.mean())
        if share_treated <= 0:
            raise EstimationError("no treated rows", method=self.name)

        # The same stabilised weights the balance diagnostics are computed on.
        # Using raw odds here while reporting balance on capped ones would mean
        # the diagnostic describes a different estimator than the one that
        # produced the number.
        odds = att_weights(
            e, treated, stabilise_at=self._config.propensity.stabilise_weights_at
        )
        residual = y - mu0
        # The ATT influence function. The first term is the treated residual; the
        # second removes the part of it explained by control rows that look
        # equally promotable. Where mu0 is perfect the residuals vanish and this
        # reduces to a difference of outcome-model predictions; where the
        # propensity is perfect the weighting alone carries it.
        raw = t * residual - (1.0 - t) * odds * residual
        tau = float(raw.mean() / share_treated)
        return (raw - t * tau) / share_treated

    def estimate_ate(self) -> EffectEstimate:
        data, nuisance = self._require()
        t = data.t
        y = data.y
        e = nuisance.propensity
        mu0 = nuisance.mu0

        n_treated = int(t.sum())
        if n_treated == 0:
            raise EstimationError("no treated rows", method=self.name)

        # The same stabilised weights the balance diagnostics are computed on.
        # Using raw odds here while reporting balance on capped ones would mean
        # the diagnostic describes a different estimator than the one that
        # produced the number.
        odds = att_weights(
            e, t, stabilise_at=self._config.propensity.stabilise_weights_at
        )
        residual = y - mu0
        effect = float(
            (residual[t].sum() - (odds[~t] * residual[~t]).sum()) / n_treated
        )

        psi = self.influence()
        se = _clustered_standard_error(psi, data.frame)

        alpha = self._config.uncertainty.alpha
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        # The counterfactual baseline is the outcome model's prediction on the
        # treated rows: what those store-days would have sold unpromoted. Using
        # the raw control mean instead would divide by a different population
        # and make the percentage incomparable to the effect above it.
        baseline = float(mu0[t].mean())

        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(effect / se)))) if se > 0 else None

        warnings: list[str] = []
        if nuisance.mu0_r2 is not None and nuisance.mu0_r2 < 0.1:
            warnings.append(
                f"the control outcome model explains only "
                f"{nuisance.mu0_r2:.1%} of variance out of fold, so the "
                f"doubly robust correction rests almost entirely on the "
                f"propensity model being right"
            )

        # Calibration check. Since E[(1-T) * e/(1-e)] = E[e] = P(T=1), the
        # control odds must sum to roughly the treated count. When they do not,
        # the propensity model is miscalibrated in level and every weighted term
        # above is scaled by the same error - which is exactly how a time-block
        # fold structure once turned a true +65% into -424% here. Cheap to
        # compute, and it fails loudly in the one place the estimate is silently
        # wrong.
        weight_sum = float(odds[~t].sum())
        calibration = weight_sum / n_treated if n_treated else float("nan")
        if not 0.7 <= calibration <= 1.4:
            warnings.append(
                f"propensity calibration is off: control weights sum to "
                f"{weight_sum:,.0f} against {n_treated:,} treated rows "
                f"(ratio {calibration:.2f}, expected ~1.0). The weighted terms "
                f"in the estimator are scaled by this error, so the effect is "
                f"not trustworthy until the assignment model is fixed"
            )

        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / baseline if baseline > 0 else 0.0,
            baseline_units=baseline,
            n_treated=n_treated,
            n_control=int((~t).sum()),
            standard_error=se,
            ci_lower=effect - critical * se,
            ci_upper=effect + critical * se,
            p_value=p_value,
            confidence_level=self._config.uncertainty.confidence_level,
            assumptions=[
                "Treatment is ignorable given the pre-treatment covariates.",
                "Every treated unit had a non-zero chance of not being promoted.",
                "Consistent if EITHER the outcome model or the propensity model "
                "is correctly specified - not necessarily both.",
                "The interval is asymptotic and assumes the cross-fitted "
                "nuisance models converge fast enough for the remainder to "
                "vanish.",
            ],
            warnings=warnings,
            diagnostics={
                "mu0_r2": nuisance.mu0_r2 if nuisance.mu0_r2 is not None else float("nan"),
                "mean_propensity_treated": float(e[t].mean()),
                "mean_propensity_control": float(e[~t].mean()),
                "max_control_odds": float(odds[~t].max()) if (~t).any() else 0.0,
                "weight_calibration": calibration,
            },
        )

    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray:
        raise EstimationError(
            "AIPW estimates an average; use the DR-learner for CATE", method=self.name
        )

    def _require(self) -> tuple[CovariateFrame, NuisanceFit]:
        if self._data is None or self._nuisance is None:
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        return self._data, self._nuisance


class DRLearner:
    """Doubly robust learner for conditional effects."""

    name = "dr_learner"

    def __init__(self, *, config: PromoUpliftConfig | None = None) -> None:
        self._config = config or get_promo_uplift_config()
        self._data: CovariateFrame | None = None
        self._nuisance: NuisanceFit | None = None
        self._model: LGBMRegressor | None = None
        self._pseudo: np.ndarray | None = None
        self._winsor_share: float = 0.0

    def fit(self, data: CovariateFrame, nuisance: NuisanceFit | None = None) -> DRLearner:
        self._data = data
        self._nuisance = nuisance or fit_nuisances(data, config=self._config)

        pseudo = self.pseudo_outcome()
        # Winsorise before fitting. The score carries a 1/e or 1/(1-e) factor, so
        # with propensities clipped at 0.02 a single row can contribute fifty
        # times the outcome. An L2 regression on a target that heavy-tailed is
        # fitted largely to its extremes: measured on the confounded synthetic
        # panel the unwinsorised learner returned +91% against a true +63%.
        # Trimming the outermost 1% of scores at each tail is standard for
        # DR-learners and is reported rather than silent.
        self._winsor_share = float(
            np.mean((pseudo < np.percentile(pseudo, 1)) | (pseudo > np.percentile(pseudo, 99)))
        )
        pseudo = np.clip(pseudo, np.percentile(pseudo, 1), np.percentile(pseudo, 99))
        self._pseudo = pseudo

        # An L2 objective here, not Poisson: the pseudo-outcome is a signed
        # effect and is routinely negative, which a Poisson likelihood cannot
        # represent at all.
        model = LGBMRegressor(
            objective="l2",
            verbose=-1,
            **self._config.outcome_model.params(),
        )
        model.fit(data.X, pseudo, categorical_feature=list(data.categorical_names) or "auto")
        self._model = model
        return self

    def pseudo_outcome(self) -> np.ndarray:
        """The doubly robust score, one value per row.

        .. code-block:: text

            Y* = mu1(X) - mu0(X)
                 + T*(Y - mu1(X))/e(X)
                 - (1-T)*(Y - mu0(X))/(1-e(X))

        Its conditional mean given ``X`` is the CATE, so regressing it on ``X``
        estimates the CATE directly. Individually these values are extremely
        noisy - a single row's score can be wildly negative - which is why they
        are smoothed by a regression rather than read one at a time.
        """
        data, nuisance = self._require_data()
        t = data.t.astype(float)
        y = data.y
        e = nuisance.propensity
        return (
            nuisance.mu1
            - nuisance.mu0
            + t * (y - nuisance.mu1) / np.clip(e, 1e-12, None)
            - (1.0 - t) * (y - nuisance.mu0) / np.clip(1.0 - e, 1e-12, None)
        )

    def estimate_cate(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        return np.asarray(self._model.predict(X), dtype=float)

    def estimate_ate(self) -> EffectEstimate:
        """The ATT, as the mean **fitted** CATE over treated rows.

        Averaging the raw pseudo-outcomes instead would be unbiased in theory
        and unusable in practice. Each score contains a ``1/e`` or ``1/(1-e)``
        term, so a treated row with a small propensity contributes a value tens
        of times the outcome, and the mean is dominated by whichever handful of
        rows drew the most extreme weights. Measured on the confounded synthetic
        panel, the raw average came back at +98% against a true +63%; the fitted
        CATE - which regresses those scores on covariates and so pools
        information across similar rows - lands far closer.

        The interval still comes from the spread of the pseudo-outcomes, which
        ignores the smoothing and is therefore conservative. Overstating
        uncertainty is the safe direction for a number that will be used to
        allocate budget.
        """
        data, _ = self._require_data()
        if self._pseudo is None:
            raise EstimationError(f"{self.name} is not fitted", method=self.name)

        t = data.t
        treated_scores = self._pseudo[t]
        effect = float(self.estimate_cate(data.X[t]).mean())
        # Clustered on the listing, for the same reason as AIPW: rows within a
        # product-store are far from independent.
        se = _clustered_standard_error(
            treated_scores - treated_scores.mean(), data.frame[t]
        ) * (len(treated_scores) / max(int(t.sum()), 1))

        nuisance = self._nuisance
        if nuisance is None:
            # An explicit check rather than `assert`: asserts are stripped under
            # `python -O`, so the narrowing would vanish in exactly the
            # deployment where an unfitted estimator fails most obscurely.
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        baseline = float(nuisance.mu0[t].mean())

        alpha = self._config.uncertainty.alpha
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / baseline if baseline > 0 else 0.0,
            baseline_units=baseline,
            n_treated=int(t.sum()),
            n_control=int((~t).sum()),
            standard_error=se,
            ci_lower=effect - critical * se,
            ci_upper=effect + critical * se,
            confidence_level=self._config.uncertainty.confidence_level,
            assumptions=[
                "Treatment is ignorable given the pre-treatment covariates.",
                "Consistent if EITHER nuisance model is correctly specified.",
                "Segment effects assume the CATE model generalises to the "
                "segment's covariate region; a segment with few treated rows is "
                "extrapolation, not estimation.",
                "This learner exists to rank segments. For the aggregate effect "
                "AIPW is the efficient estimator and is the headline number; "
                "the two are reported side by side so a divergence is visible.",
            ],
            diagnostics={
                "pseudo_outcome_std": float(self._pseudo.std(ddof=1)),
                "winsorised_share": self._winsor_share,
                "cate_spread": float(
                    np.percentile(self.estimate_cate(data.X), 90)
                    - np.percentile(self.estimate_cate(data.X), 10)
                ),
            },
        )

    def segment_effects(
        self, by: str, *, min_treated: int = 30
    ) -> pd.DataFrame:
        """Mean CATE by segment, over treated rows.

        Segments below ``min_treated`` are returned with a null effect rather
        than a number. A segment-level uplift computed from eight promotions is
        a rounding error with a label on it, and Step 8 would allocate budget
        against it.
        """
        data, _ = self._require_data()
        if self._pseudo is None or by not in data.frame.columns:
            return pd.DataFrame(columns=["segment", "n_treated", "uplift_units", "uplift_pct"])

        nuisance = self._nuisance
        if nuisance is None:
            # An explicit check rather than `assert`: asserts are stripped under
            # `python -O`, so the narrowing would vanish in exactly the
            # deployment where an unfitted estimator fails most obscurely.
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        treated = data.t
        frame = pd.DataFrame(
            {
                "segment": data.frame.loc[treated, by].astype(str).to_numpy(),
                # Fitted CATE for the point estimate, raw scores for the spread:
                # the same split as `estimate_ate`, and for the same reason.
                "cate": self.estimate_cate(data.X[treated]),
                "score": self._pseudo[treated],
                "baseline": nuisance.mu0[treated],
            }
        )
        grouped = frame.groupby("segment", observed=True).agg(
            n_treated=("cate", "size"),
            uplift_units=("cate", "mean"),
            baseline=("baseline", "mean"),
            score_std=("score", "std"),
        )
        grouped["uplift_pct"] = grouped["uplift_units"] / grouped["baseline"].replace(0, np.nan)
        grouped["standard_error"] = grouped["score_std"] / np.sqrt(grouped["n_treated"])
        sparse = grouped["n_treated"] < min_treated
        grouped.loc[sparse, ["uplift_units", "uplift_pct", "standard_error"]] = np.nan
        grouped["estimable"] = ~sparse
        return grouped.drop(columns=["score_std"]).reset_index()

    def _require_data(self) -> tuple[CovariateFrame, NuisanceFit]:
        if self._data is None or self._nuisance is None:
            raise EstimationError(f"{self.name} is not fitted", method=self.name)
        return self._data, self._nuisance


def bootstrap_interval(
    estimate_fn: object,
    data: CovariateFrame,
    *,
    config: PromoUpliftConfig | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval, resampling whole series.

    Resampling **series**, not rows. Rows within a product-store are serially
    correlated - a listing running hot stays hot for weeks - so a row bootstrap
    treats one listing as hundreds of independent observations and produces
    intervals several times too narrow. Resampling the cluster is the standard
    correction and it is the difference between an interval that covers and one
    that decorates.
    """
    settings = config or get_promo_uplift_config()
    rng = np.random.default_rng(settings.uncertainty.seed)

    keys = data.frame[["product_id", "store_id"]].agg("|".join, axis=1)
    unique = keys.unique()
    index_by_key = {key: np.flatnonzero((keys == key).to_numpy()) for key in unique}

    estimates: list[float] = []
    for _ in range(settings.uncertainty.bootstrap_samples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_key[key] for key in drawn])
        resampled = CovariateFrame(
            frame=data.frame.iloc[rows].reset_index(drop=True),
            feature_names=data.feature_names,
            categorical_names=data.categorical_names,
            outcome=data.outcome,
            groups=data.groups,
        )
        try:
            estimates.append(float(estimate_fn(resampled)))  # type: ignore[operator]
        except (EstimationError, ValueError):
            # A resample that lost an entire arm cannot produce an estimate.
            # Skipped rather than substituted, and the count of usable draws is
            # returned so a thin bootstrap is visible.
            continue

    if len(estimates) < 20:
        raise EstimationError(
            f"only {len(estimates)} bootstrap resamples produced an estimate; "
            f"the interval would be unreliable",
            method="bootstrap",
        )

    alpha = settings.uncertainty.alpha
    values = np.array(estimates)
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
        float(len(estimates)),
    )


__all__ = [
    "AIPWEstimator",
    "DRLearner",
    "EffectEstimate",
    "IPWEstimator",
    "NuisanceFit",
    "UpliftEstimator",
    "assign_folds",
    "bootstrap_interval",
    "fit_nuisances",
]
