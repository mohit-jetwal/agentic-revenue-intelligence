"""Treatment assignment model and the overlap guardrails (brief section 12).

The propensity score is ``e(X) = P(T = 1 | X)`` - the probability this
product-store-day would have been promoted, given everything knowable before the
promotion started. Rosenbaum and Rubin's result is what makes it useful: if
treatment is ignorable given ``X``, it is also ignorable given ``e(X)`` alone. A
30-dimensional adjustment problem collapses to a one-dimensional one.

**What the model is for, and what it is not for.** A propensity model is not
trying to predict treatment well. A model with perfect discrimination is a
disaster: it means treated and control units are perfectly separable, so for
every treated unit there is no comparable control and no comparison is possible.
The right target is *balance*, not AUC, and this module reports both so the
difference is visible. An AUC near 0.5 with good balance is a better outcome than
an AUC of 0.95.

**Overlap is the assumption that actually bites.** Ignorability is untestable;
positivity is not. Where ``e(X)`` approaches 0 or 1 the inverse-probability
weight explodes - a propensity of 0.001 hands one observation a weight of 1000,
and the "estimate" becomes that row's outcome with extra arithmetic. So the
scores are trimmed, the trimmed share is reported, and past a configured
threshold the estimate is **refused** rather than returned with a wide interval.
A wide interval says "we are unsure"; the honest statement here is "this design
does not identify the effect on these units".

**Why logistic by default.** Section 34 asks what drives treatment assignment,
and a linear model in the log-odds answers that directly with a coefficient per
covariate. Gradient boosting finds interactions unaided but is opaque and tends
to produce confident, extreme scores - exactly the ones that break weighting. The
interactions that matter here are known in advance (category by season, which is
how the platform generator targets promotions), so they are constructed
explicitly rather than discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.exceptions import EstimationError, OverlapViolationError

logger = get_logger(__name__)

#: Covariates whose interaction with the categoricals is constructed by hand.
#: These carry the confounder: in the platform generator, promotion timing is
#: drawn with weights ``exp(targeting * 2 * seasonal[category])``, so the
#: relationship between date and treatment differs *by category*. A model with
#: additive season and additive category cannot represent that, and the back-door
#: path stays open no matter how well the model fits.
_INTERACTION_TERMS: tuple[str, ...] = (
    "season_sin_1",
    "season_cos_1",
    "season_sin_2",
    "season_cos_2",
)


@dataclass
class OverlapReport:
    """Whether treated and control units share covariate support."""

    n_treated: int
    n_control: int
    #: Share of rows whose score fell outside the clip range.
    trimmed_share: float
    #: Effective sample size under the weights, as a fraction of rows. Weights
    #: concentrated on a few observations carry far less information than the
    #: row count implies, and this is the number that says so.
    effective_sample_fraction: float
    min_treated_score: float
    max_control_score: float
    #: Rows whose score sits outside the region where both arms are represented.
    off_support_rows: int
    #: Discrimination. Reported for context, NOT optimised - see the module
    #: docstring for why a high value is a warning rather than a success.
    auc: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        auc = f", AUC {self.auc:.3f}" if self.auc is not None else ""
        return (
            f"overlap: {self.trimmed_share:.1%} trimmed, ESS "
            f"{self.effective_sample_fraction:.1%} of rows, "
            f"{self.off_support_rows:,} off support{auc}"
        )


@dataclass
class PropensityResult:
    """Fitted scores and the weights derived from them."""

    scores: np.ndarray
    #: ATT weights: 1 for treated, e/(1-e) for control. See :func:`att_weights`.
    weights: np.ndarray
    #: True where the row survived trimming.
    kept: np.ndarray
    overlap: OverlapReport
    #: Log-odds coefficients by covariate, when the model is linear. Empty for
    #: the boosted variant, because a gain score is not a coefficient and
    #: presenting it as one would invite a causal reading of a predictive number.
    coefficients: dict[str, float] = field(default_factory=dict)


class PropensityModel:
    """``P(T = 1 | X)`` with an explicit, inspectable design matrix."""

    def __init__(self, *, config: PromoUpliftConfig | None = None) -> None:
        self._config = config or get_promo_uplift_config()
        self._encoder: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | HistGradientBoostingClassifier | None = None
        self._numeric: tuple[str, ...] = ()
        self._categorical: tuple[str, ...] = ()
        self._design_names: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X: pd.DataFrame,
        t: np.ndarray,
        *,
        numeric: tuple[str, ...],
        categorical: tuple[str, ...],
    ) -> PropensityModel:
        """Fit the assignment model."""
        self._numeric = numeric
        self._categorical = categorical

        if len(np.unique(t)) < 2:
            raise EstimationError(
                "the treatment indicator has a single value, so no assignment "
                "model can be fitted; check the treatment definition",
                method="propensity",
            )

        design = self._build_design(X, fit=True)
        kind = self._config.propensity.model

        if kind == "logistic":
            # L2 with a moderate C: the design matrix has one-hot columns and
            # hand-built interactions, so some are near-collinear. Unregularised
            # logistic regression on that produces enormous offsetting
            # coefficients and scores pinned at 0 and 1 - which then destroys
            # overlap for reasons that have nothing to do with the data.
            # `penalty="l2"` is the default and is deprecated as an explicit
            # argument from scikit-learn 1.8, so only C is set. The
            # regularisation is still L2.
            model: LogisticRegression | HistGradientBoostingClassifier = LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs"
            )
        elif kind in {"gradient_boosting", "hist_gradient_boosting"}:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                random_state=self._config.outcome_model.seed,
            )
        else:
            raise EstimationError(
                f"unknown propensity.model {kind!r}; expected 'logistic' or "
                f"'gradient_boosting'",
                method="propensity",
            )

        model.fit(design, t.astype(int))
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Propensity scores for the given covariates."""
        if self._model is None:
            raise EstimationError("propensity model is not fitted", method="propensity")
        design = self._build_design(X, fit=False)
        return np.asarray(self._model.predict_proba(design)[:, 1], dtype=float)

    def coefficients(self) -> dict[str, float]:
        """Log-odds coefficients, largest absolute value first.

        Section 34 asks what drives treatment assignment. This answers it - and
        the docs say plainly that it describes *who gets promoted*, not what
        promotions do. Reading a propensity coefficient as an effect is a
        category error the report is written to prevent.
        """
        if not isinstance(self._model, LogisticRegression):
            return {}
        values = dict(zip(self._design_names, self._model.coef_[0], strict=True))
        return dict(sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def _build_design(self, X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Numeric block, one-hot block, and the season-by-category interactions."""
        numeric = X[list(self._numeric)].to_numpy(dtype=float)
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self._scaler = StandardScaler().fit(numeric)
        if self._scaler is None:
            # Explicit rather than `assert`: asserts are stripped under
            # `python -O`, so the narrowing would vanish in the deployment where
            # an unfitted model fails most obscurely.
            raise EstimationError("propensity model is not fitted", method="propensity")
        scaled = self._scaler.transform(numeric)

        blocks = [scaled]
        names = list(self._numeric)

        if self._categorical:
            frame = X[list(self._categorical)].astype(str)
            if fit:
                # No `drop="first"`. Combined with `handle_unknown="ignore"` it
                # is genuinely ambiguous: an unseen category encodes as all
                # zeros, which is exactly how the dropped reference level is
                # encoded - so an unknown region would be silently scored as
                # whichever region happened to be dropped. Cross-fitting makes
                # this a live risk rather than a theoretical one, since a fold
                # may not contain every category. Keeping all levels
                # reintroduces collinearity, which the L2 penalty already
                # handles.
                self._encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ).fit(frame)
            if self._encoder is None:
                raise EstimationError(
                    "propensity encoder is not fitted", method="propensity"
                )
            dummies = self._encoder.transform(frame)
            blocks.append(dummies)
            names.extend(self._encoder.get_feature_names_out(self._categorical))

            interactions, interaction_names = self._interactions(X, dummies, names)
            if interactions.size:
                blocks.append(interactions)
                names.extend(interaction_names)

        if fit:
            self._design_names = tuple(names)
        return np.hstack(blocks)

    def _interactions(
        self, X: pd.DataFrame, dummies: np.ndarray, names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Products of the seasonal harmonics with each category dummy."""
        present = [c for c in _INTERACTION_TERMS if c in X.columns]
        if not present or dummies.size == 0:
            return np.empty((len(X), 0)), []

        if self._encoder is None:
            raise EstimationError("propensity encoder is not fitted", method="propensity")
        dummy_names = list(self._encoder.get_feature_names_out(self._categorical))
        season = X[present].to_numpy(dtype=float)

        columns = []
        labels = []
        for j, dummy_name in enumerate(dummy_names):
            for k, term in enumerate(present):
                columns.append(dummies[:, j] * season[:, k])
                labels.append(f"{dummy_name}:{term}")
        return np.column_stack(columns), labels


def att_weights(
    scores: np.ndarray,
    t: np.ndarray,
    *,
    stabilise_at: float | None = None,
) -> np.ndarray:
    """Weights targeting the effect on the treated.

    Treated units get weight 1 - they are the population of interest, so they
    are already correctly represented. Control units get ``e / (1 - e)``, which
    up-weights controls that *look* treated and down-weights those that do not,
    reshaping the control group to resemble the treated one.

    ATT rather than ATE deliberately. The business question is "what did the
    promotions we ran achieve", not "what would happen if we promoted
    everything". ATT also needs overlap only on the treated support, which is a
    materially weaker requirement - there is no need for every unpromoted
    store-day to have had a plausible chance of promotion.

    **Stabilisation.** ``e/(1-e)`` diverges as ``e`` approaches 1: a control row
    scored 0.98 receives weight 49 while the 99th percentile of weights sits
    below 1. A handful of such rows then *are* the weighted control mean.
    Measured on the confounded synthetic panel, this overshot every demand
    covariate - balance went from +0.27 before weighting to -0.38 after, worse
    in the opposite direction. Capping the weights at a high percentile trades a
    little bias for a large variance reduction and is standard practice
    (Crump et al., Lee et al.). The share of weight moved is reported by
    :func:`assess_overlap` rather than absorbed silently.
    """
    weights = np.ones_like(scores, dtype=float)
    control = ~t.astype(bool)
    weights[control] = scores[control] / np.clip(1.0 - scores[control], 1e-12, None)

    if stabilise_at is not None and control.any():
        cap = float(np.percentile(weights[control], stabilise_at))
        weights[control] = np.minimum(weights[control], cap)
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ESS: ``(sum w)^2 / sum w^2``.

    The number of equally weighted observations carrying the same information.
    Ten thousand rows with an ESS of 300 is a 300-observation study, and every
    confidence interval should be read that way.
    """
    total = float(np.sum(weights))
    squared = float(np.sum(np.square(weights)))
    return total * total / squared if squared > 0 else 0.0


def assess_overlap(
    scores: np.ndarray,
    t: np.ndarray,
    *,
    config: PromoUpliftConfig | None = None,
) -> tuple[np.ndarray, OverlapReport]:
    """Trim extreme scores and judge whether a comparison is supportable.

    Returns the keep mask and the report. Does **not** raise - the caller decides
    whether a violation is fatal, because a sensitivity run legitimately wants to
    see what happens outside the guardrails.
    """
    settings = config or get_promo_uplift_config()
    rule = settings.propensity
    low, high = rule.clip

    treated = t.astype(bool)
    # Trimming means *dropping* rows, not clipping their scores to the boundary.
    # Clipping keeps an extreme observation and hands it the largest weight the
    # range allows, which is the worst of both: the row still dominates, and the
    # trimmed-share diagnostic reads zero so nobody notices.
    kept = (scores >= low) & (scores <= high)
    trimmed_share = float(1.0 - kept.mean()) if len(scores) else 0.0

    weights = att_weights(scores, treated, stabilise_at=rule.stabilise_weights_at)
    ess_fraction = (
        effective_sample_size(weights[kept]) / max(int(kept.sum()), 1) if kept.any() else 0.0
    )

    # Common support: the region where both arms actually appear. Outside it the
    # outcome model is extrapolating rather than interpolating, and no amount of
    # weighting supplies the missing data.
    treated_scores = scores[treated & kept]
    control_scores = scores[~treated & kept]
    if len(treated_scores) and len(control_scores):
        support_low = max(treated_scores.min(), control_scores.min())
        support_high = min(treated_scores.max(), control_scores.max())
        off_support = int(((scores < support_low) | (scores > support_high)).sum())
        min_treated = float(treated_scores.min())
        max_control = float(control_scores.max())
    else:
        off_support = len(scores)
        min_treated = float("nan")
        max_control = float("nan")

    warnings: list[str] = []
    if trimmed_share > rule.max_trimmed_share:
        warnings.append(
            f"{trimmed_share:.1%} of rows fall outside the propensity range "
            f"{rule.clip}, above the {rule.max_trimmed_share:.0%} limit; treated "
            f"and control units do not share enough covariate support for a "
            f"comparison to be identified"
        )
    if ess_fraction < rule.min_effective_sample_fraction:
        warnings.append(
            f"effective sample size is {ess_fraction:.1%} of rows, below the "
            f"{rule.min_effective_sample_fraction:.0%} floor; the weights are "
            f"concentrated on few observations, so the interval understates how "
            f"little information the estimate rests on"
        )

    report = OverlapReport(
        n_treated=int(treated.sum()),
        n_control=int((~treated).sum()),
        trimmed_share=trimmed_share,
        effective_sample_fraction=float(ess_fraction),
        min_treated_score=min_treated,
        max_control_score=max_control,
        off_support_rows=off_support,
        auc=_auc(scores, treated),
        warnings=warnings,
    )
    logger.info("promo_uplift.overlap_assessed", **{
        "trimmed_share": round(trimmed_share, 4),
        "ess_fraction": round(float(ess_fraction), 4),
        "off_support": off_support,
    })
    return kept, report


def fit_propensity(
    X: pd.DataFrame,
    t: np.ndarray,
    *,
    numeric: tuple[str, ...],
    categorical: tuple[str, ...],
    config: PromoUpliftConfig | None = None,
    raise_on_violation: bool = True,
) -> PropensityResult:
    """Fit, score, trim and assess in one call."""
    settings = config or get_promo_uplift_config()
    model = PropensityModel(config=settings).fit(X, t, numeric=numeric, categorical=categorical)
    scores = model.predict(X)

    kept, overlap = assess_overlap(scores, t, config=settings)
    if raise_on_violation and overlap.warnings:
        raise OverlapViolationError(
            "; ".join(overlap.warnings),
            assumption="positivity/overlap",
            diagnostic="propensity_trimming",
            observed=overlap.trimmed_share,
            threshold=settings.propensity.max_trimmed_share,
        )

    return PropensityResult(
        scores=scores,
        weights=att_weights(
            scores, t.astype(bool), stabilise_at=settings.propensity.stabilise_weights_at
        ),
        kept=kept,
        overlap=overlap,
        coefficients=model.coefficients(),
    )


def _auc(scores: np.ndarray, treated: np.ndarray) -> float | None:
    """Rank-based AUC, computed without importing a metrics module for one number."""
    if treated.all() or not treated.any():
        return None
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos = int(treated.sum())
    n_neg = int((~treated).sum())
    return float((ranks[treated].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


__all__ = [
    "OverlapReport",
    "PropensityModel",
    "PropensityResult",
    "assess_overlap",
    "att_weights",
    "effective_sample_size",
    "fit_propensity",
]
