"""Own-price elasticity: four estimators, one known answer.

Elasticity is the percentage change in demand per percentage change in price.
The estimation is trivial if prices move at random. They do not, and the way
they move is what breaks the naive answer.

**The problem.** `data/generation/generators/pricing_generator.py` raises prices
into anticipated strong demand and discounts into weak demand. So a regression of
log quantity on log price partly recovers the *pricing manager's* behaviour
rather than the *shopper's*, and the bias is toward zero: products look less
price-sensitive than they are, which encourages exactly the wrong recommendation.

Four estimators, ordered by what they assume:

``naive_ols``
    ``log q ~ log p``. Assumes price is exogenous. It is not, so this is
    expected to be attenuated - and it is kept precisely to show by how much.

``panel_fe``
    Adds product and time fixed effects, absorbing anything constant within a
    listing and anything common to a date. Removes the part of the endogeneity
    that lives in persistent listing quality and market-wide movements; leaves
    the part driven by *listing-specific demand shocks the manager saw and we
    did not*.

``iv_2sls``
    Instruments price with the category commodity cost index. Costs shift price
    through pass-through but do not enter demand directly - the textbook
    exclusion restriction, and here it holds by construction because the
    generator never puts cost into the demand equation.

``randomised``
    Restricts to price changes tagged ``randomised_test``, which the generator
    makes exogenous. The cleanest identification available and the smallest
    sample. Effectively the experiment you would run if you could.

All four are scored against ``ground_truth/elasticity.json``. The interesting
output is not which one wins - it is the *ordering*, because a method that
recovers truth on data where truth is known is the method to trust where it is
not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from app.observability.logging import get_logger

logger = get_logger(__name__)

#: Minimum distinct price points a listing needs before its elasticity means
#: anything. Below this the regression is fitting a handful of price levels and
#: the standard error will not say so loudly enough.
MIN_PRICE_POINTS = 5

#: Minimum rows for an estimate to be attempted at all.
MIN_ROWS = 60


@dataclass
class ElasticityEstimate:
    """One estimator's answer for one product."""

    method: str
    elasticity: float
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    r_squared: float | None = None
    n_obs: int = 0
    n_price_points: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_elastic(self) -> bool:
        """|e| > 1 - a price rise reduces revenue.

        This is the flag a pricing decision actually turns on, which is why it
        is a property rather than something the caller recomputes.
        """
        return abs(self.elasticity) > 1.0

    def summary(self) -> str:
        band = (
            f" [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}]"
            if self.confidence_interval
            else ""
        )
        return f"{self.method}: {self.elasticity:+.3f}{band} on {self.n_obs:,} rows"


def prepare_panel(
    sales: pd.DataFrame,
    *,
    costs: pd.DataFrame | None = None,
    drop_promotions: bool = True,
) -> pd.DataFrame:
    """Build the estimation frame: log quantity, log price, and the instrument.

    ``drop_promotions`` defaults True and matters more than it looks. A
    promotion moves price *and* applies a mechanic lift at the same time, so a
    promoted row attributes the mechanic's effect to the price cut and
    exaggerates elasticity. Step 7 measured that mechanic at +17.7% against a
    price channel of +45.6% - large enough to distort the coefficient badly.

    Zero-unit rows are dropped rather than offset. ``log(0)`` is undefined, and
    the usual ``log(q + 1)`` fix quietly changes the quantity being estimated
    from an elasticity into something with no clean interpretation.
    """
    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    price_column = "selling_price" if "selling_price" in frame.columns else "regular_price"
    frame = frame[(frame["units"] > 0) & (frame[price_column] > 0)]

    if drop_promotions and "promotion_flag" in frame.columns:
        frame = frame[~frame["promotion_flag"].astype(bool)]

    if "stockout_flag" in frame.columns:
        # A censored row records what was available, not what was wanted. Left
        # in, it looks like weak demand at whatever price happened to be set.
        frame = frame[~frame["stockout_flag"].astype(bool)]

    frame = frame.assign(
        log_units=np.log(frame["units"].to_numpy(dtype=float)),
        log_price=np.log(frame[price_column].to_numpy(dtype=float)),
    )

    if costs is not None and not costs.empty and "category" in frame.columns:
        instrument = costs.copy()
        instrument["date"] = pd.to_datetime(instrument["date"])
        frame = frame.merge(instrument, on=["date", "category"], how="left")
        frame["log_cost"] = np.log(frame["cost_index"].clip(lower=1e-6))

    return frame.reset_index(drop=True)


def _ols(
    y: np.ndarray, X: np.ndarray, *, cluster: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with robust standard errors, clustered where possible.

    Heteroskedasticity-robust rather than classical because the residual
    variance plainly is not constant - a hero SKU and a slow mover have very
    different noise.

    **Clustered on the listing when ``cluster`` is supplied**, and that is not
    optional refinement. Rows within a product-store are strongly serially
    correlated: a listing running hot stays hot for weeks, so the effective
    number of independent observations is closer to the number of *listings*
    than the number of rows. The unclustered interval on the test panel was
    narrow enough to exclude the known elasticity by 0.013 while the point
    estimate was within 0.02 of it - the estimate was fine and the interval was
    lying. Step 7 hit the identical failure on the uplift influence function.

    The meat matrix is built by summing ``X'e`` within each cluster rather than
    per row, which is the standard sandwich form.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)

    if cluster is None:
        meat = X.T @ np.diag(residuals**2) @ X
    else:
        codes = pd.factorize(cluster)[0]
        n_clusters = int(codes.max()) + 1 if len(codes) else 0
        if n_clusters < 2:
            meat = X.T @ np.diag(residuals**2) @ X
        else:
            scores = X * residuals[:, None]
            # Sum the scores within each cluster, then take the outer product of
            # those sums. This is what allows arbitrary correlation inside a
            # listing while assuming independence between them.
            grouped = np.zeros((n_clusters, X.shape[1]))
            np.add.at(grouped, codes, scores)
            meat = grouped.T @ grouped
            meat *= n_clusters / max(n_clusters - 1, 1)

    covariance = xtx_inv @ meat @ xtx_inv
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else 0.0
    return beta, np.sqrt(np.clip(np.diag(covariance), 0.0, None)), r_squared


def _listing_cluster(frame: pd.DataFrame) -> np.ndarray | None:
    """Cluster key: the product-store listing."""
    if "product_id" not in frame.columns or "store_id" not in frame.columns:
        return None
    return (
        frame["product_id"].astype(str) + "|" + frame["store_id"].astype(str)
    ).to_numpy()


def _finish(
    method: str,
    beta: np.ndarray,
    errors: np.ndarray,
    r_squared: float,
    frame: pd.DataFrame,
    *,
    coefficient_index: int = 1,
    diagnostics: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> ElasticityEstimate:
    """Assemble an estimate from a fitted regression."""
    elasticity = float(beta[coefficient_index])
    se = float(errors[coefficient_index])
    dof = max(len(frame) - len(beta), 1)
    critical = float(stats.t.ppf(0.975, dof))

    return ElasticityEstimate(
        method=method,
        elasticity=elasticity,
        standard_error=se,
        confidence_interval=(elasticity - critical * se, elasticity + critical * se),
        p_value=float(2 * (1 - stats.t.cdf(abs(elasticity / se), dof))) if se > 0 else None,
        r_squared=r_squared,
        n_obs=len(frame),
        n_price_points=int(frame["log_price"].round(4).nunique()),
        diagnostics=diagnostics or {},
        warnings=warnings or [],
    )


def naive_ols(frame: pd.DataFrame) -> ElasticityEstimate:
    """``log q ~ log p``. Kept to demonstrate the bias, not to be believed.

    Expected to be attenuated toward zero, because price responds to demand
    shocks the model cannot see.
    """
    y = frame["log_units"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(frame)), frame["log_price"].to_numpy(dtype=float)])
    beta, errors, r2 = _ols(y, X, cluster=_listing_cluster(frame))
    return _finish(
        "naive_ols",
        beta,
        errors,
        r2,
        frame,
        warnings=[
            "Assumes price is exogenous. It is not: prices are raised into "
            "anticipated strong demand, so this estimate is biased toward zero "
            "and understates price sensitivity."
        ],
    )


def panel_fixed_effects(frame: pd.DataFrame) -> ElasticityEstimate:
    """Product and time fixed effects, absorbed by within-transformation.

    Demeaning within each group rather than building dummy columns: with
    hundreds of listings and a thousand dates the dummy design matrix is enormous
    and almost entirely zeros. The within estimator gives identical coefficients
    at a fraction of the memory.
    """
    working = frame.copy()
    if "store_id" in working.columns and "product_id" in working.columns:
        working["_unit"] = (
            working["product_id"].astype(str) + "|" + working["store_id"].astype(str)
        )
    else:
        working["_unit"] = working.get("product_id", pd.Series("all", index=working.index))

    y = working["log_units"].to_numpy(dtype=float)
    x = working["log_price"].to_numpy(dtype=float)

    def demean(values: np.ndarray, groups: pd.Series) -> np.ndarray:
        series = pd.Series(values, index=working.index)
        return (series - series.groupby(groups.to_numpy()).transform("mean")).to_numpy()

    # Sequential demeaning by unit then by date. Exact for a balanced panel and a
    # very close approximation otherwise - and the alternative, iterating to
    # convergence, buys a third decimal place nobody acts on.
    y_within = demean(demean(y, working["_unit"]), working["date"])
    x_within = demean(demean(x, working["_unit"]), working["date"])

    X = np.column_stack([np.zeros(len(working)), x_within])
    beta, errors, r2 = _ols(y_within, X, cluster=working["_unit"].to_numpy())

    n_units = int(working["_unit"].nunique())
    n_dates = int(working["date"].nunique())
    return _finish(
        "panel_fe",
        beta,
        errors,
        r2,
        working,
        diagnostics={"n_units": float(n_units), "n_periods": float(n_dates)},
        warnings=[
            "Absorbs anything constant within a listing and anything common to "
            "a date. Does NOT absorb listing-specific demand shocks the pricing "
            "manager observed and this model did not - the residual endogeneity."
        ],
    )


def iv_two_stage(frame: pd.DataFrame) -> ElasticityEstimate:
    """2SLS instrumenting price with the category commodity cost index.

    The exclusion restriction - cost shifts price but does not enter demand
    directly - holds **by construction** here: `sales_generator.py` never puts
    cost into the demand equation. That is what makes this dataset able to show
    the method working rather than merely applied.

    **Measured outcome on this dataset: it does not work, and the reason is
    worth more than the method.** Across 25 products 2SLS recovered the true
    elasticity with a mean absolute error of 1.63 and a correlation of 0.25,
    against panel fixed effects at 0.08 and 0.99 - and it produced positive
    (wrong-signed) elasticities for some products.

    The first-stage F statistics were enormous: median 484, maximum 10,038. So
    this is not weak-instrument bias. **A strong first stage does not make an
    instrument valid.** The commodity cost index varies only at category x date,
    which means within any one product it is a pure time series - and time is
    exactly what has to be controlled for, because seasonality drives demand
    directly. The exclusion restriction fails not because cost enters demand,
    but because the only variation the instrument has is variation that other
    things also have.

    :func:`instrument_diagnostics` checks for this directly rather than relying
    on the F statistic, which is the number that looked reassuring while the
    estimate was wrong.
    """
    if "log_cost" not in frame.columns:
        raise ValueError(
            "no cost index on the frame; pass costs= to prepare_panel to enable 2SLS"
        )

    working = frame.dropna(subset=["log_cost"])
    if len(working) < MIN_ROWS:
        raise ValueError(f"only {len(working)} rows with a cost index; need {MIN_ROWS}")

    y = working["log_units"].to_numpy(dtype=float)
    price = working["log_price"].to_numpy(dtype=float)
    cost = working["log_cost"].to_numpy(dtype=float)
    constant = np.ones(len(working))

    # First stage: price on the instrument.
    first_stage_X = np.column_stack([constant, cost])
    gamma, gamma_se, first_r2 = _ols(price, first_stage_X)
    fitted_price = first_stage_X @ gamma

    f_statistic = float((gamma[1] / gamma_se[1]) ** 2) if gamma_se[1] > 0 else 0.0

    # Second stage: demand on the fitted price.
    second_stage_X = np.column_stack([constant, fitted_price])
    beta, errors, r2 = _ols(y, second_stage_X, cluster=_listing_cluster(working))

    warnings: list[str] = [
        "Identification rests on the exclusion restriction: input costs shift "
        "price but do not affect demand directly."
    ]
    if f_statistic < 10.0:
        warnings.append(
            f"WEAK INSTRUMENT: first-stage F is {f_statistic:.1f}, below the "
            f"conventional threshold of 10. A weak instrument biases 2SLS toward "
            f"OLS while reporting standard errors that do not reflect it, so "
            f"this estimate is less trustworthy than the naive one it replaces."
        )
    warnings.extend(instrument_diagnostics(working))

    return _finish(
        "iv_2sls",
        beta,
        errors,
        r2,
        working,
        diagnostics={
            "first_stage_f": f_statistic,
            "first_stage_r2": first_r2,
            "cost_passthrough": float(gamma[1]),
        },
        warnings=warnings,
    )


def instrument_diagnostics(frame: pd.DataFrame) -> list[str]:
    """Whether the instrument has usable variation at the estimation grain.

    This exists because the first-stage F statistic did not catch the failure.
    On this dataset F ran to a median of 484 while 2SLS returned elasticities
    with a correlation of 0.25 to truth and occasionally the wrong sign.

    The check that *does* catch it: if the instrument takes one value per date
    across the whole estimation frame, it is a pure time series. Its only
    variation is temporal, and temporal variation is what seasonality and trend
    also have - so the second stage cannot separate the price effect from the
    calendar. An instrument must vary *within* the dimension you are willing to
    control for, and this one does not.
    """
    warnings: list[str] = []
    if "log_cost" not in frame.columns:
        return warnings

    per_date = frame.groupby("date")["log_cost"].nunique()
    if len(per_date) and int(per_date.max()) <= 1:
        warnings.append(
            "INSTRUMENT HAS NO CROSS-SECTIONAL VARIATION: the cost index takes a "
            "single value per date across this frame, so it is a pure time "
            "series. Any time-varying demand driver - seasonality, trend, "
            "holidays - shares that variation, and the second stage cannot "
            "separate them. Measured on this dataset, 2SLS recovered truth at "
            "correlation 0.25 against panel fixed effects at 0.99. Prefer "
            "panel_fe. A strong first stage does not make an instrument valid."
        )

    correlation = float(frame["log_cost"].corr(frame["log_price"]))
    if abs(correlation) < 0.05:
        warnings.append(
            f"instrument-price correlation is {correlation:+.3f}; the first "
            f"stage has almost nothing to work with"
        )
    return warnings


def randomised_subset(frame: pd.DataFrame) -> ElasticityEstimate:
    """OLS restricted to exogenous price changes.

    The generator tags a configurable fraction of price moves as
    ``randomised_test``. On that subset price is exogenous by construction, so
    even the naive estimator is unbiased. This is the closest thing to an
    experiment the data contains and the natural benchmark for the other three.

    The cost is sample size, which is why the interval is wide and why this is a
    cross-check rather than the headline.
    """
    if "price_change_reason" not in frame.columns:
        raise ValueError("no price_change_reason column; cannot isolate randomised tests")

    # Forward-filled within a listing: the tag marks the day the price *changed*,
    # and the exogenous price then holds until the next change. Using only the
    # change days would discard almost every observation of the price it set.
    working = frame.sort_values(["product_id", "store_id", "date"]).copy()
    is_test = working["price_change_reason"].astype(str) == "randomised_test"
    working["_regime"] = (
        is_test.groupby(
            [working["product_id"], working["store_id"]], observed=True
        ).cumsum()
    )
    working["_exogenous"] = (
        is_test.groupby([working["product_id"], working["store_id"]], observed=True)
        .transform("cummax")
        .astype(bool)
    )
    working = working[working["_exogenous"] & (working["_regime"] > 0)]

    if len(working) < MIN_ROWS:
        raise ValueError(
            f"only {len(working)} rows follow a randomised price test; need {MIN_ROWS}"
        )

    y = working["log_units"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(working)), working["log_price"].to_numpy(dtype=float)])
    beta, errors, r2 = _ols(y, X, cluster=_listing_cluster(working))

    return _finish(
        "randomised",
        beta,
        errors,
        r2,
        working,
        diagnostics={"share_of_panel": len(working) / max(len(frame), 1)},
        warnings=[
            "Restricted to price regimes set by a randomised test, where price "
            "is exogenous by construction. Cleanest identification available, "
            "and the smallest sample - the interval is correspondingly wide."
        ],
    )


def estimate_all(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...] = ("naive_ols", "panel_fe", "iv_2sls", "randomised"),
) -> dict[str, ElasticityEstimate]:
    """Run every applicable estimator, recording failures rather than raising.

    One method being inapplicable - no cost index joined, too few randomised
    price changes - is information, not a fatal error. The comparison is the
    deliverable, and a missing row in it should say why.
    """
    runners = {
        "naive_ols": naive_ols,
        "panel_fe": panel_fixed_effects,
        "iv_2sls": iv_two_stage,
        "randomised": randomised_subset,
    }

    results: dict[str, ElasticityEstimate] = {}
    for name in methods:
        if name not in runners:
            continue
        try:
            results[name] = runners[name](frame)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.info("elasticity.method_unavailable", method=name, error=str(exc))
    return results


#: Estimator preference, best first. Ordered by *measured* recovery against
#: ``ground_truth/elasticity.json`` across 25 products, not by theoretical
#: appeal - which is why 2SLS, the textbook answer, ranks last here.
#:
#:     panel_fe     MAE 0.076   ratio 1.046   corr 0.99
#:     naive_ols    MAE 0.796   ratio 0.557   corr 0.31
#:     randomised   MAE 0.829   ratio 0.773   corr 0.10
#:     iv_2sls      MAE 1.631   ratio 0.808   corr 0.25
METHOD_PREFERENCE: tuple[str, ...] = ("panel_fe", "randomised", "naive_ols", "iv_2sls")

#: Methods that must never be selected automatically. ``naive_ols`` is kept for
#: the comparison table - it demonstrates the attenuation - but selecting it
#: would ship an estimate known to recover only ~56% of the true elasticity.
NOT_SELECTABLE: frozenset[str] = frozenset({"naive_ols", "iv_2sls"})


def select_estimate(
    estimates: dict[str, ElasticityEstimate],
) -> tuple[ElasticityEstimate | None, str]:
    """Choose the estimate to report, and say why.

    Preference is by measured recovery, not by theory. A selection without a
    stated rationale is indistinguishable from picking the number someone liked.
    """
    for method in METHOD_PREFERENCE:
        if method in estimates and method not in NOT_SELECTABLE:
            excluded = sorted(set(estimates) - {method})
            reason = (
                f"{method} selected: highest measured recovery against known "
                f"elasticities among the methods that ran"
            )
            if excluded:
                reason += f". Also computed: {', '.join(excluded)}"
            return estimates[method], reason

    if estimates:
        return None, (
            "only estimators known to be biased on this data produced a result "
            f"({', '.join(sorted(estimates))}); no estimate is reported"
        )
    return None, "no estimator produced a result"


def comparison_table(
    estimates: dict[str, ElasticityEstimate],
    *,
    truth: float | None = None,
) -> pd.DataFrame:
    """Side-by-side comparison, optionally scored against a known elasticity."""
    rows = []
    for name, estimate in estimates.items():
        row: dict[str, object] = {
            "method": name,
            "elasticity": estimate.elasticity,
            "std_error": estimate.standard_error,
            "ci_lower": estimate.confidence_interval[0]
            if estimate.confidence_interval
            else None,
            "ci_upper": estimate.confidence_interval[1]
            if estimate.confidence_interval
            else None,
            "n_obs": estimate.n_obs,
            "is_elastic": estimate.is_elastic,
            "selectable": name not in NOT_SELECTABLE,
        }
        if truth is not None:
            row["truth"] = truth
            row["error"] = estimate.elasticity - truth
            row["covers_truth"] = bool(
                estimate.confidence_interval
                and estimate.confidence_interval[0] <= truth <= estimate.confidence_interval[1]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def check_identification(frame: pd.DataFrame) -> list[str]:
    """Warnings about whether this slice can identify an elasticity at all."""
    warnings: list[str] = []

    if len(frame) < MIN_ROWS:
        warnings.append(f"only {len(frame)} usable rows; estimates will be unstable")

    price_points = int(frame["log_price"].round(4).nunique())
    if price_points < MIN_PRICE_POINTS:
        warnings.append(
            f"only {price_points} distinct price points. Elasticity is identified "
            f"by price *variation*; with this few levels the estimate is fitting "
            f"a handful of points and the interval will not say so loudly enough"
        )

    spread = float(frame["log_price"].std())
    if spread < 0.02:
        warnings.append(
            f"log-price standard deviation is {spread:.4f} - almost no price "
            f"variation, so the coefficient is near-unidentified"
        )
    return warnings


def estimation_window(frame: pd.DataFrame) -> tuple[date, date]:
    dates = pd.to_datetime(frame["date"])
    return dates.min().date(), dates.max().date()


__all__ = [
    "METHOD_PREFERENCE",
    "MIN_PRICE_POINTS",
    "MIN_ROWS",
    "NOT_SELECTABLE",
    "ElasticityEstimate",
    "check_identification",
    "comparison_table",
    "estimate_all",
    "estimation_window",
    "instrument_diagnostics",
    "iv_two_stage",
    "naive_ols",
    "panel_fixed_effects",
    "prepare_panel",
    "randomised_subset",
    "select_estimate",
]
