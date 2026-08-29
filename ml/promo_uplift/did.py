"""Difference-in-differences, and the test that can reject it (section 15).

DiD compares the *change* in a treated group against the *change* in a control
group:

.. code-block:: text

    effect = (treated_post - treated_pre) - (control_post - control_pre)

Its appeal is that it differences away every time-invariant difference between
the groups. A store that simply sells more, a SKU with better distribution, a
region with more traffic - none of it matters, because it is present in both the
pre and post terms and cancels. That is a genuinely powerful property, and it is
why DiD survives in settings where no covariate set would be convincing.

**It rests entirely on parallel trends**: absent treatment, the two groups would
have moved *together*. That assumption is not implied by the data, it is not
implied by randomisation, and it is routinely asserted rather than checked.

Here it is checked, and the check can fail. In the platform generator promotion
timing is drawn with weights ``exp(targeting * 2 * seasonal)`` - promotions are
placed on rising seasonal demand. So treated listings are on an *upward* pre-trend
relative to controls before anything happens, and DiD would read the continuation
of that climb as the effect of the promotion. The brief says not to reach for DiD
automatically; this module is what "not automatically" looks like in code.

**When DiD is the right tool here**: a promotion that ran in some stores and not
others, on the same product, at the same time, where the store choice was driven
by something unrelated to the demand trajectory. That is a real and common
design. **When it is not**: a promotion timed to a demand upswing, which is most
of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from app.observability.logging import get_logger
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.exceptions import EstimationError
from ml.promo_uplift.treatment import DATE, KEYS, AnalysisFrame, RowRole

logger = get_logger(__name__)


@dataclass
class ParallelTrendsTest:
    """Whether treated and control moved together before treatment.

    Implemented as a regression of the outcome on ``time x treated`` over the
    pre-period only. Under parallel trends the interaction coefficient is zero:
    the groups have the same slope. A significant coefficient means they were
    already diverging, and DiD would attribute that divergence to the treatment.
    """

    #: Difference in pre-period slope, treated minus control, in units per day.
    slope_difference: float
    standard_error: float
    t_statistic: float
    p_value: float
    alpha: float
    n_pre_days: int
    n_treated_series: int
    n_control_series: int

    @property
    def parallel(self) -> bool:
        """Whether the test *fails to reject* parallel trends.

        Note the direction carefully. Not rejecting is weak evidence - it can
        simply mean the pre-period was too short or too noisy to detect a real
        divergence. Rejecting is strong evidence *against*. This asymmetry is
        why passing the test licenses DiD only provisionally, and failing it
        disqualifies DiD outright.
        """
        return self.p_value >= self.alpha

    def summary(self) -> str:
        verdict = "not rejected" if self.parallel else "REJECTED"
        return (
            f"parallel trends {verdict}: pre-period slope difference "
            f"{self.slope_difference:+.4f} units/day "
            f"(t={self.t_statistic:.2f}, p={self.p_value:.4f}) over "
            f"{self.n_pre_days} days"
        )


def test_parallel_trends(
    analysis: AnalysisFrame,
    *,
    config: PromoUpliftConfig | None = None,
) -> ParallelTrendsTest:
    """Compare pre-treatment slopes between the arms.

    The pre-period for a treated listing is the window before its first
    qualifying event; for a control listing it is the same calendar window, so
    both groups are measured over the same dates and a market-wide movement
    affects both.
    """
    settings = config or get_promo_uplift_config()
    panel = analysis.frame
    events = analysis.events

    if events.empty:
        raise EstimationError("no qualifying events to define a pre-period", method="did")

    first_start = pd.to_datetime(events["start_date"]).min()
    pre_start = first_start - pd.Timedelta(days=settings.controls.pre_period_days)

    pre = panel[(panel[DATE] >= pre_start) & (panel[DATE] < first_start)].copy()
    if pre.empty:
        raise EstimationError(
            f"no rows in the {settings.controls.pre_period_days}-day pre-period "
            f"before {first_start.date()}; parallel trends cannot be tested",
            method="did",
        )

    # A listing is "treated" for this test if it is ever treated, not if it is
    # treated on the row in question - every row here is pre-treatment by
    # construction, so the row-level flag would be uniformly False.
    ever_treated = set(
        map(tuple, events[list(KEYS)].drop_duplicates().to_numpy())
    )
    keys = list(map(tuple, pre[list(KEYS)].to_numpy()))
    pre["_treated_series"] = [k in ever_treated for k in keys]

    if pre["_treated_series"].nunique() < 2:
        raise EstimationError(
            "the pre-period contains only one group, so trends cannot be "
            "compared between arms",
            method="did",
        )

    time = (pre[DATE] - pre[DATE].min()).dt.days.to_numpy(dtype=float)
    treated = pre["_treated_series"].to_numpy(dtype=float)
    y = pre[settings.target].to_numpy(dtype=float)

    # y ~ 1 + time + treated + time:treated. The interaction is the quantity of
    # interest; the main effects absorb the level difference between groups and
    # any trend common to both.
    design = np.column_stack([np.ones_like(time), time, treated, time * treated])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    residuals = y - design @ coefficients
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = float(residuals @ residuals) / dof
    try:
        covariance = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError as exc:
        raise EstimationError(
            "the parallel-trends design matrix is singular; the pre-period "
            "probably contains a single date or a single series",
            method="did",
        ) from exc

    slope_difference = float(coefficients[3])
    se = float(np.sqrt(covariance[3, 3]))
    t_stat = slope_difference / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof)))

    test = ParallelTrendsTest(
        slope_difference=slope_difference,
        standard_error=se,
        t_statistic=t_stat,
        p_value=p_value,
        alpha=settings.validation.parallel_trends_alpha,
        n_pre_days=int(pre[DATE].nunique()),
        n_treated_series=int(
            pre.loc[pre["_treated_series"], list(KEYS)].drop_duplicates().shape[0]
        ),
        n_control_series=int(
            pre.loc[~pre["_treated_series"], list(KEYS)].drop_duplicates().shape[0]
        ),
    )
    logger.info("promo_uplift.parallel_trends_tested", parallel=test.parallel, p=round(p_value, 5))
    return test


class DifferenceInDifferences:
    """Two-group, two-period DiD with a parallel-trends gate."""

    name = "difference_in_differences"

    def __init__(self, *, config: PromoUpliftConfig | None = None) -> None:
        self._config = config or get_promo_uplift_config()
        self._test: ParallelTrendsTest | None = None

    @property
    def parallel_trends(self) -> ParallelTrendsTest | None:
        return self._test

    def estimate(self, analysis: AnalysisFrame) -> EffectEstimate:
        """The DiD estimate, always accompanied by its own validity test.

        The estimate is returned even when parallel trends is rejected, carrying
        a warning that says so and why. Suppressing it would hide the size of
        the error a reader would have made by reaching for DiD - which, given
        the whole point of running several estimators is to compare them, is
        exactly the information the comparison exists to produce.
        """
        settings = self._config
        panel = analysis.frame
        events = analysis.events

        self._test = test_parallel_trends(analysis, config=settings)

        first_start = pd.to_datetime(events["start_date"]).min()
        pre_start = first_start - pd.Timedelta(days=settings.controls.pre_period_days)

        ever_treated = set(map(tuple, events[list(KEYS)].drop_duplicates().to_numpy()))
        keys = list(map(tuple, panel[list(KEYS)].to_numpy()))
        treated_series = np.array([k in ever_treated for k in keys])

        pre_mask = (panel[DATE] >= pre_start) & (panel[DATE] < first_start)
        post_mask = panel["role"] == RowRole.TREATED

        # The control group's "post" is the same calendar window, so a
        # market-wide movement in that period is differenced out rather than
        # attributed to the promotion.
        post_start = panel.loc[post_mask, DATE].min()
        post_end = panel.loc[post_mask, DATE].max()
        control_post = (
            ~treated_series
            & (panel[DATE] >= post_start)
            & (panel[DATE] <= post_end)
        )

        target = settings.target
        cells = {
            "treated_pre": panel.loc[pre_mask & treated_series, target],
            "treated_post": panel.loc[post_mask, target],
            "control_pre": panel.loc[pre_mask & ~treated_series, target],
            "control_post": panel.loc[control_post, target],
        }
        empty = [name for name, values in cells.items() if values.empty]
        if empty:
            raise EstimationError(
                f"the difference-in-differences design has empty cells: "
                f"{', '.join(empty)}. Every one of the four groups must contain "
                f"observations for the estimator to be defined",
                method=self.name,
            )

        means = {name: float(values.mean()) for name, values in cells.items()}
        treated_change = means["treated_post"] - means["treated_pre"]
        control_change = means["control_post"] - means["control_pre"]
        effect = treated_change - control_change

        # Standard error from the four cell variances. Independent cells is an
        # approximation - rows within a listing are correlated - so this
        # interval is narrower than the truth. Flagged in the assumptions rather
        # than quietly presented as exact.
        variance = sum(
            float(values.var(ddof=1)) / len(values) for values in cells.values()
        )
        se = float(np.sqrt(variance))

        alpha = settings.uncertainty.alpha
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        baseline = means["treated_pre"] + control_change

        test = self._test
        warnings: list[str] = []
        if not test.parallel:
            warnings.append(
                f"PARALLEL TRENDS REJECTED (p={test.p_value:.4f}). Treated and "
                f"control series were already diverging by "
                f"{test.slope_difference:+.4f} units/day before any promotion "
                f"ran, so this estimate contains that divergence as well as any "
                f"treatment effect. Do not use it as the causal estimate"
            )
        warnings.append(
            "the interval treats the four cells as independent; rows within a "
            "listing are serially correlated, so it is narrower than the truth"
        )

        return EffectEstimate(
            method=self.name,
            ate=effect,
            ate_pct=effect / baseline if baseline > 0 else 0.0,
            baseline_units=baseline,
            n_treated=len(cells["treated_post"]),
            n_control=len(cells["control_post"]),
            standard_error=se,
            ci_lower=effect - critical * se,
            ci_upper=effect + critical * se,
            confidence_level=settings.uncertainty.confidence_level,
            assumptions=[
                "PARALLEL TRENDS: absent the promotion, treated and control "
                "series would have moved together. This is tested, not assumed "
                f"- see the diagnostic (p={test.p_value:.4f}).",
                "No spillover: the promotion did not affect the control group. "
                "Untested here, and cannibalisation between substitutes would "
                "violate it.",
                "Stable composition: the same listings are present before and "
                "after.",
            ],
            warnings=warnings,
            diagnostics={
                "treated_pre": means["treated_pre"],
                "treated_post": means["treated_post"],
                "control_pre": means["control_pre"],
                "control_post": means["control_post"],
                "treated_change": treated_change,
                "control_change": control_change,
                "parallel_trends_p": test.p_value,
                "pre_slope_difference": test.slope_difference,
            },
        )


__all__ = ["DifferenceInDifferences", "ParallelTrendsTest", "test_parallel_trends"]
