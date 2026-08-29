"""End-to-end promo uplift estimation (brief sections 26, 29).

Orchestration only - every decision lives in the module that owns it. The
sequence matters, though, and it is the sequence the brief's section 29 asks for:

.. code-block:: text

    quality checks        does the data support a causal question at all?
    treatment frame       which rows are treated, control, washout, excluded
    control pool          which controls are comparable enough to use
    covariates            strictly pre-treatment, anchored at the event start
    nuisance models       cross-fitted mu0, mu1, e
    overlap + balance     is a comparison identified on these units?
    estimators            naive, baseline, DiD, IPW, AIPW, DR-learner
    placebo + sensitivity does the method find effects that are not there?
    business impact       units, revenue, profit, ROI
    selection             which estimate survives, and why

**Diagnostics run before the estimate is trusted, not after it is published.**
The placebo test in particular can invalidate the whole run, and discovering
that after a number has been quoted is worse than not computing it.

**Every estimator runs, including the ones expected to fail.** The naive
comparison and DiD are in the table precisely so a reader can see the size of
the error each avoids. A pipeline that ran only the method it intended to use
would produce the same headline number with none of the argument behind it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.promo_uplift.baseline import BaselineCounterfactual, NaiveEstimator
from ml.promo_uplift.business import BusinessImpact, business_impact, event_level_impact
from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.controls import ControlPool, build_control_pool
from ml.promo_uplift.diagnostics import (
    GroundTruthComparison,
    PlaceboResult,
    SensitivityResult,
    SensitivityRow,
    ValidationVerdict,
    evaluate_placebo,
    judge,
    placebo_frame,
    validate_against_ground_truth,
)
from ml.promo_uplift.did import DifferenceInDifferences, ParallelTrendsTest
from ml.promo_uplift.estimators import (
    AIPWEstimator,
    DRLearner,
    EffectEstimate,
    IPWEstimator,
    NuisanceFit,
    fit_nuisances,
)
from ml.promo_uplift.evaluate import (
    MethodRow,
    comparison_table,
    format_comparison,
    segment_summary,
    select_method,
)
from ml.promo_uplift.exceptions import EstimationError, PromoUpliftError
from ml.promo_uplift.features import CovariateFrame, build_covariates
from ml.promo_uplift.matching import BalanceReport, balance_table
from ml.promo_uplift.propensity import OverlapReport, assess_overlap, att_weights
from ml.promo_uplift.quality import QualityReport, check_panel
from ml.promo_uplift.treatment import AnalysisFrame, RowRole, build_analysis_frame

logger = get_logger(__name__)


@dataclass
class UpliftRun:
    """Everything one estimation run produced."""

    config: PromoUpliftConfig
    quality: QualityReport
    analysis: AnalysisFrame
    pool: ControlPool
    covariates: CovariateFrame
    nuisance: NuisanceFit
    overlap: OverlapReport
    balance: BalanceReport

    estimates: dict[str, EffectEstimate] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    impacts: dict[str, BusinessImpact] = field(default_factory=dict)
    verdicts: dict[str, ValidationVerdict] = field(default_factory=dict)

    placebo: PlaceboResult | None = None
    sensitivity: SensitivityResult | None = None
    parallel_trends: ParallelTrendsTest | None = None
    ground_truth: GroundTruthComparison | None = None

    selected: str | None = None
    selection_reason: str = ""
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    event_impact: pd.DataFrame | None = None
    cate: np.ndarray | None = None
    #: Retained so segment effects can be recomputed on a different column
    #: without refitting the nuisance models.
    learner: DRLearner | None = None
    elapsed_seconds: float = 0.0

    @property
    def headline(self) -> EffectEstimate | None:
        return self.estimates.get(self.selected) if self.selected else None

    @property
    def headline_impact(self) -> BusinessImpact | None:
        return self.impacts.get(self.selected) if self.selected else None

    @property
    def validation_status(self) -> str:
        if self.selected is None:
            return "failed"
        verdict = self.verdicts.get(self.selected)
        return verdict.status if verdict else "not_assessed"

    def comparison(self) -> pd.DataFrame:
        rows = [
            MethodRow(
                method=name,
                estimate=estimate,
                verdict=self.verdicts.get(name),
                incremental_units=self.impacts[name].incremental_units
                if name in self.impacts
                else None,
                incremental_profit=self.impacts[name].incremental_profit
                if name in self.impacts
                else None,
                roi=self.impacts[name].roi if name in self.impacts else None,
            )
            for name, estimate in self.estimates.items()
        ]
        return comparison_table(rows)

    def warnings(self) -> list[str]:
        """Everything a caller must be told, deduplicated and ordered."""
        collected: list[str] = []
        collected.extend(self.quality.messages())
        collected.extend(self.pool.warnings)
        if self.selected and self.selected in self.verdicts:
            verdict = self.verdicts[self.selected]
            collected.extend(verdict.blocking)
            collected.extend(verdict.warnings)
        if self.headline_impact and self.headline_impact.warnings:
            collected.extend(self.headline_impact.warnings)
        # dict preserves insertion order and deduplicates in one pass, without
        # the `seen.add(...)` side effect inside a comprehension condition.
        return list(dict.fromkeys(collected))


def run_uplift(
    panel: pd.DataFrame,
    *,
    config: PromoUpliftConfig | None = None,
    baseline_units: pd.Series | None = None,
    ground_truth_dir: Path | None = None,
    run_placebo: bool = True,
    run_sensitivity: bool = True,
    segment_by: tuple[str, ...] = ("category", "region", "promotion_type"),
) -> UpliftRun:
    """Estimate promotional uplift end to end."""
    settings = config or get_promo_uplift_config()
    started = time.perf_counter()

    quality = check_panel(panel, config=settings)
    analysis = build_analysis_frame(panel, config=settings)

    if settings.stockouts.exclude_censored_rows and "stockout_flag" in analysis.frame.columns:
        analysis = _drop_censored(analysis)

    pool = build_control_pool(analysis, config=settings)
    covariates = build_covariates(
        pool.frame, analysis.events, config=settings, history=analysis.frame
    )
    nuisance = fit_nuisances(covariates, config=settings)

    _, overlap = assess_overlap(nuisance.propensity, covariates.t, config=settings)
    weights = att_weights(
        nuisance.propensity,
        covariates.t,
        stabilise_at=settings.propensity.stabilise_weights_at,
    )
    balance = balance_table(
        covariates.X,
        covariates.t,
        weights=weights,
        numeric=covariates.numeric_names(),
        categorical=covariates.categorical_names,
        config=settings,
    )

    run = UpliftRun(
        config=settings,
        quality=quality,
        analysis=analysis,
        pool=pool,
        covariates=covariates,
        nuisance=nuisance,
        overlap=overlap,
        balance=balance,
    )

    _run_estimators(run, baseline_units=baseline_units)

    if not run.estimates:
        raise EstimationError(
            "no estimator produced a result; see the failures for why",
            method="pipeline",
            failures=run.failures,
        )

    if run_placebo:
        run.placebo = _run_placebo(run)
    if run_sensitivity:
        run.sensitivity = _run_sensitivity(run)

    for name, estimate in run.estimates.items():
        run.impacts[name] = business_impact(estimate, analysis, config=settings)
        run.verdicts[name] = judge(
            estimate=estimate,
            balance=balance if name != "naive_during_vs_before" else None,
            overlap_warnings=overlap.warnings,
            placebo=run.placebo,
            sensitivity=run.sensitivity,
            config=settings,
        )

    rows = [
        MethodRow(method=name, estimate=estimate, verdict=run.verdicts.get(name))
        for name, estimate in run.estimates.items()
    ]
    best, reason = select_method(rows)
    run.selected = best.method if best else None
    run.selection_reason = reason

    _run_segments(run, segment_by=segment_by)

    if ground_truth_dir is not None and run.headline is not None:
        run.ground_truth = validate_against_ground_truth(
            run.headline, analysis.events, ground_truth_dir
        )

    run.elapsed_seconds = time.perf_counter() - started
    logger.info(
        "promo_uplift.run_complete",
        selected=run.selected,
        status=run.validation_status,
        seconds=round(run.elapsed_seconds, 1),
    )
    return run


def _drop_censored(analysis: AnalysisFrame) -> AnalysisFrame:
    """Remove stockout rows, recording how much each arm lost.

    The counts are kept because they are the evidence for whether the exclusion
    was selective. A gap between the arms means promotions caused the censoring,
    which makes this a post-treatment filter and biases the estimate downward.
    """
    frame = analysis.frame
    censored = frame["stockout_flag"].astype(bool)
    treated = frame["role"] == RowRole.TREATED

    treated_share = float(censored[treated].mean()) if treated.any() else 0.0
    control_share = float(censored[~treated].mean()) if (~treated).any() else 0.0

    kept = frame[~censored].copy()
    warnings = list(analysis.warnings)
    gap = treated_share - control_share
    if abs(gap) > 0.01:
        warnings.append(
            f"stockout censoring is differential: {treated_share:.1%} of treated "
            f"rows were excluded against {control_share:.1%} of control rows. "
            f"Stockouts are a consequence of the promotion, so dropping them "
            f"removes the highest-demand promotion days and understates uplift"
        )

    excluded = dict(analysis.excluded)
    excluded.update(
        {
            "censored_total": int(censored.sum()),
            "censored_treated": int((censored & treated).sum()),
            "censored_control": int((censored & ~treated).sum()),
        }
    )
    return AnalysisFrame(
        frame=kept.reset_index(drop=True),
        events=analysis.events,
        excluded=excluded,
        warnings=warnings,
    )


def _run_estimators(run: UpliftRun, *, baseline_units: pd.Series | None) -> None:
    """Fit every enabled estimator, recording failures rather than raising.

    One estimator failing is information, not a fatal error. DiD with an empty
    cell or the baseline counterfactual without a trained artifact should not
    stop AIPW from producing a number - and the reason each was unavailable
    belongs in the report.
    """
    settings = run.config
    enabled = set(settings.estimators.enabled())

    def attempt(name: str, fn: object) -> None:
        if name not in enabled:
            return
        try:
            run.estimates[name] = fn()  # type: ignore[operator]
        except PromoUpliftError as exc:
            run.failures[name] = str(exc)
            logger.info("promo_uplift.estimator_failed", method=name, error=str(exc))
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            run.failures[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("promo_uplift.estimator_error", method=name, error=str(exc))

    attempt(
        "naive_during_vs_before",
        lambda: NaiveEstimator().estimate(run.analysis, config=settings),
    )
    attempt(
        "baseline_counterfactual",
        lambda: BaselineCounterfactual().estimate(
            run.analysis, baseline_units, config=settings
        ),
    )

    did = DifferenceInDifferences(config=settings)
    attempt("difference_in_differences", lambda: did.estimate(run.analysis))
    run.parallel_trends = did.parallel_trends

    attempt(
        "inverse_probability_weighting",
        lambda: IPWEstimator(config=settings).fit(run.covariates, run.nuisance).estimate_ate(),
    )
    attempt(
        "augmented_ipw",
        lambda: AIPWEstimator(config=settings).fit(run.covariates, run.nuisance).estimate_ate(),
    )

    if "dr_learner" in enabled:
        try:
            learner = DRLearner(config=settings).fit(run.covariates, run.nuisance)
            run.estimates["dr_learner"] = learner.estimate_ate()
            run.cate = learner.estimate_cate(run.covariates.X)
            run.learner = learner
        except (PromoUpliftError, ValueError) as exc:
            run.failures["dr_learner"] = str(exc)


def _run_placebo(run: UpliftRun) -> PlaceboResult | None:
    """Estimate an effect where none exists, using the shipped pipeline.

    Deliberately re-runs the full path - treatment frame, control pool,
    covariates, nuisances, AIPW - rather than reusing anything fitted above. A
    placebo that shared a fitted model with the real run would test less than it
    appears to.
    """
    try:
        shifted = placebo_frame(run.analysis, config=run.config)
        pool = build_control_pool(shifted, config=run.config)
        covariates = build_covariates(
            pool.frame, shifted.events, config=run.config, history=shifted.frame
        )
        nuisance = fit_nuisances(covariates, config=run.config)
        estimate = (
            AIPWEstimator(config=run.config).fit(covariates, nuisance).estimate_ate()
        )
    except (PromoUpliftError, ValueError, KeyError) as exc:
        logger.info("promo_uplift.placebo_unavailable", error=str(exc))
        return None

    return evaluate_placebo(
        estimate, reference=run.estimates.get("augmented_ipw"), config=run.config
    )


def _run_sensitivity(run: UpliftRun) -> SensitivityResult | None:
    """Re-estimate across defensible specification choices.

    Only the choices nobody can prove correct are varied: the washout length,
    the control window, and the trimming level. Varying the estimator instead
    would measure something else - that is what the comparison table is for.
    """
    reference = run.estimates.get("augmented_ipw")
    if reference is None:
        return None

    rows: list[SensitivityRow] = []
    settings = run.config
    panel = run.analysis.frame

    def estimate_with(updated: PromoUpliftConfig) -> tuple[float, int]:
        analysis = build_analysis_frame(panel, config=updated)
        pool = build_control_pool(analysis, config=updated)
        covariates = build_covariates(
            pool.frame, analysis.events, config=updated, history=analysis.frame
        )
        nuisance = fit_nuisances(covariates, config=updated)
        estimate = AIPWEstimator(config=updated).fit(covariates, nuisance).estimate_ate()
        return estimate.ate_pct, estimate.n_treated

    for washout in settings.validation.sensitivity_washout_days:
        # The control window has to clear the washout or the config refuses, so
        # both move together for the wider settings.
        window = max(settings.controls.same_series_window_days, washout + 1)
        updated = settings.model_copy(
            update={
                "treatment": settings.treatment.model_copy(update={"washout_days": washout}),
                "controls": settings.controls.model_copy(
                    update={"same_series_window_days": window}
                ),
                "validation": settings.validation.model_copy(
                    update={"placebo_shift_days": max(washout + 1, 30)}
                ),
            }
        )
        rows.append(_sensitivity_row("washout_days", washout, updated, estimate_with))

    for window in settings.validation.sensitivity_control_windows:
        if window <= settings.treatment.washout_days:
            continue
        updated = settings.model_copy(
            update={
                "controls": settings.controls.model_copy(
                    update={"same_series_window_days": window}
                )
            }
        )
        rows.append(_sensitivity_row("control_window_days", window, updated, estimate_with))

    for trim in settings.validation.sensitivity_trim_levels:
        updated = settings.model_copy(
            update={
                "propensity": settings.propensity.model_copy(
                    update={"clip": (trim, 1.0 - trim)}
                )
            }
        )
        rows.append(_sensitivity_row("propensity_trim", trim, updated, estimate_with))

    return SensitivityResult(rows=rows, reference_pct=reference.ate_pct)


def _sensitivity_row(
    parameter: str,
    value: object,
    updated: PromoUpliftConfig,
    estimate_with: object,
) -> SensitivityRow:
    try:
        effect, n_treated = estimate_with(updated)  # type: ignore[operator]
        return SensitivityRow(parameter, value, effect, n_treated)
    except (PromoUpliftError, ValueError, KeyError) as exc:
        return SensitivityRow(parameter, value, float("nan"), 0, failed=str(exc))


def _run_segments(run: UpliftRun, *, segment_by: tuple[str, ...]) -> None:
    """Segment-level CATE, plus the per-event table Step 8 will consume."""
    learner = run.learner
    if learner is None:
        return

    for column in segment_by:
        if column not in run.covariates.frame.columns:
            continue
        try:
            segments = learner.segment_effects(column, min_treated=30)
            run.segments[column] = segment_summary(segments)
        except (ValueError, KeyError) as exc:
            logger.info("promo_uplift.segment_failed", column=column, error=str(exc))

    if run.cate is not None:
        treated = run.covariates.t
        # Aligned against the *covariate* frame's treated rows, not the analysis
        # frame's. The covariate build drops rows without a complete
        # pre-treatment history, so the two differ - and pairing each effect
        # with the wrong promotion would be worse than producing no table.
        try:
            run.event_impact = event_level_impact(
                run.cate[treated],
                run.analysis,
                treated_rows=run.covariates.frame[treated],
                config=run.config,
            )
        except ValueError as exc:
            logger.warning("promo_uplift.event_impact_unavailable", error=str(exc))


def _segment_markdown(frame: pd.DataFrame) -> str:
    """Segment table as markdown.

    Written out rather than using ``DataFrame.to_markdown``, which needs
    ``tabulate`` - an extra dependency for one table in one report.
    """
    lines = [
        "| Segment | Treated rows | Uplift | Classification | Action |",
        "|---|---|---|---|---|",
    ]
    for row in frame.to_dict("records"):
        uplift = (
            f"{float(row['uplift_pct']):+.1%}"
            if pd.notna(row["uplift_pct"])
            else "not estimable"
        )
        lines.append(
            f"| {row['segment']} | {int(row['n_treated']):,} | {uplift} | "
            f"{row['classification']} | {row['action']} |"
        )
    return "\n".join(lines)


def report(run: UpliftRun) -> str:
    """A markdown report of everything the run established."""
    lines: list[str] = ["# Promo uplift estimation", ""]

    lines.append(f"**Estimand.** {run.config.treatment_definition()}")
    lines.append("")
    lines.append(f"Config fingerprint `{run.config.fingerprint()}`, "
                 f"{run.elapsed_seconds:.1f}s.")
    lines.append("")

    headline = run.headline
    if headline is None:
        lines.append("## No defensible estimate")
        lines.append("")
        lines.append(run.selection_reason)
    else:
        interval = headline.interval_pct()
        band = f" (95% CI [{interval[0]:+.1%}, {interval[1]:+.1%}])" if interval else ""
        lines.append(f"## Headline: {headline.ate_pct:+.1%}{band}")
        lines.append("")
        lines.append(run.selection_reason)
        lines.append("")
        impact = run.headline_impact
        if impact:
            lines.append(f"- {impact.summary()}")
        lines.append(f"- validation: **{run.validation_status}**")

    lines.extend(["", "## Method comparison", "", format_comparison(run.comparison())])

    lines.extend(["", "## Diagnostics", ""])
    lines.append(f"- {run.overlap.summary()}")
    lines.append(f"- {run.balance.summary()}")
    if run.placebo:
        lines.append(f"- {run.placebo.summary()}")
    if run.sensitivity:
        lines.append(f"- {run.sensitivity.summary()}")
    if run.parallel_trends:
        lines.append(f"- {run.parallel_trends.summary()}")
    if run.ground_truth:
        lines.append(f"- {run.ground_truth.summary()}")

    if run.failures:
        lines.extend(["", "## Estimators that did not run", ""])
        lines.extend(f"- **{name}**: {reason}" for name, reason in run.failures.items())

    warnings = run.warnings()
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)

    for column, frame in run.segments.items():
        if frame.empty:
            continue
        lines.extend(["", f"## Uplift by {column}", "", _segment_markdown(frame)])

    lines.extend(["", "## Data quality", "", run.quality.render()])
    return "\n".join(lines)


__all__ = ["UpliftRun", "report", "run_uplift"]
