"""Command-line interface.

Run with ``ari <command>`` after ``uv sync``, or ``python -m app.cli``.

Step 1 provides introspection commands only - enough to verify the skeleton is
wired correctly without starting the server. Data generation, training and
evaluation commands are added by their respective steps.
"""

from __future__ import annotations

from typing import Any

import typer

from app.config.settings import get_settings
from app.observability.logging import configure_logging
from app.services.container import Container

app = typer.Typer(
    name="ari",
    help="Agentic Revenue Intelligence - developer CLI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def config() -> None:
    """Show effective configuration. Secrets are redacted."""
    settings = get_settings()
    typer.echo(f"environment      : {settings.app.environment.value}")
    typer.echo(f"version          : {settings.app.version}")
    typer.echo(f"project root     : {settings.project_root}")
    typer.echo("")
    typer.echo(f"parquet root     : {settings.resolve(settings.data.parquet_root)}")
    typer.echo(f"duckdb path      : {settings.resolve(settings.data.duckdb_path)}")
    typer.echo(f"app database     : {settings.data.app_database_url}")
    typer.echo("")
    typer.echo(f"llm model        : {settings.llm.model}")
    typer.echo(f"llm planner      : {settings.llm.planner_model}")
    typer.echo(f"llm key set      : {settings.llm.is_configured}")
    typer.echo("")
    typer.echo(f"mlflow tracking  : {settings.ml.tracking_uri}")
    typer.echo(f"vector backend   : {settings.vectorstore.backend.value}")
    typer.echo("")
    typer.echo(
        "agent budget     : "
        f"{settings.agent.max_iterations} iterations, "
        f"{settings.agent.max_tool_calls} tool calls, "
        f"{settings.agent.max_execution_seconds:.0f}s, "
        f"{settings.agent.max_token_budget} tokens"
    )


@app.command()
def health() -> None:
    """Probe every dependency and report status.

    Exits non-zero if a component that should be available is not, so this can
    be used as a smoke check in CI.
    """
    configure_logging()
    container = Container()
    typer.echo(f"environment: {container.environment.value}\n")

    hard_failure = False
    for name, ok, detail in container.health_checks():
        marker = "OK  " if ok else "-- "
        typer.echo(f"{marker} {name:<18} {detail}")

    typer.echo("\nComponents marked '--' are implemented in later steps; see README.")
    raise typer.Exit(code=1 if hard_failure else 0)


@app.command()
def tools() -> None:
    """List analytical tools available to agents."""
    container = Container()
    registry = container.tool_registry
    if len(registry) == 0:
        typer.echo("No tools registered yet (registered in Stage 1 Step 13).")
        return
    for name in registry.names():
        spec = registry.get(name).spec()
        typer.echo(f"{name:<28} [{spec.permission}] {spec.description.splitlines()[0]}")


@app.command("generate-data")
def generate_data(
    profile: str = typer.Option("dev", help="Dataset profile: smoke, dev or stress."),
    seed: int | None = typer.Option(None, help="Override the profile's random seed."),
    products: int | None = typer.Option(None, help="Override the product count."),
    stores: int | None = typer.Option(None, help="Override the store count."),
    customers: int | None = typer.Option(None, help="Override the customer count."),
) -> None:
    """Generate the synthetic CPG/Retail dataset."""
    from data.generation.pipeline import generate_dataset

    configure_logging()
    overrides = {
        "scale.products": products,
        "scale.stores": stores,
        "scale.customers": customers,
    }
    result = generate_dataset(profile, seed=seed, overrides=overrides)
    typer.echo("")
    typer.echo(result.summary())
    typer.echo("")
    typer.echo(f"gold content hash: {result.gold_hash[:16]}")
    typer.echo("Next: uv run ari validate-data --profile " + profile)


@app.command("validate-data")
def validate_data(
    profile: str = typer.Option("dev", help="Profile the dataset was generated with."),
    sample_rows: int = typer.Option(
        400_000, help="Rows of the large facts to load; 0 loads everything."
    ),
) -> None:
    """Validate business invariants and relationship recovery.

    Exits non-zero when an ``error``-severity invariant fails or an intended
    relationship is missing, so this can gate a pipeline.
    """
    from data.validation.report import validate_dataset, write_report

    configure_logging()
    settings = get_settings()
    root = settings.resolve(settings.data.parquet_root).parent

    report = validate_dataset(root, sample_rows=sample_rows or None)
    markdown_path, json_path = write_report(report, root)

    summary = report.checks.summary()
    typer.echo(
        f"invariants   : {summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['warnings']} warnings"
    )
    passed_relationships = sum(1 for r in report.relationships if r.passed)
    typer.echo(f"relationships: {passed_relationships}/{len(report.relationships)} passed")
    typer.echo("")

    for check in report.checks.failures + report.checks.warnings:
        typer.echo(f"  [{check.status}] {check.name} ({check.table}): {check.message}")
    for relationship in report.failed_relationships:
        typer.echo(
            f"  [FAIL] {relationship.name}: observed={relationship.observed} "
            f"expected={relationship.expected}"
        )

    typer.echo("")
    typer.echo(f"report: {markdown_path}")
    typer.echo(f"json  : {json_path}")
    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def forecast(
    product: str = typer.Option(..., help="Product to forecast, e.g. P00003."),
    store: str | None = typer.Option(None, help="Store. Omit to aggregate across stores."),
    horizon: int = typer.Option(28, help="Days ahead: 7, 14, 28, 30 or 90."),
    as_of: str | None = typer.Option(
        None, help="Forecast origin (YYYY-MM-DD). Defaults to the latest fully-informed date."
    ),
    daily: bool = typer.Option(False, "--daily", help="Print the day-by-day path."),
) -> None:
    """Generate a demand forecast from the trained model.

    Exits non-zero on a refusal, so this can gate a pipeline. A refusal is not a
    crash: it prints the error code, whether re-planning could succeed, and what
    would have worked instead.
    """
    from datetime import date as date_type

    from app.schemas.domain import ForecastHorizon
    from app.schemas.forecast import ForecastErrorResponse, ForecastRequest

    configure_logging()

    try:
        selected = next(h for h in ForecastHorizon if h.days == horizon)
    except StopIteration:
        supported = ", ".join(str(h.days) for h in ForecastHorizon)
        typer.echo(f"horizon must be one of {supported}; got {horizon}")
        raise typer.Exit(code=2) from None

    service = Container().forecasting_service
    response = service.forecast(
        ForecastRequest(
            horizon=selected,
            product_ids=[product],
            store_ids=[store] if store else None,
            as_of_date=date_type.fromisoformat(as_of) if as_of else None,
            include_points=daily,
        )
    )

    # `isinstance` rather than a status-string check: it narrows the union for
    # the type checker as well as at runtime.
    if isinstance(response, ForecastErrorResponse):
        typer.echo(f"[{response.error_code}] {response.message}")
        typer.echo(f"recoverable: {response.recoverable}")
        if response.detail:
            typer.echo(f"detail     : {response.detail}")
        raise typer.Exit(code=1)

    typer.echo(response.summary())
    typer.echo("")
    typer.echo(f"as of        : {response.as_of_date}")
    typer.echo(f"series       : {response.series_count}")
    if response.total_predicted_revenue is not None:
        typer.echo(f"revenue      : {response.total_predicted_revenue:,.0f}")
    if response.confidence is not None:
        typer.echo(f"coverage     : {response.confidence:.0%} (measured, not asserted)")
    if response.fallback_used:
        typer.echo(f"fallback     : {response.fallback_reason}")

    if daily and response.points:
        typer.echo("")
        typer.echo(f"{'date':<12}{'units':>10}{'lower':>10}{'upper':>10}")
        for point in response.points:
            lower = f"{point.lower_bound:,.0f}" if point.lower_bound is not None else "-"
            upper = f"{point.upper_bound:,.0f}" if point.upper_bound is not None else "-"
            typer.echo(
                f"{point.date!s:<12}{point.predicted_units:>10,.1f}{lower:>10}{upper:>10}"
            )

    if response.warnings:
        typer.echo("")
        for warning in response.warnings:
            typer.echo(f"  ! {warning}")


@app.command("evaluate-forecast")
def evaluate_forecast() -> None:
    """Print the evaluation report for the persisted forecasting model.

    Reads the report written at training time rather than re-scoring. Re-scoring
    on demand would tempt a caller to treat "run it again" as a way to get a
    number they preferred, and the report is already the justification for the
    selection.
    """
    from ml.forecasting.config import get_forecast_config
    from ml.forecasting.pipeline import default_output_dir

    configure_logging()
    config = get_forecast_config()

    for directory in (
        default_output_dir(config),
        default_output_dir(config).parent / "forecasting_sampled",
        default_output_dir(config).parent / "forecasting",
    ):
        report = directory / "evaluation_report.md"
        if report.is_file():
            typer.echo(report.read_text(encoding="utf-8"))
            typer.echo("")
            typer.echo(f"source: {report}")
            return

    typer.echo("No evaluation report found. Train a model first:")
    typer.echo("  uv run python scripts/train_forecast.py --seed 42")
    raise typer.Exit(code=1)


@app.command("forecast-quality")
def forecast_quality(
    series: int = typer.Option(200, help="Series to sample for the check."),
    seed: int = typer.Option(42, help="Sampling seed."),
) -> None:
    """Data-quality report on the forecasting grain.

    Distinct from ``validate-data``, which checks the generated dataset against
    its own contract. This asks whether the panel is fit to *forecast* from -
    missing dates, duplicate grain, censored targets - which are different
    questions with different answers.

    Exits non-zero on any FAIL.
    """
    from ml.forecasting.config import get_forecast_config
    from ml.forecasting.dataset import build_history
    from ml.forecasting.quality import check_panel, missing_value_summary
    from ml.forecasting.sampling import sample_series

    configure_logging()
    repository = Container().data_repository
    config = get_forecast_config()

    sample = sample_series(repository, n_series=series, seed=seed)
    panel = build_history(repository, config, sample)

    report = check_panel(panel)
    typer.echo(report.render())

    nulls = missing_value_summary(panel)
    if not nulls.empty and nulls["null_rate"].iloc[0] > 0:
        typer.echo("")
        typer.echo("## Highest null rates")
        typer.echo("")
        typer.echo(nulls.to_string(index=False))

    raise typer.Exit(code=0 if report.ok else 1)


@app.command()
def uplift(
    promotion: str | None = typer.Option(None, help="A specific promotion id."),
    product: str | None = typer.Option(None, help="Restrict to one product."),
    store: str | None = typer.Option(None, help="Restrict to one store."),
    region: str | None = typer.Option(None, help="Restrict to one region."),
    events: int = typer.Option(10, help="Individual promotions to list."),
    segments: bool = typer.Option(True, help="Show segment-level uplift."),
) -> None:
    """Incremental sales and profit caused by a promotion.

    Prints the treatment definition first. An uplift number is uninterpretable
    without it - "+18%" measured over the event window and "+18%" net of
    pull-forward are different quantities, and only one of them is the return on
    a promotion.

    Exits non-zero on a refusal, or when causal validation failed.
    """
    from app.schemas.promo_uplift import UpliftErrorResponse, UpliftRequest

    configure_logging()
    response = Container().promo_uplift_service.estimate_uplift(
        UpliftRequest(
            promotion_ids=[promotion] if promotion else None,
            product_ids=[product] if product else None,
            store_ids=[store] if store else None,
            region=region,
            include_segments=segments,
            max_events=max(events, 1),
        )
    )

    if isinstance(response, UpliftErrorResponse):
        typer.echo(f"[{response.error_code}] {response.message}")
        typer.echo(f"recoverable: {response.recoverable}")
        if response.detail:
            typer.echo(f"detail     : {response.detail}")
        raise typer.Exit(code=1)

    typer.echo(f"Estimand: {response.treatment_definition}")
    typer.echo("")
    interval = ""
    if response.confidence_interval is not None:
        band = response.confidence_interval
        interval = (
            f"  [{band.lower:+.1%}, {band.upper:+.1%}] "
            f"at {band.confidence_level:.0%}"
        )
    typer.echo(f"uplift          : {response.uplift_pct:+.1%}{interval}")
    typer.echo(f"incremental     : {response.incremental_units:,.0f} units")
    typer.echo(f"                  {response.incremental_revenue:,.0f} revenue")
    typer.echo(f"                  {response.incremental_profit:,.0f} profit")
    typer.echo(f"spend           : {response.promotion_spend:,.0f}")
    roi = f"{response.roi:.2f}" if response.roi is not None else "n/a (no spend recorded)"
    typer.echo(f"ROI             : {roi}")
    typer.echo(f"method          : {response.method}")
    typer.echo(f"validation      : {response.validation_status}")
    typer.echo(f"promotions      : {response.events_analysed:,}")

    if response.comparison:
        typer.echo("")
        typer.echo(f"{'method':<34}{'uplift':>10}  eligible")
        for row in response.comparison:
            mark = "yes" if row.eligible else "no"
            typer.echo(f"{row.method:<34}{row.uplift_pct:>+10.1%}  {mark}")

    if response.events:
        typer.echo("")
        typer.echo(f"{'promotion':<14}{'product':<10}{'store':<10}{'profit':>12}{'ROI':>8}")
        for event in response.events[:events]:
            event_roi = f"{event.roi:.2f}" if event.roi is not None else "-"
            typer.echo(
                f"{event.promotion_id:<14}{event.product_id:<10}{event.store_id:<10}"
                f"{event.incremental_profit:>12,.0f}{event_roi:>8}"
            )

    if response.segments:
        typer.echo("")
        typer.echo(f"{'dimension':<16}{'segment':<16}{'uplift':>10}  action")
        for segment in response.segments:
            value = f"{segment.uplift_pct:+.1%}" if segment.uplift_pct is not None else "n/a"
            typer.echo(
                f"{segment.dimension:<16}{segment.segment:<16}{value:>10}  "
                f"{segment.classification}"
            )

    if response.assumptions:
        typer.echo("")
        typer.echo("assumptions:")
        for assumption in response.assumptions:
            typer.echo(f"  - {assumption}")

    if response.warnings:
        typer.echo("")
        for warning in response.warnings:
            typer.echo(f"  ! {warning}")

    # A failed validation exits non-zero even though a number was produced. A
    # script piping this into a report should have to opt in to using an
    # estimate whose causal assumptions did not hold.
    raise typer.Exit(code=0 if response.is_causal else 1)


@app.command("uplift-quality")
def uplift_quality(
    series: int = typer.Option(200, help="Series to sample for the check."),
    seed: int = typer.Option(42, help="Sampling seed."),
) -> None:
    """Data-quality report on the causal grain.

    Different questions from ``forecast-quality``. A forecasting model degrades
    when its inputs are imperfect and the backtest shows it; a causal estimate
    misleads instead, silently. These checks look for the problems that bias an
    effect - missing treatment labels, overlapping promotions, censoring that
    differs between the arms - and state the direction of each bias.

    Exits non-zero on any FAIL.
    """
    from ml.forecasting.sampling import sample_series
    from ml.promo_uplift.config import get_promo_uplift_config
    from ml.promo_uplift.quality import check_panel

    configure_logging()
    repository = Container().data_repository
    config = get_promo_uplift_config()

    sample = sample_series(repository, n_series=series, seed=seed)
    panel = _uplift_panel(repository, sample)

    report = check_panel(panel, config=config)
    typer.echo(report.render())
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("uplift-validate")
def uplift_validate(
    scenario: str | None = typer.Option(
        None, help="One scenario name. Omit to run all of them."
    ),
    series: int = typer.Option(150, help="Synthetic series per scenario."),
) -> None:
    """Recover known treatment effects from synthetic data.

    The only test that can establish a causal estimator is correct. Real data
    has no counterfactual, so a method can score perfectly on every predictive
    metric while being wrong about the effect by any margin. Here the effect is
    applied by hand and recorded, so recovery is checkable.

    Exits non-zero if any scenario is not recovered.
    """
    from ml.promo_uplift.config import get_promo_uplift_config
    from ml.promo_uplift.controls import build_control_pool
    from ml.promo_uplift.estimators import AIPWEstimator, fit_nuisances
    from ml.promo_uplift.features import build_covariates
    from ml.promo_uplift.synthetic import SCENARIOS, generate, scenario_config
    from ml.promo_uplift.treatment import build_analysis_frame

    configure_logging()
    base = get_promo_uplift_config()
    names = [scenario] if scenario else list(SCENARIOS)

    typer.echo(f"{'scenario':<18}{'true':>9}{'naive':>9}{'AIPW':>9}{'error':>9}  verdict")
    failures = 0
    for name in names:
        if name not in SCENARIOS:
            typer.echo(f"unknown scenario {name!r}; expected one of {', '.join(SCENARIOS)}")
            raise typer.Exit(code=2)

        config = scenario_config(name, base)
        panel = generate(name, config=config, n_series=series)
        analysis = build_analysis_frame(panel.observable(), config=config)
        pool = build_control_pool(analysis, config=config)
        covariates = build_covariates(
            pool.frame, analysis.events, config=config, history=analysis.frame
        )
        nuisance = fit_nuisances(covariates, config=config)
        estimate = AIPWEstimator(config=config).fit(covariates, nuisance).estimate_ate()

        y, t = covariates.y, covariates.t
        naive = y[t].mean() / y[~t].mean() - 1.0
        error = abs(estimate.ate_pct - panel.true_att_pct)
        ok = error <= base.synthetic.recovery_tolerance * max(
            abs(panel.true_att_pct) / 0.15, 1.0
        )
        failures += not ok
        typer.echo(
            f"{name:<18}{panel.true_att_pct:>+9.1%}{naive:>+9.1%}"
            f"{estimate.ate_pct:>+9.1%}{error:>9.1%}  {'PASS' if ok else 'FAIL'}"
        )

    raise typer.Exit(code=0 if failures == 0 else 1)


def _uplift_panel(repository: Any, pairs: Any) -> Any:
    """Sales joined to the promotion calendar, at the causal grain.

    Kept out of the command body because the same join is what
    ``scripts/estimate_uplift.py`` builds, and two copies of a join that decides
    which rows count as treated is exactly the kind of duplication that lets
    them drift apart.
    """
    from ml.promo_uplift.data import build_uplift_panel

    return build_uplift_panel(repository, pairs)


@app.command()
def elasticity(
    product: str = typer.Option(..., help="Product to estimate elasticity for."),
    region: str | None = typer.Option(None, help="Restrict to one region."),
    cross: bool = typer.Option(False, help="Also show substitutes and complements."),
    compare: bool = typer.Option(False, help="Show every estimator side by side."),
) -> None:
    """Own-price elasticity, and optionally what the product competes with.

    Prints the method before the number. On this data the naive estimator
    recovers only ~56% of the true elasticity, so an elasticity that travels
    without saying how it was identified invites the wrong pricing decision.
    """
    from app.schemas.elasticity import ElasticityErrorResponse, ElasticityRequest

    configure_logging()
    response = Container().elasticity_service.estimate(
        ElasticityRequest(
            product_id=product,
            region=region,
            include_cross_price=cross,
            include_comparison=compare,
        )
    )

    if isinstance(response, ElasticityErrorResponse):
        typer.echo(f"[{response.error_code}] {response.message}")
        typer.echo(f"recoverable: {response.recoverable}")
        raise typer.Exit(code=1)

    band = ""
    if response.confidence_interval is not None:
        low, high = response.confidence_interval
        band = f"  [{low:.3f}, {high:.3f}]"

    typer.echo(f"method      : {response.method}")
    typer.echo(f"elasticity  : {response.elasticity:+.3f}{band}")
    typer.echo(
        f"reading     : {'ELASTIC' if response.is_elastic else 'inelastic'} "
        f"- a price rise {response.revenue_direction}"
    )
    typer.echo(f"sample      : {response.sample_size:,} rows")
    if response.estimation_window:
        start, end = response.estimation_window
        typer.echo(f"window      : {start} .. {end}")

    if response.comparison:
        typer.echo("")
        typer.echo(f"{'method':<14}{'elasticity':>12}  selectable")
        for row in response.comparison:
            mark = "yes" if row.selectable else "no"
            typer.echo(f"{row.method:<14}{row.elasticity:>12.3f}  {mark}")

    if response.cross_price:
        typer.echo("")
        typer.echo(f"tested {response.pairs_tested} candidate pairs")
        typer.echo(f"{'product':<12}{'cross':>9}  relationship  strength")
        for record in response.cross_price:
            if record.is_significant:
                typer.echo(
                    f"{record.source_product_id:<12}{record.cross_elasticity:>9.3f}  "
                    f"{record.relationship_type:<13} {record.strength}"
                )
        if not response.substitutes and not response.complements:
            typer.echo("  no significant relationships after multiple-testing correction")

    if response.warnings:
        typer.echo("")
        for warning in response.warnings:
            typer.echo(f"  ! {warning}")


@app.command()
def prompts() -> None:
    """List available prompt versions."""
    from prompts.registry import list_prompts

    available = list_prompts()
    if not available:
        typer.echo("No prompts found.")
        return
    for name, versions in available.items():
        typer.echo(f"{name:<20} {', '.join(versions) or '(none)'}")


@app.command("evaluate-agent")
def evaluate_agent(
    provider: str = typer.Option(
        "stub",
        help="stub (offline keyword baseline) or claude (costs money, hits the API).",
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Record this run as the committed baseline."
    ),
    detail: bool = typer.Option(False, "--detail", help="Print every question."),
    output: str | None = typer.Option(None, help="Write the full report as JSON here."),
) -> None:
    """Score the agent against the golden set derived from injected scenarios.

    The stub run is offline, free and deterministic, and measures the *harness*
    plus a deliberately weak keyword planner - it is the floor, not a benchmark.
    The claude run is the capability measurement.
    """
    import json as _json
    from pathlib import Path as _Path

    from app.llm.stub import StubProvider
    from evaluation.baseline_planner import KeywordBaseline
    from evaluation.golden_set import load_golden_set
    from evaluation.runner import (
        compare_to_baseline,
        load_baseline,
        run_golden_set,
        write_baseline,
    )

    configure_logging()
    container = Container()
    registry = container.tool_registry
    questions = load_golden_set()

    if provider == "stub":
        stub = StubProvider()
        KeywordBaseline(available_tools=set(registry.names())).script(
            stub, list(questions)
        )
        llm: Any = stub
        name = "stub+keyword"
    elif provider == "claude":
        llm = container.llm_provider
        name = llm.model_name
    else:
        typer.echo(f"unknown provider '{provider}'. Use 'stub' or 'claude'.")
        raise typer.Exit(code=2)

    typer.echo(f"Running {len(questions)} golden questions against {name}...\n")
    run = run_golden_set(llm, registry, questions=questions, provider_name=name)
    report = run.as_dict()

    coverage = report["coverage"]
    typer.echo(
        f"coverage         : {coverage['answerable']} answerable, "
        f"{coverage['abstention_expected']} expect abstention"
    )
    typer.echo(f"answerable mean  : {report['answerable_mean']:.3f}")
    typer.echo(f"abstention mean  : {report['abstention_mean']:.3f}")
    if report["artefact_gaps"]:
        gaps = ", ".join(f"{k} x{v}" for k, v in report["artefact_gaps"].items())
        typer.echo(f"artefact gaps    : {gaps}")
        typer.echo(
            "                   (tools ran and found no data for that "
            "product/window - retrain, not re-prompt)"
        )
    typer.echo("")
    for dimension, value in report["dimensions"].items():
        typer.echo(f"  {dimension:<16} {value:.3f}")
    typer.echo("")
    for label, value in report["by_label"].items():
        typer.echo(f"  {label:<22} {value:.3f}")

    if run.failures:
        typer.echo("")
        for question_id, error in run.failures.items():
            typer.echo(f"  ! {question_id}: {error}")

    if detail:
        typer.echo("")
        for score in report["questions_detail"]:
            typer.echo(f"{score['question_id']:<8} {score['overall']:.2f}  {score['label']}")
            for note in score["notes"]:
                typer.echo(f"         - {note}")

    if output:
        _Path(output).write_text(_json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nreport written to {output}")

    if update_baseline:
        path = write_baseline(run)
        typer.echo(f"\nbaseline updated: {path}")
        return

    regressions = compare_to_baseline(run, load_baseline(name))
    if regressions:
        typer.echo("\nREGRESSIONS against the committed baseline:")
        for regression in regressions:
            typer.echo(
                f"  {regression.dimension:<24} "
                f"{regression.baseline:.3f} -> {regression.current:.3f} "
                f"({regression.delta:+.3f})"
            )
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
