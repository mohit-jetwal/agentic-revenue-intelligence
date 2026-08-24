"""Command-line interface.

Run with ``ari <command>`` after ``uv sync``, or ``python -m app.cli``.

Step 1 provides introspection commands only - enough to verify the skeleton is
wired correctly without starting the server. Data generation, training and
evaluation commands are added by their respective steps.
"""

from __future__ import annotations

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
def prompts() -> None:
    """List available prompt versions."""
    from prompts.registry import list_prompts

    available = list_prompts()
    if not available:
        typer.echo("No prompts found.")
        return
    for name, versions in available.items():
        typer.echo(f"{name:<20} {', '.join(versions) or '(none)'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
