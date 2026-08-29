"""Running the golden set and reporting what happened.

Takes a registry and a provider rather than building them, so the same runner
grades the real tools against Claude, fake tools against a scripted stub, and
anything in between. The alternative - a runner that constructs its own
dependencies - would make the offline test and the real run different code, and
the one that matters would be the one never exercised.

**A failed investigation is a score of zero, not a crash.** An agent that throws
on four questions and answers sixteen well has a real problem, and a runner that
aborts on the first exception reports it as "no result" instead of "0.0 on those
four". The error text travels into the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.recommendation import RecommendationAgent
from app.agents.supervisor import SupervisorAgent
from app.config.settings import AgentSettings
from app.guardrails.budget import BudgetTracker
from app.llm.base import LLMProvider
from app.observability.logging import get_logger
from app.schemas.agent_state import AgentState, new_agent_state
from app.tools.registry import ToolRegistry
from app.workflows.graph import InvestigationDeps, run_investigation
from evaluation.golden_set import GoldenQuestion, coverage_summary, load_golden_set
from evaluation.scoring import RunScore, score_run

logger = get_logger(__name__)

#: Where committed baselines live, so a regression is a diff rather than a
#: memory of what the number used to be.
BASELINE_DIR = Path("evaluation/baselines")


def baseline_path(provider: str, directory: Path | None = None) -> Path:
    """One baseline per provider.

    Kept apart because they measure different things. The keyword floor and a
    language model do not belong in the same file, and a Claude run graded
    against the stub's numbers would report noise as improvement.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in provider)
    return (directory or BASELINE_DIR) / f"{safe}.json"


@dataclass
class EvaluationRun:
    """A finished run: the scores plus everything needed to interpret them."""

    score: RunScore
    coverage: dict[str, Any]
    failures: dict[str, str]
    provider: str
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "provider": self.provider,
            "duration_seconds": round(self.duration_seconds, 2),
            "coverage": self.coverage,
            "failures": self.failures,
            **self.score.as_dict(),
        }


def build_deps(
    provider: LLMProvider,
    registry: ToolRegistry,
    *,
    settings: AgentSettings | None = None,
) -> InvestigationDeps:
    """Assemble the graph dependencies for an evaluation run."""
    agent_settings = settings or AgentSettings()
    return InvestigationDeps(
        supervisor=SupervisorAgent(provider=provider, tools=registry.specs()),
        registry=registry,
        budget=BudgetTracker.from_settings(agent_settings),
        critic=CriticAgent(provider=provider),
        recommender=RecommendationAgent(
            provider=provider,
            approval_threshold=agent_settings.human_approval_threshold,
        ),
        max_replans=agent_settings.max_replans,
    )


def run_golden_set(
    provider: LLMProvider,
    registry: ToolRegistry,
    *,
    questions: tuple[GoldenQuestion, ...] | None = None,
    settings: AgentSettings | None = None,
    provider_name: str | None = None,
) -> EvaluationRun:
    """Run every golden question and score the results."""
    graded = questions if questions is not None else load_golden_set()
    started = datetime.now(UTC)
    results: list[tuple[GoldenQuestion, AgentState]] = []
    failures: dict[str, str] = {}

    for question in graded:
        # A fresh budget per question. Sharing one would make a late question's
        # score depend on how expensive the earlier ones happened to be, which
        # is a property of the set's ordering, not of the agent.
        deps = build_deps(provider, registry, settings=settings)
        try:
            # The question id is prepended so a scripted provider can key on it.
            # Harmless to a real model, which reads it as a reference.
            state = run_investigation(
                f"[{question.question_id}] {question.question}", deps
            )
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            logger.warning(
                "evaluation.question_failed",
                question_id=question.question_id,
                error=str(exc),
            )
            failures[question.question_id] = f"{type(exc).__name__}: {exc}"
            state = new_agent_state(question.question)
        results.append((question, state))

    name = str(provider_name or getattr(provider, "model_name", None) or "unknown")
    duration = (datetime.now(UTC) - started).total_seconds()

    run = EvaluationRun(
        score=score_run(results, provider=name),
        coverage=coverage_summary(tuple(graded)),
        failures=failures,
        provider=name,
        duration_seconds=duration,
    )
    logger.info(
        "evaluation.completed",
        provider=name,
        questions=len(results),
        answerable_mean=round(run.score.answerable_mean, 3),
        abstention_mean=round(run.score.abstention_mean, 3),
        failures=len(failures),
    )
    return run


# --------------------------------------------------------------------------
# Baseline comparison
# --------------------------------------------------------------------------


def load_baseline(provider: str, path: Path | None = None) -> dict[str, Any] | None:
    """The committed score for this provider, or None if none was recorded."""
    source = path or baseline_path(provider)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def write_baseline(run: EvaluationRun, path: Path | None = None) -> Path:
    """Record a run as the baseline for its provider.

    Per-question detail is kept. Knowing the mean fell is the alarm; knowing
    *which* questions fell is the diagnosis, and reconstructing that from a
    single number is impossible.
    """
    target = path or baseline_path(run.provider)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(run.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


@dataclass
class Regression:
    """One dimension that moved the wrong way."""

    dimension: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def compare_to_baseline(
    run: EvaluationRun,
    baseline: dict[str, Any] | None,
    *,
    tolerance: float = 0.02,
) -> list[Regression]:
    """Which headline numbers fell beyond tolerance.

    A tolerance exists because scores move slightly for reasons that are not
    regressions - a tie broken differently, a float summed in another order. A
    check that fires on noise gets ignored, and an ignored check is not a check.
    """
    if not baseline:
        return []
    if baseline.get("provider") != run.provider:
        # Refusing rather than comparing. A mismatched baseline produces a
        # number-shaped answer to a question nobody asked.
        raise ValueError(
            f"baseline was recorded for provider "
            f"{baseline.get('provider')!r}, not {run.provider!r}"
        )

    current = run.as_dict()
    regressions: list[Regression] = []

    for key in ("answerable_mean", "abstention_mean"):
        before = float(baseline.get(key, 0.0))
        after = float(current.get(key, 0.0))
        if after < before - tolerance:
            regressions.append(Regression(key, before, after))

    for name, before in (baseline.get("dimensions") or {}).items():
        after = float((current.get("dimensions") or {}).get(name, 0.0))
        if after < float(before) - tolerance:
            regressions.append(Regression(f"dimensions.{name}", float(before), after))

    return regressions


__all__ = [
    "BASELINE_DIR",
    "EvaluationRun",
    "Regression",
    "baseline_path",
    "build_deps",
    "compare_to_baseline",
    "load_baseline",
    "run_golden_set",
    "write_baseline",
]
