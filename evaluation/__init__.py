"""Agent evaluation against a golden set with known answers.

The step that makes the rest defensible. Every earlier step validated a *model*
against ground truth; this validates the *agent* - whether it selects the right
tools, gathers what it needs, gets the direction right, and declines when the
evidence cannot support an answer.

Three modules:

``golden_set``  questions derived from the scenarios Step 2 injected
``scoring``     four dimensions, each scored independently and reported apart
``runner``      executes the set and aggregates, against any provider

The headline caveat, stated here so it is impossible to miss: **a stub run does
not measure the model.** The stub returns what it was scripted to return, so a
stub score measures the harness and the graph. What makes the stub number
meaningful is that it is scripted from a deliberately weak keyword planner - so
it is a real measurement of a real (bad) policy, and the floor a language model
has to beat to justify its cost.
"""

from evaluation.golden_set import GoldenQuestion, coverage_summary, load_golden_set
from evaluation.scoring import QuestionScore, RunScore, score_question, score_run

__all__ = [
    "GoldenQuestion",
    "QuestionScore",
    "RunScore",
    "coverage_summary",
    "load_golden_set",
    "score_question",
    "score_run",
]
