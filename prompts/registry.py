"""Versioned prompt loading.

Prompts live in files under ``prompts/<agent>/<version>.md``, not in Python
string literals. Two reasons, both practical:

* A prompt change is a behaviour change. In a file it shows up in a diff and can
  be reviewed; embedded in code it hides among refactors.
* Evaluation results are only meaningful against a known prompt. Recording
  ``prompt_version`` in the trace lets a regression be traced to the edit that
  caused it - which is impossible if the prompt is whatever happened to be in
  the source at the time.

Loaded content is cached, so file I/O happens once per version per process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_ROOT = Path(__file__).resolve().parent


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt name/version does not exist on disk."""


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> str:
    """Return the prompt text for ``name`` at ``version``.

    ``name`` is the agent directory, e.g. ``"supervisor"``.
    """
    path = PROMPTS_ROOT / name / f"{version}.md"
    if not path.is_file():
        available = list_versions(name)
        raise PromptNotFoundError(
            f"No prompt {name!r} version {version!r} at {path}. "
            f"Available versions: {available or 'none'}"
        )
    return path.read_text(encoding="utf-8").strip()


def list_versions(name: str) -> list[str]:
    """Available versions for a prompt, newest-looking last."""
    directory = PROMPTS_ROOT / name
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.md"))


def list_prompts() -> dict[str, list[str]]:
    """Every prompt name mapped to its available versions."""
    return {
        d.name: list_versions(d.name)
        for d in sorted(PROMPTS_ROOT.iterdir())
        if d.is_dir() and not d.name.startswith("_") and d.name != "__pycache__"
    }


def clear_cache() -> None:
    """Drop cached prompt text. Intended for tests and hot-reload."""
    load_prompt.cache_clear()
