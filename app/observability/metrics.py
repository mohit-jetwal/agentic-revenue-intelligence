"""In-process counters backing ``GET /metrics``.

Intentionally minimal. A Prometheus exporter with no scraper is dead weight in
Stage 1, and in Stage 2 monitoring is Databricks-native (system tables, lakehouse
monitoring, MLflow). This exists so the endpoint from section 20 is real rather
than a stub, and so agent-level counters (tool calls, tokens) have somewhere to
accumulate from Step 16 onward.

Counters are process-local and reset on restart. That is acceptable for a
single-process dev server and is called out in the README.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    """Thread-safe counters.

    A lock rather than ``itertools.count`` because reads must see a consistent
    snapshot across counters, not just an atomic increment on one.
    """

    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    requests_total: int = 0
    requests_failed: int = 0
    investigations_started: int = 0
    investigations_completed: int = 0
    tool_calls_total: int = 0
    tokens_used_total: int = 0

    def increment(self, name: str, amount: int = 1) -> None:
        if not hasattr(self, name) or name.startswith("_"):
            raise KeyError(f"unknown metric: {name!r}")
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self.started_at, 3),
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "investigations_started": self.investigations_started,
                "investigations_completed": self.investigations_completed,
                "tool_calls_total": self.tool_calls_total,
                "tokens_used_total": self.tokens_used_total,
            }

    def reset(self) -> None:
        """Intended for tests only."""
        with self._lock:
            self.started_at = time.monotonic()
            self.requests_total = 0
            self.requests_failed = 0
            self.investigations_started = 0
            self.investigations_completed = 0
            self.tool_calls_total = 0
            self.tokens_used_total = 0


METRICS = MetricsRegistry()
