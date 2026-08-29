"""Reading and writing application state.

A narrow store, deliberately. It records investigations, appends trace events and
takes feedback; it does not know what an investigation *means*. Keeping the
interpretation in the service layer is what lets the same store back the API, the
CLI and the UI without any of them reaching into SQL.

**Schema creation is idempotent and happens on construction.** No migration tool
for four tables that no deployed instance has yet written to. When a second
instance exists, Alembic is the answer; adding it now would be ceremony around a
file that gets deleted whenever the demo is reset.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.observability.logging import get_logger
from app.schemas.api import (
    InvestigationResponse,
    InvestigationStatus,
    TraceEvent,
    TraceResponse,
)
from app.schemas.domain import BusinessObjective, IntentType
from app.store.models import Base, FeedbackRow, InvestigationRow, TraceEventRow

logger = get_logger(__name__)


def _ensure_parent(url: str) -> None:
    """Create the directory a SQLite file lives in.

    SQLite will not create a missing directory and fails with an opaque "unable
    to open database file" if one is absent - which on a fresh clone is the
    first thing anyone hits.
    """
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database:
        if parsed.database == ":memory:":
            return
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class InvestigationStore:
    """Persistence for investigations, traces and feedback."""

    database_url: str
    _engine: Engine | None = None
    _sessions: sessionmaker[Session] | None = None

    def __post_init__(self) -> None:
        _ensure_parent(self.database_url)
        connect_args: dict[str, Any] = {}
        if self.database_url.startswith("sqlite"):
            # The API runs investigations on a threadpool, so the connection is
            # not guaranteed to be reused on the thread that opened it.
            connect_args["check_same_thread"] = False
        self._engine = create_engine(self.database_url, connect_args=connect_args)
        self._sessions = sessionmaker(bind=self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        # Raised rather than asserted: an `assert` vanishes under `python -O`,
        # and the failure it would have caught becomes a `NoneType is not
        # callable` on the next line instead.
        if self._sessions is None:
            raise RuntimeError(
                "InvestigationStore was not initialised; __post_init__ did not run"
            )
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- investigations -----------------------------------------------------

    def create(
        self,
        *,
        investigation_id: str,
        trace_id: str,
        question: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record an investigation as started.

        Written *before* the agent runs, not after. An investigation that
        crashes mid-flight is exactly the one worth having a record of, and a
        store written only on success cannot show you the failures.
        """
        with self.session() as session:
            session.add(
                InvestigationRow(
                    investigation_id=investigation_id,
                    trace_id=trace_id,
                    question=question,
                    user_id=user_id,
                    session_id=session_id,
                    status=InvestigationStatus.RUNNING.value,
                )
            )

    def complete(
        self,
        investigation_id: str,
        *,
        status: InvestigationStatus,
        intent: str | None = None,
        objective: str | None = None,
        recommendation: dict[str, Any] | None = None,
        tool_calls: int = 0,
        replans: int = 0,
        error: str | None = None,
    ) -> None:
        with self.session() as session:
            row = session.get(InvestigationRow, investigation_id)
            if row is None:
                logger.warning("store.complete_missing", investigation_id=investigation_id)
                return
            row.status = status.value
            row.intent = intent
            row.objective = objective
            row.recommendation = recommendation
            row.tool_calls = tool_calls
            row.replans = replans
            row.error = error
            row.completed_at = datetime.now(UTC)

    def get(self, investigation_id: str) -> InvestigationResponse | None:
        with self.session() as session:
            row = session.get(InvestigationRow, investigation_id)
            return _to_response(row) if row else None

    def recent(self, limit: int = 20) -> list[InvestigationResponse]:
        with self.session() as session:
            rows = session.scalars(
                select(InvestigationRow)
                .order_by(InvestigationRow.created_at.desc())
                .limit(limit)
            ).all()
            return [_to_response(row) for row in rows]

    # -- traces -------------------------------------------------------------

    def append_events(self, investigation_id: str, events: list[TraceEvent]) -> None:
        """Append trace events in one transaction.

        Batched rather than written per event: a trace is only meaningful whole,
        and a partial one from a failed write would misrepresent what happened
        more than no trace at all.
        """
        if not events:
            return
        with self.session() as session:
            session.add_all(
                TraceEventRow(
                    investigation_id=investigation_id,
                    sequence=event.sequence,
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    actor=event.actor,
                    summary=event.summary,
                    tool_name=event.tool_name,
                    duration_ms=event.duration_ms,
                    payload=event.payload,
                )
                for event in events
            )

    def get_trace(self, investigation_id: str) -> TraceResponse | None:
        with self.session() as session:
            row = session.get(InvestigationRow, investigation_id)
            if row is None:
                return None
            events = session.scalars(
                select(TraceEventRow)
                .where(TraceEventRow.investigation_id == investigation_id)
                .order_by(TraceEventRow.sequence)
            ).all()
            return TraceResponse(
                investigation_id=investigation_id,
                trace_id=row.trace_id,
                events=[
                    TraceEvent(
                        sequence=e.sequence,
                        timestamp=e.timestamp,
                        event_type=e.event_type,
                        actor=e.actor,
                        summary=e.summary,
                        tool_name=e.tool_name,
                        duration_ms=e.duration_ms,
                        payload=e.payload or {},
                    )
                    for e in events
                ],
            )

    # -- feedback -----------------------------------------------------------

    def add_feedback(
        self,
        *,
        investigation_id: str,
        helpful: bool,
        rating: int | None = None,
        comment: str | None = None,
        user_id: str | None = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        with self.session() as session:
            session.add(
                FeedbackRow(
                    feedback_id=feedback_id,
                    investigation_id=investigation_id,
                    helpful=helpful,
                    rating=rating,
                    comment=comment,
                    user_id=user_id,
                )
            )
        return feedback_id

    def feedback_for(self, investigation_id: str) -> list[dict[str, Any]]:
        with self.session() as session:
            rows = session.scalars(
                select(FeedbackRow)
                .where(FeedbackRow.investigation_id == investigation_id)
                .order_by(FeedbackRow.created_at)
            ).all()
            return [
                {
                    "feedback_id": r.feedback_id,
                    "helpful": r.helpful,
                    "rating": r.rating,
                    "comment": r.comment,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    # -- maintenance --------------------------------------------------------

    def purge(self, investigation_id: str) -> bool:
        """Delete an investigation and its trace. Feedback survives."""
        with self.session() as session:
            row = session.get(InvestigationRow, investigation_id)
            if row is None:
                return False
            session.execute(
                delete(TraceEventRow).where(
                    TraceEventRow.investigation_id == investigation_id
                )
            )
            session.delete(row)
            return True

    def health_check(self) -> tuple[bool, str]:
        try:
            with self.session() as session:
                count = len(session.scalars(select(InvestigationRow.investigation_id)).all())
            return True, f"sqlite app state ({count} investigations)"
        except Exception as exc:  # noqa: BLE001 - health checks must not raise
            return False, f"{type(exc).__name__}: {exc}"


def _to_response(row: InvestigationRow) -> InvestigationResponse:
    """Rebuild the API model from a stored row.

    The recommendation is re-validated through Pydantic on the way out rather
    than trusted. A row written by an older version of the model would otherwise
    surface as a shape the API promised but no longer produces.
    """
    from app.schemas.agent_state import Recommendation

    recommendation = None
    if row.recommendation:
        try:
            recommendation = Recommendation.model_validate(row.recommendation)
        except Exception as exc:  # noqa: BLE001 - a stale row must not 500 the read
            logger.warning(
                "store.recommendation_unreadable",
                investigation_id=row.investigation_id,
                error=str(exc),
            )

    return InvestigationResponse(
        investigation_id=row.investigation_id,
        trace_id=row.trace_id,
        status=InvestigationStatus(row.status),
        question=row.question,
        intent=IntentType(row.intent) if row.intent else None,
        objective=BusinessObjective(row.objective) if row.objective else None,
        recommendation=recommendation,
        created_at=row.created_at,
        completed_at=row.completed_at,
    )


__all__ = ["InvestigationStore"]
