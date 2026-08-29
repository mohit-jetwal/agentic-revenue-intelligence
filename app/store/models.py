"""Application-state tables: investigations, traces, feedback.

**Not analytics.** DuckDB and Parquet hold the business data and answer
analytical questions over millions of rows. This holds what the *application*
did - which questions were asked, what the agent decided, what a person thought
of it. Two engines, one job each, and mixing them would put a write-heavy
transactional workload on a columnar store built for scans.

**Why the recommendation is stored as JSON rather than normalised.**
``Recommendation`` is a Pydantic model with nested evidence, scenarios and
assumptions, and it is always read whole. Normalising it into five tables would
buy join flexibility nobody needs and cost a migration every time the model
gains a field. The trade-off is that the recommendation is not queryable by
SQL - accepted, because the questions asked of this store are "what did
investigation X conclude", never "find every recommendation mentioning margin".

**Traces are append-only and sequenced.** An investigation's trace is the record
of how it reached its answer, and a record that can be edited after the fact is
not evidence of anything.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class InvestigationRow(Base):
    """One question asked and, eventually, answered."""

    __tablename__ = "investigations"

    investigation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    objective: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: The serialised `Recommendation`. Null until the investigation concludes -
    #: and null is the honest value for one that failed, rather than an empty
    #: recommendation that would read as "we found nothing" instead of "we
    #: never finished".
    recommendation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    #: Counters worth keeping without deserialising the whole trace.
    tool_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    replans: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[list[TraceEventRow]] = relationship(
        back_populates="investigation",
        cascade="all, delete-orphan",
        order_by="TraceEventRow.sequence",
    )

    __table_args__ = (Index("ix_investigations_created", "created_at"),)


class TraceEventRow(Base):
    """One step in how an investigation reached its answer.

    ``summary`` is a short, user-facing rationale. Private chain-of-thought is
    never written here: a store is the place a leak becomes permanent, and the
    constraint is worth enforcing at the schema rather than trusting each
    caller to remember it.
    """

    __tablename__ = "trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    investigation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("investigations.investigation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    actor: Mapped[str] = mapped_column(String(48), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    investigation: Mapped[InvestigationRow] = relationship(back_populates="events")

    __table_args__ = (
        # An investigation's events are always read in order and never
        # individually, so the useful index is the composite one.
        Index("ix_trace_investigation_sequence", "investigation_id", "sequence"),
    )


class FeedbackRow(Base):
    """What a person thought of a recommendation.

    Not linked by foreign key to `investigations`. Feedback on an investigation
    that was later purged is still a signal about the system, and a cascade
    delete would quietly destroy the only human-labelled data this platform
    produces.
    """

    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


__all__ = ["Base", "FeedbackRow", "InvestigationRow", "TraceEventRow"]
