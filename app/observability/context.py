"""Trace context propagation.

A single investigation fans out across the API layer, several agent nodes and
many tool calls. Correlating those into one story requires a trace identifier
that follows execution without being threaded through every function signature.

``contextvars`` gives us that, and unlike a thread-local it behaves correctly
under ``asyncio`` - each task inherits a copy of the context at creation, so
concurrent investigations cannot overwrite each other's trace ids.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_investigation_id: ContextVar[str | None] = ContextVar("investigation_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def get_trace_id() -> str | None:
    return _trace_id.get()


def get_investigation_id() -> str | None:
    return _investigation_id.get()


def get_user_id() -> str | None:
    return _user_id.get()


def current_context() -> dict[str, str]:
    """Non-null context values, ready to merge into a log record."""
    ctx: dict[str, str] = {}
    if (tid := _trace_id.get()) is not None:
        ctx["trace_id"] = tid
    if (iid := _investigation_id.get()) is not None:
        ctx["investigation_id"] = iid
    if (uid := _user_id.get()) is not None:
        ctx["user_id"] = uid
    return ctx


@contextmanager
def trace_context(
    trace_id: str | None = None,
    *,
    investigation_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[str]:
    """Bind trace context for the duration of a block.

    Yields the effective trace id. Tokens are reset in reverse order on exit so
    nested contexts restore correctly.
    """
    tid = trace_id or _trace_id.get() or new_trace_id()
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (_trace_id, _trace_id.set(tid))
    ]
    if investigation_id is not None:
        tokens.append((_investigation_id, _investigation_id.set(investigation_id)))
    if user_id is not None:
        tokens.append((_user_id, _user_id.set(user_id)))
    try:
        yield tid
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
