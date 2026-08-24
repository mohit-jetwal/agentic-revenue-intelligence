"""ASGI entrypoint.

Run with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from app.api.app import create_app

app = create_app()
