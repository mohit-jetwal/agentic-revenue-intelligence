"""API surface: health, metrics, trace propagation, and honest stubs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.middleware import TRACE_HEADER
from app.config.settings import DataSettings
from tests.conftest import build_settings

pytestmark = pytest.mark.integration


# --- health ---------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_health_reports_environment_and_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["environment"] == "local"
    assert body["name"] == "agentic-revenue-intelligence"
    assert body["version"]


def test_health_lists_every_dependency(client: TestClient) -> None:
    names = {d["name"] for d in client.get("/health").json()["dependencies"]}
    assert {
        "data_repository",
        "model_registry",
        "vector_store",
        "llm_provider",
        "tool_registry",
    } <= names


def test_health_is_degraded_before_data_is_generated(tmp_path: Path) -> None:
    """Honest reporting: no data and no API key is degraded, not healthy.

    Builds its own app pointed at an empty temporary root. Using the configured
    root would make the result depend on whether the developer happens to have
    generated a dataset, so the test would pass or fail for reasons unrelated to
    the code.
    """
    settings = build_settings(
        data=DataSettings(_env_file=None, parquet_root=tmp_path / "gold")
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    by_name = {d["name"]: d for d in body["dependencies"]}
    assert by_name["data_repository"]["status"] == "not_configured"
    assert "Step 2" in by_name["data_repository"]["detail"]


def test_health_is_ok_once_data_exists(smoke_result: object, tmp_path: Path) -> None:
    """The counterpart: with a dataset present, the repository reports OK."""
    settings = build_settings(
        data=DataSettings(
            _env_file=None,
            parquet_root=smoke_result.root / "gold",  # type: ignore[attr-defined]
        )
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/health").json()

    by_name = {d["name"]: d for d in body["dependencies"]}
    assert by_name["data_repository"]["status"] == "ok"
    assert "v1.0-smoke" in by_name["data_repository"]["detail"]


def test_health_stays_200_when_a_dependency_is_missing(client: TestClient) -> None:
    """A health endpoint that fails on a known-missing dependency is useless."""
    assert client.get("/health").status_code == 200


# --- metrics --------------------------------------------------------------


def test_metrics_returns_counters(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert set(body) == {
        "uptime_seconds",
        "requests_total",
        "requests_failed",
        "investigations_started",
        "investigations_completed",
        "tool_calls_total",
        "tokens_used_total",
    }


def test_request_counter_increments(client: TestClient) -> None:
    client.get("/health")
    client.get("/health")
    assert client.get("/metrics").json()["requests_total"] >= 3


# --- trace propagation ----------------------------------------------------


def test_trace_id_is_returned(client: TestClient) -> None:
    assert client.get("/health").headers[TRACE_HEADER]


def test_inbound_trace_id_is_honoured(client: TestClient) -> None:
    """Lets a caller correlate this request with its own upstream trace."""
    response = client.get("/health", headers={TRACE_HEADER: "trace-from-caller"})
    assert response.headers[TRACE_HEADER] == "trace-from-caller"


# --- stub endpoints -------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/chat", {"question": "Why did revenue decline?"}),
        ("post", "/investigate", {"question": "Why did revenue decline?"}),
        ("post", "/scenario", {"description": "Raise price 5%"}),
        ("get", "/investigation/abc", None),
        ("get", "/investigation/abc/trace", None),
        ("get", "/models", None),
        ("post", "/feedback", {"investigation_id": "abc", "helpful": True}),
    ],
)
def test_unimplemented_endpoints_return_501_naming_their_step(
    client: TestClient, method: str, path: str, payload: dict[str, object] | None
) -> None:
    """A 501 that names the step is a roadmap; a 404 is just absence."""
    response = getattr(client, method)(path, **({"json": payload} if payload else {}))

    assert response.status_code == 501
    body = response.json()
    assert body["error"] == "not_implemented"
    assert body["implemented_in"]
    assert body["trace_id"]


def test_all_documented_endpoints_appear_in_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    expected = {
        "/health",
        "/metrics",
        "/chat",
        "/investigate",
        "/scenario",
        "/investigation/{investigation_id}",
        "/investigation/{investigation_id}/trace",
        "/models",
        "/feedback",
    }
    assert expected <= set(paths)


# --- validation and errors ------------------------------------------------


def test_invalid_request_body_returns_structured_422(client: TestClient) -> None:
    response = client.post("/chat", json={"question": "x"})  # below min_length
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["trace_id"]


def test_unknown_path_returns_structured_error(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "http_error"
