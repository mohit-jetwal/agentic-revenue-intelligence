"""The API, the app-state store and the trace, end to end.

Runs against the stub provider through a real `TestClient`, so these exercise
the same path a browser would rather than calling the service directly. An
endpoint that broke its contract while the service underneath still worked is
exactly the failure a direct-call test cannot see.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.agents.critic import CriticAssessment
from app.agents.recommendation import DraftRecommendation
from app.agents.supervisor import IntentClassification, PlannedStep, ProposedPlan
from app.api.app import create_app
from app.config.settings import DataSettings
from app.schemas.api import InvestigationStatus
from app.schemas.domain import BusinessObjective, IntentType
from app.services.container import Container, set_container
from app.store.investigations import InvestigationStore
from tests.conftest import build_settings

pytestmark = pytest.mark.integration

PROFIT = 2_000_000.0


@pytest.fixture
def container(tmp_path, monkeypatch) -> Iterator[Container]:
    """A container on a throwaway database with a scripted provider."""
    monkeypatch.setenv("LLM__PROVIDER", "stub")
    settings = build_settings(
        data=DataSettings(
            _env_file=None,
            app_database_url=f"sqlite:///{tmp_path / 'app_state.sqlite'}",
        )
    )
    built = Container(settings)

    stub = built.llm_provider
    stub.script_structured(  # type: ignore[attr-defined]
        IntentClassification(
            intent=IntentType.PROMOTION_DECISION,
            objective=BusinessObjective.MAXIMISE_PROFIT,
            entities={"products": ["P00091"]},
        )
    )
    stub.script_structured(  # type: ignore[attr-defined]
        ProposedPlan(
            steps=[
                PlannedStep(
                    tool_name="estimate_promo_uplift",
                    rationale="measure incrementality",
                    parameters={"product_id": "P00091"},
                )
            ]
        )
    )
    stub.script_structured(CriticAssessment(sufficient=True, confidence=0.8))  # type: ignore[attr-defined]
    stub.script_structured(  # type: ignore[attr-defined]
        DraftRecommendation(
            executive_summary="The promotion produced measurable incremental volume.",
            recommended_action="Continue funding it at the current depth.",
            confidence=0.8,
            estimated_profit_impact=PROFIT,
        )
    )

    set_container(built)
    yield built
    set_container(None)


@pytest.fixture
def client(container: Container) -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def store(tmp_path) -> InvestigationStore:
    return InvestigationStore(f"sqlite:///{tmp_path / 'store.sqlite'}")


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


class TestStore:
    def test_an_investigation_is_recorded_before_it_runs(
        self, store: InvestigationStore
    ) -> None:
        """Written first, not on success. An investigation that crashes is
        exactly the one worth having a record of."""
        store.create(investigation_id="i1", trace_id="t1", question="why?")
        found = store.get("i1")

        assert found is not None
        assert found.status is InvestigationStatus.RUNNING

    def test_a_missing_investigation_reads_as_none(
        self, store: InvestigationStore
    ) -> None:
        assert store.get("nope") is None

    def test_completing_stores_the_recommendation(
        self, store: InvestigationStore
    ) -> None:
        store.create(investigation_id="i1", trace_id="t1", question="why?")
        store.complete(
            "i1",
            status=InvestigationStatus.COMPLETED,
            intent="promotion_decision",
            recommendation={
                "executive_summary": "s",
                "recommended_action": "a",
                "confidence": 0.7,
            },
            tool_calls=2,
        )
        found = store.get("i1")

        assert found is not None
        assert found.recommendation is not None
        assert found.recommendation.confidence == 0.7
        assert found.intent is IntentType.PROMOTION_DECISION

    def test_an_unreadable_stored_recommendation_does_not_break_the_read(
        self, store: InvestigationStore
    ) -> None:
        """A row written by an older model version must not 500 the endpoint -
        it should surface as an investigation with no recommendation, which is
        what it now is."""
        store.create(investigation_id="i1", trace_id="t1", question="why?")
        store.complete(
            "i1",
            status=InvestigationStatus.COMPLETED,
            recommendation={"nonsense": True},
        )
        found = store.get("i1")

        assert found is not None
        assert found.recommendation is None

    def test_completing_an_unknown_investigation_is_ignored(
        self, store: InvestigationStore
    ) -> None:
        store.complete("ghost", status=InvestigationStatus.COMPLETED)

        assert store.get("ghost") is None

    def test_feedback_outlives_a_purged_investigation(
        self, store: InvestigationStore
    ) -> None:
        """The only human-labelled signal the platform produces. A cascade
        delete would destroy the scarcest data here."""
        store.create(investigation_id="i1", trace_id="t1", question="why?")
        store.add_feedback(investigation_id="i1", helpful=True, rating=5)

        assert store.purge("i1")
        assert store.get("i1") is None
        assert len(store.feedback_for("i1")) == 1

    def test_trace_events_keep_their_order(self, store: InvestigationStore) -> None:
        from datetime import UTC, datetime

        from app.schemas.api import TraceEvent

        store.create(investigation_id="i1", trace_id="t1", question="why?")
        store.append_events(
            "i1",
            [
                TraceEvent(
                    sequence=n,
                    timestamp=datetime.now(UTC),
                    event_type="observation",
                    actor="supervisor",
                    summary=f"step {n}",
                )
                for n in (1, 2, 3)
            ],
        )
        trace = store.get_trace("i1")

        assert trace is not None
        assert [e.sequence for e in trace.events] == [1, 2, 3]


# --------------------------------------------------------------------------
# The endpoints
# --------------------------------------------------------------------------


class TestChat:
    def test_a_question_produces_a_recommendation(self, client: TestClient) -> None:
        response = client.post("/chat", json={"question": "Did the promotion work?"})

        assert response.status_code == 200
        body = response.json()
        assert body["recommendation"]["recommended_action"]
        assert body["investigation_id"]

    def test_the_approval_flag_survives_the_round_trip(
        self, client: TestClient
    ) -> None:
        """The flag a caller must check before acting. Losing it in
        serialisation would turn a gated recommendation into a cleared one."""
        response = client.post("/chat", json={"question": "Did the promotion work?"})

        assert response.json()["recommendation"]["requires_human_approval"]

    def test_a_too_short_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/chat", json={"question": "x"}).status_code == 422


class TestInvestigationRetrieval:
    def test_an_investigation_can_be_fetched_after_the_fact(
        self, client: TestClient
    ) -> None:
        created = client.post("/chat", json={"question": "Did the promotion work?"})
        investigation_id = created.json()["investigation_id"]

        response = client.get(f"/investigation/{investigation_id}")

        assert response.status_code == 200
        assert response.json()["question"] == "Did the promotion work?"

    def test_a_recommendation_above_the_threshold_awaits_approval(
        self, client: TestClient
    ) -> None:
        """A distinct status, not a flag on a completed one. From the business's
        point of view it has not finished."""
        created = client.post("/chat", json={"question": "Did the promotion work?"})
        response = client.get(f"/investigation/{created.json()['investigation_id']}")

        assert response.json()["status"] == "awaiting_approval"

    def test_an_unknown_id_is_a_404(self, client: TestClient) -> None:
        assert client.get("/investigation/nope").status_code == 404
        assert client.get("/investigation/nope/trace").status_code == 404


class TestTrace:
    def test_the_trace_records_the_whole_investigation(
        self, client: TestClient
    ) -> None:
        created = client.post("/chat", json={"question": "Did the promotion work?"})
        trace = client.get(
            f"/investigation/{created.json()['investigation_id']}/trace"
        ).json()

        types = [event["event_type"] for event in trace["events"]]
        assert "intent_classified" in types
        assert "plan_created" in types
        assert "critic_verdict" in types
        assert "recommendation" in types

    def test_events_are_sequenced(self, client: TestClient) -> None:
        created = client.post("/chat", json={"question": "Did the promotion work?"})
        trace = client.get(
            f"/investigation/{created.json()['investigation_id']}/trace"
        ).json()

        sequences = [event["sequence"] for event in trace["events"]]
        assert sequences == sorted(sequences)
        assert sequences[0] == 1


class TestScenario:
    def _request(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "description": "Cut the price by 5%",
            "product_ids": ["P00091"],
            "price_change_pct": -5.0,
        }
        payload.update(overrides)
        return payload

    def test_a_price_lever_projects(self, client: TestClient) -> None:
        response = client.post("/scenario", json=self._request())

        assert response.status_code == 200
        assert "profit_impact" in response.json()["result"]

    def test_an_unmodellable_lever_is_reported_not_dropped(
        self, client: TestClient
    ) -> None:
        """A projection that silently ignored the inventory change the caller
        asked about would answer a different question than the one posed."""
        response = client.post(
            "/scenario", json=self._request(inventory_change_pct=10.0)
        )

        warnings = response.json()["warnings"]
        assert any("inventory_change_pct was ignored" in w for w in warnings)

    def test_extra_products_are_reported(self, client: TestClient) -> None:
        response = client.post(
            "/scenario", json=self._request(product_ids=["P00091", "P00119"])
        )

        assert any("only P00091" in w for w in response.json()["warnings"])

    def test_no_modellable_lever_is_rejected(self, client: TestClient) -> None:
        """Better than projecting nothing and returning a zero, which would read
        as 'this change has no effect'."""
        response = client.post(
            "/scenario",
            json={
                "description": "Spend more on promotion",
                "product_ids": ["P00091"],
                "promotion_spend_change": 5000.0,
            },
        )

        assert response.status_code == 422
        assert "modellable lever" in response.text

    def test_no_product_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/scenario", json={"description": "Cut prices", "price_change_pct": -5.0}
        )

        assert response.status_code == 422


class TestFeedback:
    def test_feedback_is_accepted(self, client: TestClient) -> None:
        created = client.post("/chat", json={"question": "Did the promotion work?"})
        response = client.post(
            "/feedback",
            json={
                "investigation_id": created.json()["investigation_id"],
                "helpful": True,
                "rating": 5,
            },
        )

        assert response.status_code == 200
        assert response.json()["received"]

    def test_feedback_on_an_unknown_investigation_is_still_kept(
        self, client: TestClient
    ) -> None:
        """Rejecting it because a demo database was reset would discard the
        scarcest data this platform produces."""
        response = client.post(
            "/feedback", json={"investigation_id": "gone", "helpful": False}
        )

        assert response.status_code == 200


class TestModels:
    def test_an_empty_registry_answers_honestly(self, client: TestClient) -> None:
        """"None registered yet" rather than a 501 - the question was answerable
        and the answer is zero."""
        response = client.get("/models")

        assert response.status_code == 200
        assert response.json()["models"] == []


class TestHealth:
    def test_the_app_store_is_probed(self, client: TestClient) -> None:
        names = [d["name"] for d in client.get("/health").json()["dependencies"]]

        assert "investigation_store" in names
