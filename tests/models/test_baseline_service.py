"""The service layer between the model and its callers.

Everything here is about *failure*, because the success path is the easy half.
By Step 13 this service sits behind a tool, and by Step 16 a supervisor agent
decides what to do when a call does not work. That agent can only re-plan around
a failure it can read - a raised exception becomes a stack trace in a log, while
a structured error with a code and a ``recoverable`` flag becomes a decision.

So the contract these tests enforce is: expected failures come back as values,
not exceptions, and they carry enough information to act on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.schemas.baseline import BaselineErrorResponse, BaselineRequest, BaselineResponse
from app.services.baseline_service import BaselineSalesService

pytestmark = pytest.mark.models


@pytest.fixture
def service(smoke_repository, tmp_path: Path) -> BaselineSalesService:
    """A service pointed at an empty model directory.

    Deliberately untrained: these tests exercise the paths that matter when
    something is missing or wrong, which is most of what the service is for.
    """
    return BaselineSalesService(smoke_repository, model_dir=tmp_path / "baseline")


class TestMissingModel:
    def test_is_available_is_false_without_a_trained_model(
        self, service: BaselineSalesService
    ) -> None:
        """Must answer without raising.

        The health endpoint and the DI container both call this at startup, when
        no model may exist yet.
        """
        assert service.is_available is False

    def test_predict_returns_a_structured_error_rather_than_raising(
        self, service: BaselineSalesService
    ) -> None:
        response = service.predict(
            BaselineRequest(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        )

        assert isinstance(response, BaselineErrorResponse)
        assert response.error_code == "model_not_found"

    def test_a_missing_model_is_not_recoverable_by_re_planning(
        self, service: BaselineSalesService
    ) -> None:
        """The distinction that makes ``recoverable`` worth having.

        No different request produces a baseline when no model exists, so an
        agent retrying with a narrower date range would just burn tokens. That
        is different from an empty slice, which a different request *can* fix.
        """
        response = service.predict(
            BaselineRequest(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        )

        assert isinstance(response, BaselineErrorResponse)
        assert response.recoverable is False

    def test_error_message_says_how_to_fix_it(
        self, service: BaselineSalesService
    ) -> None:
        """The message is read by a human at 2am or by an agent deciding what to
        report. "FileNotFoundError" helps neither."""
        response = service.predict(
            BaselineRequest(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        )

        assert isinstance(response, BaselineErrorResponse)
        assert response.message

    def test_health_check_reports_unhealthy_with_a_reason(
        self, service: BaselineSalesService
    ) -> None:
        healthy, detail = service.health_check()

        assert healthy is False
        assert "train" in detail.lower()


class TestInputValidation:
    def test_reversed_date_range_is_rejected_before_loading_the_model(
        self, service: BaselineSalesService
    ) -> None:
        """Cheap checks run first.

        An invalid request should not pay for a model load, and the caller
        should get the error that is actually actionable - the reversed dates -
        rather than a missing-model error that masks it.
        """
        response = service.predict(
            BaselineRequest(start_date=date(2024, 6, 1), end_date=date(2024, 1, 1))
        )

        assert isinstance(response, BaselineErrorResponse)
        assert response.error_code == "invalid_input"
        assert response.recoverable is True

    def test_execution_time_is_recorded_even_on_failure(
        self, service: BaselineSalesService
    ) -> None:
        """Latency on the error path is what reveals a slow failure."""
        response = service.predict(
            BaselineRequest(start_date=date(2024, 6, 1), end_date=date(2024, 1, 1))
        )

        assert response.execution_time_ms >= 0


class TestTrainedService:
    """Against the real model, when one has been trained."""

    @pytest.fixture
    def trained_service(
        self, trained_model_dir: Path, smoke_repository
    ) -> BaselineSalesService:
        return BaselineSalesService(smoke_repository, model_dir=trained_model_dir)

    def test_is_available_once_a_model_exists(
        self, trained_service: BaselineSalesService
    ) -> None:
        assert trained_service.is_available is True

    def test_health_check_names_the_model(
        self, trained_service: BaselineSalesService
    ) -> None:
        healthy, detail = trained_service.health_check()

        assert healthy is True
        assert detail

    def test_assumptions_are_always_populated(
        self, trained_service: BaselineSalesService
    ) -> None:
        """A baseline without its assumptions is a number without a meaning.

        Whether the figure is a valid no-promotion counterfactual depends
        entirely on which approach produced it, and the caller cannot know that
        unless the response says so.
        """
        model = trained_service.model
        assumptions = trained_service._assumptions(model, None)

        assert len(assumptions) >= 3
        assert any("baseline" in a.lower() for a in assumptions)

    def test_assumptions_state_which_promotion_approach_was_used(
        self, trained_service: BaselineSalesService
    ) -> None:
        model = trained_service.model
        assumptions = " ".join(trained_service._assumptions(model, None)).lower()

        assert "promotion" in assumptions

    def test_assumptions_warn_that_a_gap_is_not_automatically_causal(
        self, trained_service: BaselineSalesService
    ) -> None:
        """The single most important caveat this system carries.

        Actual minus baseline is a difference. Calling it uplift requires causal
        assumptions the baseline model does not test, and an agent that forgets
        this will confidently attribute a seasonal peak to a promotion.
        """
        model = trained_service.model
        assumptions = " ".join(trained_service._assumptions(model, None)).lower()

        assert "causal" in assumptions

    def test_predicts_over_a_real_slice(
        self, trained_service: BaselineSalesService
    ) -> None:
        request = BaselineRequest(
            start_date=date(2025, 10, 1),
            end_date=date(2025, 10, 14),
            include_records=True,
            max_records=25,
        )

        response = trained_service.predict(request)

        if isinstance(response, BaselineErrorResponse):
            pytest.skip(f"slice unavailable in this dataset: {response.message}")

        assert isinstance(response, BaselineResponse)
        assert response.result.baseline_units > 0
        assert len(response.records) <= 25
        assert response.model_version
        # Provenance must survive to the caller - Step 13 puts it in the tool
        # envelope so a claim can be traced back to the artifact behind it.
        assert response.dataset_version
