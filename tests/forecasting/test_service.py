"""The forecasting service (brief sections 19, 29).

Mostly about failure, because the success path is the easy half. By Step 16 a
supervisor agent decides what to do when a call does not work, and it can only
decide from a failure it can read: a code, a message, and a ``recoverable`` flag
telling it whether a different request could succeed.

Section 29's validation list is covered here - horizon validity, as-of validity,
model presence - along with the fallback contract from section 28.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.schemas.domain import ForecastHorizon
from app.schemas.forecast import ForecastErrorResponse, ForecastRequest, ForecastResponse
from app.services.forecast_service import ForecastingService

pytestmark = pytest.mark.models


@pytest.fixture
def untrained_service(smoke_repository, tmp_path: Path) -> ForecastingService:
    return ForecastingService(smoke_repository, model_dir=tmp_path / "forecasting")


class TestMissingModel:
    def test_is_available_answers_without_raising(self, untrained_service) -> None:
        """The health endpoint and the DI container both call this at startup,
        when no model may exist yet."""
        assert untrained_service.is_available is False

    def test_forecast_returns_a_structured_error(self, untrained_service) -> None:
        response = untrained_service.forecast(ForecastRequest(horizon=ForecastHorizon.D30))

        assert isinstance(response, ForecastErrorResponse)
        assert response.error_code == "model_not_found"

    def test_missing_model_is_not_recoverable(self, untrained_service) -> None:
        response = untrained_service.forecast(ForecastRequest())

        assert isinstance(response, ForecastErrorResponse)
        assert response.recoverable is False

    def test_health_check_says_how_to_fix_it(self, untrained_service) -> None:
        healthy, detail = untrained_service.health_check()

        assert healthy is False
        assert "train_forecast" in detail

    def test_execution_time_recorded_on_failure(self, untrained_service) -> None:
        response = untrained_service.forecast(ForecastRequest())

        assert response.execution_time_ms >= 0


class TestTrainedService:
    """Against the real trained model, when one exists."""

    @pytest.fixture
    def service(self) -> ForecastingService:
        directory = Path("data/local/models/forecasting_sampled")
        if not (directory / "model.joblib").is_file():
            pytest.skip(
                "no trained forecaster; run "
                "`uv run python scripts/train_forecast.py --smoke`"
            )
        from app.services.container import Container

        return ForecastingService(Container().data_repository, model_dir=directory)

    def test_forecasts_a_single_series(self, service) -> None:
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D30,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
            )
        )

        assert isinstance(response, ForecastResponse)
        assert response.total_predicted_units > 0
        assert response.horizon_days == 30
        assert len(response.points) == 30

    def test_every_horizon_is_servable(self, service) -> None:
        """Section 1 asks for 7/14/30/90; all four must actually work."""
        pair = service.model.pairs.iloc[0]

        for horizon in ForecastHorizon:
            response = service.forecast(
                ForecastRequest(
                    horizon=horizon,
                    product_ids=[pair.product_id],
                    store_ids=[pair.store_id],
                )
            )

            assert isinstance(response, ForecastResponse), horizon
            assert len(response.points) == horizon.days

    def test_longer_horizons_forecast_more_units(self, service) -> None:
        """A sanity check on the totals: 90 days should exceed 7 days.

        Cheap, but it would catch a horizon that silently forecast the same
        window regardless of the argument.
        """
        pair = service.model.pairs.iloc[0]
        totals = {}
        for horizon in (ForecastHorizon.D7, ForecastHorizon.D90):
            response = service.forecast(
                ForecastRequest(
                    horizon=horizon,
                    product_ids=[pair.product_id],
                    store_ids=[pair.store_id],
                )
            )
            totals[horizon] = response.total_predicted_units

        assert totals[ForecastHorizon.D90] > totals[ForecastHorizon.D7]

    def test_predictions_are_never_negative(self, service) -> None:
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D90,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
            )
        )

        assert all(point.predicted_units >= 0 for point in response.points)
        assert all(
            point.lower_bound is None or point.lower_bound >= 0 for point in response.points
        )

    def test_beyond_history_is_refused(self, service) -> None:
        """Section 29: as_of must be valid.

        The data ends 2025-12-31, so a 90-day horizon from December has no
        promotion calendar. Refusing beats assuming "no promotions planned",
        which would bias those days low while looking like a real forecast.
        """
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D90,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
                as_of_date=date(2025, 12, 15),
            )
        )

        assert isinstance(response, ForecastErrorResponse)
        assert response.error_code == "insufficient_data"
        assert response.recoverable is True

    def test_refusal_names_the_latest_workable_as_of(self, service) -> None:
        """A recoverable error must tell the caller what would work."""
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D90,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
                as_of_date=date(2025, 12, 15),
            )
        )

        assert isinstance(response, ForecastErrorResponse)
        assert "latest" in response.message.lower()

    def test_unknown_product_is_reported_not_silently_empty(self, service) -> None:
        """Returning an empty forecast would read as "no demand expected",
        which is a very different claim from "this product is not in the
        model"."""
        response = service.forecast(
            ForecastRequest(horizon=ForecastHorizon.D30, product_ids=["NOT_A_PRODUCT"])
        )

        assert isinstance(response, ForecastErrorResponse)
        assert response.error_code == "insufficient_data"

    def test_provenance_is_populated(self, service) -> None:
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D7,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
            )
        )

        assert isinstance(response, ForecastResponse)
        assert response.model_name
        assert response.model_version
        assert response.dataset_version
        assert response.feature_version

    def test_assumptions_state_the_competitor_limitation(self, service) -> None:
        """Competitor prices are OBSERVED, so they cannot be known for a future
        date. The response has to say that rather than let a reader assume the
        forecast anticipates competitor moves."""
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D30,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
            )
        )

        joined = " ".join(response.assumptions).lower()
        assert "competitor" in joined

    def test_confidence_is_absent_or_meaningful(self, service) -> None:
        """Never a fabricated number: it is measured nominal coverage or None."""
        pair = service.model.pairs.iloc[0]

        response = service.forecast(
            ForecastRequest(
                horizon=ForecastHorizon.D30,
                product_ids=[pair.product_id],
                store_ids=[pair.store_id],
            )
        )

        assert isinstance(response, ForecastResponse)
        if response.confidence is not None:
            assert response.confidence == response.accuracy.interval_nominal

    def test_health_check_names_the_estimator(self, service) -> None:
        healthy, detail = service.health_check()

        assert healthy is True
        assert detail
