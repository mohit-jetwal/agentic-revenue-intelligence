"""The ForecastingTool contract (brief sections 20, 39).

The tool is what a Claude agent will call in Step 16, and the agent cannot read
source code - it sees only the JSON schema and the envelope. So these tests
check the *contract*, not the model: that failures arrive as readable results
rather than exceptions, that a refusal says what would have worked, and that the
agent is never handed a number without the context needed to use it responsibly.

The most important test here is the one asserting ``run()`` never raises. A tool
that throws takes the agent loop down with it; a tool that returns a structured
error lets the supervisor re-plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.tool_contract import ToolErrorCode, ToolResult, ToolStatus
from app.services.forecast_service import ForecastingService
from app.tools.forecasting_tool import ForecastingTool

pytestmark = [pytest.mark.models, pytest.mark.unit]


@pytest.fixture
def untrained_tool(smoke_repository, tmp_path: Path) -> ForecastingTool:
    """A tool whose model directory is deliberately empty.

    Most of what matters about a tool contract is how it behaves when something
    is missing, and that is the path least likely to be exercised by hand.
    """
    return ForecastingTool(
        ForecastingService(smoke_repository, model_dir=tmp_path / "forecasting")
    )


class TestToolDeclaration:
    def test_declares_the_four_required_class_attributes(self) -> None:
        """``AnalyticalTool.__init_subclass__`` enforces these at import time."""
        assert ForecastingTool.name == "forecast_demand"
        assert ForecastingTool.description
        assert ForecastingTool.input_schema is not None
        assert ForecastingTool.output_schema is not None

    def test_spec_is_json_schema_the_llm_can_read(self, untrained_tool) -> None:
        spec = untrained_tool.spec()

        assert spec.name == "forecast_demand"
        assert spec.input_schema["type"] == "object"
        assert "forecast_horizon" in spec.input_schema["properties"]

    def test_description_states_what_the_tool_does_not_do(self) -> None:
        """The agent must not read a forecast as a causal claim.

        A forecast says what is likely given the plan. Attributing a change to a
        promotion is the uplift model's job, and an agent that conflates them
        will confidently credit a seasonal peak to whatever campaign was live.
        """
        description = ForecastingTool.description.lower()

        assert "predictive" in description
        assert "cause" in description or "caused" in description

    def test_requires_the_run_model_permission(self) -> None:
        """Least privilege: a caller without it never sees the tool advertised."""
        assert ForecastingTool.permission == "run_model"


class TestToolFailureBehaviour:
    def test_run_never_raises_on_a_missing_model(self, untrained_tool) -> None:
        """The property the whole envelope exists for.

        An exception here would propagate into the agent loop. A structured
        result lets the supervisor decide what to do instead.
        """
        result = untrained_tool.run({"product_id": "P00001", "forecast_horizon": 30})

        assert isinstance(result, ToolResult)
        assert not result.ok
        assert result.error is not None
        assert result.error.code == ToolErrorCode.MODEL_NOT_FOUND

    def test_a_missing_model_is_not_recoverable(self, untrained_tool) -> None:
        """No different request produces a forecast until someone trains a
        model, so an agent retrying with a narrower slice would only burn
        tokens."""
        result = untrained_tool.run({"forecast_horizon": 30})

        assert result.error is not None
        assert result.error.recoverable is False

    def test_an_unsupported_horizon_is_refused_with_the_supported_set(
        self, untrained_tool
    ) -> None:
        """Refuses rather than snapping to the nearest horizon.

        The model is trained and calibrated at 7/14/30/90. Quietly serving 45
        days as 30 would return a number whose interval does not describe it -
        and the agent would have no way to know.
        """
        result = untrained_tool.run({"forecast_horizon": 45})

        assert result.status == ToolStatus.INVALID_INPUT or not result.ok
        assert result.error is not None
        assert result.error.code == ToolErrorCode.INVALID_INPUT
        assert result.error.recoverable is True
        assert "supported_horizons" in result.error.detail

    def test_malformed_input_is_rejected_before_execution(self, untrained_tool) -> None:
        result = untrained_tool.run({"forecast_horizon": "not a number"})

        assert result.status == ToolStatus.INVALID_INPUT
        assert result.error is not None
        assert result.error.code == ToolErrorCode.INVALID_INPUT

    def test_execution_time_is_recorded_even_on_failure(self, untrained_tool) -> None:
        result = untrained_tool.run({"forecast_horizon": 30})

        assert result.execution_time_ms >= 0

    def test_failures_carry_a_trace_id(self, untrained_tool) -> None:
        """Every result is traceable back to a log line."""
        result = untrained_tool.run({"forecast_horizon": 30})

        assert result.trace_id


class TestTrainedTool:
    """Against a real trained model, when one exists."""

    @pytest.fixture
    def trained_tool(self, smoke_repository) -> ForecastingTool:
        directory = Path("data/local/models/forecasting_sampled")
        if not (directory / "model.joblib").is_file():
            pytest.skip(
                "no trained forecaster; run "
                "`uv run python scripts/train_forecast.py --smoke`"
            )
        # The real dataset's repository, since the model was trained on it.
        from app.services.container import Container

        service = ForecastingService(Container().data_repository, model_dir=directory)
        return ForecastingTool(service)

    def test_successful_forecast_carries_provenance(self, trained_tool) -> None:
        """The agent must be able to say which model produced a number."""
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {
                "product_id": pair.product_id,
                "store_id": pair.store_id,
                "forecast_horizon": 30,
            }
        )

        assert result.ok, result.error
        assert result.model_name
        assert result.model_version
        assert result.dataset_version

    def test_result_includes_measured_accuracy_not_just_a_number(
        self, trained_tool
    ) -> None:
        """A forecast without its error record is not actionable.

        The agent needs to know the model's normal error at this horizon before
        it can say whether a 3% movement means anything.
        """
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {"product_id": pair.product_id, "store_id": pair.store_id, "forecast_horizon": 30}
        )

        assert result.ok
        accuracy = result.result["accuracy"]
        assert "wmape_by_horizon" in accuracy

    def test_confidence_is_measured_coverage_not_invented(self, trained_tool) -> None:
        """Section 18: do not fabricate confidence.

        Either it is the interval's measured nominal coverage, or it is absent.
        There is no third option where a plausible-looking number is produced
        because the field exists.
        """
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {"product_id": pair.product_id, "store_id": pair.store_id, "forecast_horizon": 7}
        )

        assert result.ok
        if result.confidence is not None:
            assert 0.0 <= result.confidence <= 1.0
            # It is coverage, so it should sit near the nominal level rather
            # than at some arbitrary "high confidence" value.
            assert result.confidence >= 0.5

    def test_assumptions_travel_with_the_forecast(self, trained_tool) -> None:
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {"product_id": pair.product_id, "store_id": pair.store_id, "forecast_horizon": 30}
        )

        assert result.ok
        joined = " ".join(result.assumptions).lower()
        assert "demand" in joined
        # The caveat an agent is most likely to drop.
        assert "causal" in joined or "cause" in joined

    def test_beyond_history_is_refused_with_a_usable_alternative(
        self, trained_tool
    ) -> None:
        """The refusal must tell the agent what would have worked.

        Without the boundary, a supervisor can only give up; with it, it can
        re-plan onto a valid as-of.
        """
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {
                "product_id": pair.product_id,
                "store_id": pair.store_id,
                "forecast_horizon": 90,
                "as_of_date": "2025-12-15",
            }
        )

        assert not result.ok
        assert result.error is not None
        assert result.error.code == ToolErrorCode.INSUFFICIENT_DATA
        assert result.error.recoverable is True
        assert "latest" in result.error.message.lower()

    def test_for_llm_omits_internals(self, trained_tool) -> None:
        """What actually reaches the model's context.

        The agent should not need to know that LightGBM exists, where the
        parquet lives, or what MLflow is.
        """
        model = trained_tool._service.model
        pair = model.pairs.iloc[0]

        result = trained_tool.run(
            {"product_id": pair.product_id, "store_id": pair.store_id, "forecast_horizon": 30}
        )
        payload = result.for_llm()

        assert "result" in payload
        assert "created_at" not in payload
        serialised = str(payload).lower()
        assert "parquet" not in serialised
        assert "mlflow" not in serialised
