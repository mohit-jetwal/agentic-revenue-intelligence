"""The tool contract - seam 1.

The property under test is that ``AnalyticalTool.run`` *always* returns a
well-formed :class:`ToolResult`, whatever the tool does. This is the mechanism
behind the platform's central safety claim (Claude interprets numbers but never
produces them): if a tool could raise, or return a bare value, or omit its model
version, that claim would rest on convention rather than on code.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.observability.context import trace_context
from app.schemas.tool_contract import (
    ModelProvenance,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)
from app.tools.base import AnalyticalTool, ToolExecutionError, ToolOutput

pytestmark = pytest.mark.unit


class _Input(BaseModel):
    product_id: str
    horizon_days: int = Field(gt=0)


class _Output(BaseModel):
    predicted_units: float
    method: str


class _GoodTool(AnalyticalTool[_Input, _Output]):
    name = "good_tool"
    description = "Returns a deterministic number."
    input_schema = _Input
    output_schema = _Output

    def _execute(self, payload: _Input) -> ToolOutput[_Output]:
        return ToolOutput(
            payload=_Output(predicted_units=42.0, method="test"),
            provenance=ModelProvenance(
                model_name="test_model",
                model_version="v1.2",
                dataset_version="ds-2026-01",
            ),
            confidence=0.87,
            assumptions=["Demand is stationary over the horizon."],
        )


class _WarningTool(AnalyticalTool[_Input, _Output]):
    name = "warning_tool"
    description = "Succeeds with a caveat."
    input_schema = _Input
    output_schema = _Output

    def _execute(self, payload: _Input) -> ToolOutput[_Output]:
        return ToolOutput(
            payload=_Output(predicted_units=1.0, method="test"),
            warnings=["Only 14 observations available."],
        )


class _TypedFailureTool(AnalyticalTool[_Input, _Output]):
    name = "typed_failure_tool"
    description = "Fails in an anticipated way."
    input_schema = _Input
    output_schema = _Output

    def _execute(self, payload: _Input) -> ToolOutput[_Output]:
        raise ToolExecutionError(
            "Not enough history for this product.",
            code=ToolErrorCode.INSUFFICIENT_DATA,
            recoverable=True,
            detail={"observations": 3},
        )


class _ExplodingTool(AnalyticalTool[_Input, _Output]):
    name = "exploding_tool"
    description = "Fails in an unanticipated way."
    input_schema = _Input
    output_schema = _Output

    def _execute(self, payload: _Input) -> ToolOutput[_Output]:
        return _Output(predicted_units=1 / 0, method="boom")  # type: ignore[return-value]


# --- success path ---------------------------------------------------------


def test_success_produces_full_envelope() -> None:
    result = _GoodTool().run({"product_id": "P1", "horizon_days": 30})

    assert isinstance(result, ToolResult)
    assert result.status is ToolStatus.SUCCESS
    assert result.ok is True
    assert result.tool_name == "good_tool"
    assert result.result == {"predicted_units": 42.0, "method": "test"}
    assert result.confidence == 0.87
    assert result.assumptions == ["Demand is stationary over the horizon."]
    assert result.error is None


def test_success_carries_model_provenance() -> None:
    """Every number must be attributable to a versioned model and dataset."""
    result = _GoodTool().run({"product_id": "P1", "horizon_days": 30})
    assert result.model_name == "test_model"
    assert result.model_version == "v1.2"
    assert result.dataset_version == "ds-2026-01"


def test_execution_time_is_recorded() -> None:
    result = _GoodTool().run({"product_id": "P1", "horizon_days": 30})
    assert result.execution_time_ms >= 0


def test_warnings_downgrade_status_to_partial() -> None:
    """A caveated result must not look identical to a clean one."""
    result = _WarningTool().run({"product_id": "P1", "horizon_days": 7})
    assert result.status is ToolStatus.PARTIAL
    assert result.ok is True
    assert result.warnings == ["Only 14 observations available."]


def test_validated_input_object_is_accepted() -> None:
    result = _GoodTool().run(_Input(product_id="P1", horizon_days=30))
    assert result.status is ToolStatus.SUCCESS


# --- failure paths --------------------------------------------------------


def test_invalid_input_is_reported_not_raised() -> None:
    result = _GoodTool().run({"product_id": "P1", "horizon_days": -5})

    assert result.status is ToolStatus.INVALID_INPUT
    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT
    assert result.error.recoverable is True


def test_missing_required_field_is_reported_not_raised() -> None:
    result = _GoodTool().run({"horizon_days": 30})
    assert result.status is ToolStatus.INVALID_INPUT
    assert result.result == {}


def test_typed_failure_is_recoverable_and_keeps_its_code() -> None:
    """Re-planning keys off the error code, so it must survive the wrapper."""
    result = _TypedFailureTool().run({"product_id": "P1", "horizon_days": 30})

    assert result.status is ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INSUFFICIENT_DATA
    assert result.error.recoverable is True
    assert result.is_recoverable_failure is True
    assert result.error.detail == {"observations": 3}


def test_unexpected_exception_never_escapes() -> None:
    """An unhandled fault must not kill the graph mid-investigation."""
    result = _ExplodingTool().run({"product_id": "P1", "horizon_days": 30})

    assert result.status is ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INTERNAL_ERROR
    assert result.error.recoverable is False
    assert result.is_recoverable_failure is False


def test_failure_envelope_is_still_well_formed() -> None:
    result = _ExplodingTool().run({"product_id": "P1", "horizon_days": 30})
    assert result.tool_name == "exploding_tool"
    assert result.trace_id
    assert result.execution_time_ms >= 0


# --- trace propagation ----------------------------------------------------


def test_trace_id_is_inherited_from_context() -> None:
    with trace_context("trace-abc"):
        result = _GoodTool().run({"product_id": "P1", "horizon_days": 30})
    assert result.trace_id == "trace-abc"


def test_trace_id_is_generated_when_absent() -> None:
    result = _GoodTool().run({"product_id": "P1", "horizon_days": 30})
    assert result.trace_id


# --- contract enforcement -------------------------------------------------


def test_subclass_missing_schemas_fails_at_definition_time() -> None:
    """A malformed tool must fail on import, not mid-investigation."""
    with pytest.raises(TypeError, match="input_schema"):

        class _Broken(AnalyticalTool[_Input, _Output]):  # type: ignore[type-arg]
            name = "broken"
            description = "Missing its schemas."

            def _execute(self, payload: _Input) -> ToolOutput[_Output]:  # pragma: no cover
                raise NotImplementedError


def test_spec_is_generated_from_the_input_schema() -> None:
    """The schema Claude sees and the schema we validate must be one source."""
    spec = _GoodTool.spec()
    assert spec.name == "good_tool"
    assert spec.input_schema == _Input.model_json_schema()
    assert "product_id" in spec.input_schema["properties"]


# --- LLM-facing view ------------------------------------------------------


def test_for_llm_includes_numbers_and_provenance() -> None:
    payload = _GoodTool().run({"product_id": "P1", "horizon_days": 30}).for_llm()

    assert payload["result"] == {"predicted_units": 42.0, "method": "test"}
    assert payload["model"] == "test_model:v1.2"
    assert payload["confidence"] == 0.87
    assert "assumptions" in payload


def test_for_llm_omits_internal_plumbing() -> None:
    """Keeps token cost down and stops internals leaking into the prompt."""
    payload = _GoodTool().run({"product_id": "P1", "horizon_days": 30}).for_llm()
    assert "created_at" not in payload
    assert "execution_time_ms" not in payload


def test_for_llm_surfaces_errors_with_recoverability() -> None:
    payload = _TypedFailureTool().run({"product_id": "P1", "horizon_days": 30}).for_llm()
    assert payload["error"]["code"] == "insufficient_data"
    assert payload["error"]["recoverable"] is True
