"""Service and tool contracts, including every documented refusal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.schemas.promo_uplift import UpliftErrorResponse, UpliftRequest, UpliftResponse
from app.services.promo_uplift_service import PromoUpliftService, latest_analysis_window
from app.tools.base import ToolExecutionError
from app.tools.promo_uplift_tool import PromoUpliftInput, PromoUpliftTool
from ml.promo_uplift.config import get_promo_uplift_config
from ml.promo_uplift.estimators import EffectEstimate
from ml.promo_uplift.model import FittedUpliftModel, UpliftArtifact

pytestmark = pytest.mark.integration


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "promotion_id": ["PR01", "PR02", "PR03"],
            "product_id": ["A", "A", "B"],
            "store_id": ["S1", "S2", "S1"],
            "start_date": pd.to_datetime(["2024-03-01", "2024-04-01", "2024-05-01"]),
            "end_date": pd.to_datetime(["2024-03-10", "2024-04-10", "2024-05-10"]),
            "treated_days": [10, 10, 10],
            "observed_units": [1000.0, 900.0, 500.0],
            "incremental_units": [150.0, 90.0, -20.0],
            "incremental_revenue": [1500.0, 900.0, -200.0],
            "incremental_profit": [400.0, 100.0, -300.0],
            "promotion_spend": [200.0, 250.0, 150.0],
            "roi": [2.0, 0.4, -2.0],
            "value_destroying": [False, False, True],
            "region": ["North", "North", "South"],
        }
    )


def _artifact(*, status: str = "passed") -> UpliftArtifact:
    config = get_promo_uplift_config()
    estimate = EffectEstimate(
        method="augmented_ipw",
        ate=15.0,
        ate_pct=0.18,
        baseline_units=83.0,
        n_treated=30,
        n_control=400,
        standard_error=2.0,
        ci_lower=11.0,
        ci_upper=19.0,
        confidence_level=0.95,
        assumptions=["Treatment is ignorable given the pre-treatment covariates."],
    )
    naive = EffectEstimate(
        method="naive_during_vs_before",
        ate=40.0,
        ate_pct=0.55,
        baseline_units=73.0,
        n_treated=30,
        n_control=0,
    )
    return UpliftArtifact(
        config=config,
        estimates={"naive_during_vs_before": naive, "augmented_ipw": estimate},
        selected="augmented_ipw",
        selection_reason="weakest identifying assumptions among those that passed",
        validation_status=status,
        warnings=["stockout censoring is differential"],
        segments={
            "region": pd.DataFrame(
                {
                    "segment": ["North", "South"],
                    "n_treated": [200, 150],
                    "uplift_pct": [0.22, -0.04],
                    "baseline": [80.0, 70.0],
                    "standard_error": [0.01, 0.01],
                    "classification": ["high_uplift", "negative"],
                    "action": ["candidate for more investment", "stop"],
                    "estimable": [True, True],
                }
            )
        },
        cate_model=None,
        feature_names=("demand_mean_28",),
        categorical_names=(),
        treatment_definition="treated = a promotion of any mechanic with depth >= 5%",
        config_fingerprint=config.fingerprint(),
        dataset_version="test-v1",
        trained_at="2026-08-29T00:00:00+00:00",
    )


@pytest.fixture
def model(tmp_path: Path) -> FittedUpliftModel:
    return FittedUpliftModel(None, artifact=_artifact(), event_impact=_events())


@pytest.fixture
def service(model: FittedUpliftModel, tmp_path: Path) -> PromoUpliftService:
    return PromoUpliftService(None, model=model, model_dir=tmp_path)  # type: ignore[arg-type]


class TestSuccessfulEstimate:
    def test_aggregates_every_event(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert response.events_analysed == 3
        assert response.incremental_units == pytest.approx(220.0)
        assert response.promotion_spend == pytest.approx(600.0)

    def test_carries_the_treatment_definition(self, service: PromoUpliftService) -> None:
        """An uplift number is uninterpretable without it, so it is required
        rather than optional."""
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert response.treatment_definition

    def test_carries_provenance(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert response.model_name == "promo_uplift"
        assert response.dataset_version == "test-v1"
        assert response.feature_version

    def test_interval_is_present_when_measured(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert response.confidence_interval is not None
        assert response.confidence_interval.confidence_level == pytest.approx(0.95)

    def test_filters_to_one_promotion(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest(promotion_ids=["PR01"]))
        assert isinstance(response, UpliftResponse)
        assert response.events_analysed == 1
        assert response.incremental_units == pytest.approx(150.0)

    def test_filters_to_one_product(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest(product_ids=["A"]))
        assert isinstance(response, UpliftResponse)
        assert response.events_analysed == 2

    def test_events_are_ranked_by_roi(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert [e.promotion_id for e in response.events] == ["PR01", "PR02", "PR03"]

    def test_value_destroying_events_survive_the_filter(
        self, service: PromoUpliftService
    ) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert any(e.value_destroying for e in response.events)

    def test_a_subset_warns_that_diagnostics_describe_the_whole(
        self, service: PromoUpliftService
    ) -> None:
        response = service.estimate_uplift(UpliftRequest(promotion_ids=["PR01"]))
        assert isinstance(response, UpliftResponse)
        assert any("this response covers" in w for w in response.warnings)

    def test_comparison_marks_the_naive_method_ineligible(
        self, service: PromoUpliftService
    ) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        naive = next(
            m for m in response.comparison if m.method == "naive_during_vs_before"
        )
        assert not naive.eligible


class TestRefusals:
    def test_missing_artifact(self, tmp_path: Path) -> None:
        service = PromoUpliftService(None, model_dir=tmp_path / "absent")  # type: ignore[arg-type]
        response = service.estimate_uplift(UpliftRequest())

        assert isinstance(response, UpliftErrorResponse)
        assert response.error_code == "model_not_found"
        # No reformulation helps until someone runs an analysis.
        assert response.recoverable is False
        assert "estimate_uplift.py" in response.message

    def test_unknown_promotion_is_refused_not_answered_with_zero(
        self, service: PromoUpliftService
    ) -> None:
        """'This promotion had no effect' and 'this promotion is not in the
        analysis' are different findings."""
        response = service.estimate_uplift(UpliftRequest(promotion_ids=["NOPE"]))

        assert isinstance(response, UpliftErrorResponse)
        assert response.error_code == "insufficient_data"
        assert response.recoverable is True
        assert "NOPE" in response.message

    def test_unknown_product_yields_no_events(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest(product_ids=["ZZZ"]))
        assert isinstance(response, UpliftErrorResponse)
        assert response.error_code == "insufficient_data"

    def test_reversed_date_range_is_invalid_input(
        self, service: PromoUpliftService
    ) -> None:
        response = service.estimate_uplift(
            UpliftRequest(
                analysis_start_date=date(2024, 6, 1),
                analysis_end_date=date(2024, 1, 1),
            )
        )
        assert isinstance(response, UpliftErrorResponse)
        assert response.error_code == "invalid_input"

    def test_a_window_outside_the_analysis_is_recoverable(
        self, service: PromoUpliftService
    ) -> None:
        response = service.estimate_uplift(
            UpliftRequest(analysis_start_date=date(2030, 1, 1))
        )
        assert isinstance(response, UpliftErrorResponse)
        assert response.recoverable is True
        assert "re-run" in response.message


class TestValidationStatus:
    def test_passed_is_causal(self, service: PromoUpliftService) -> None:
        response = service.estimate_uplift(UpliftRequest())
        assert isinstance(response, UpliftResponse)
        assert response.is_causal

    def test_failed_still_returns_a_number_but_is_not_causal(
        self, tmp_path: Path
    ) -> None:
        """Withholding it does not protect anyone - the caller can compute a
        naive number in one line of SQL. The choice is between our number
        labelled and theirs unlabelled."""
        model = FittedUpliftModel(
            None, artifact=_artifact(status="failed"), event_impact=_events()
        )
        service = PromoUpliftService(None, model=model, model_dir=tmp_path)  # type: ignore[arg-type]
        response = service.estimate_uplift(UpliftRequest())

        assert isinstance(response, UpliftResponse)
        assert response.incremental_units != 0
        assert not response.is_causal
        assert response.validation_status == "failed"


class TestAnalysisWindow:
    def test_reports_the_covered_range(self, model: FittedUpliftModel) -> None:
        window = latest_analysis_window(model)
        assert window == (date(2024, 3, 1), date(2024, 5, 10))


class TestTool:
    def test_declaration_warns_the_agent_off_computing_it_itself(self) -> None:
        tool = PromoUpliftTool(None)  # type: ignore[arg-type]
        assert "Do NOT compute uplift yourself" in tool.description
        assert tool.permission == "run_model"

    def test_returns_evidence_not_just_a_number(
        self, service: PromoUpliftService
    ) -> None:
        result = PromoUpliftTool(service).run(PromoUpliftInput())
        assert result.status in {"success", "partial"}
        assert result.result is not None
        assert result.result["treatment_definition"]
        assert result.result["method_reason"]
        assert result.result["method_comparison"]

    def test_emits_no_invented_confidence_scalar(
        self, service: PromoUpliftService
    ) -> None:
        """The honest expression of uncertainty is the measured interval on the
        effect, which is in the payload. A 0-1 score would be a made-up summary
        of a quantity that was actually estimated."""
        result = PromoUpliftTool(service).run(PromoUpliftInput())
        assert result.confidence is None
        assert result.result is not None
        assert result.result["confidence_interval"] is not None

    def test_failed_validation_is_the_first_warning(self, tmp_path: Path) -> None:
        """A supervisor reading a truncated warning list must not miss the one
        that says the number is not causal."""
        model = FittedUpliftModel(
            None, artifact=_artifact(status="failed"), event_impact=_events()
        )
        service = PromoUpliftService(None, model=model, model_dir=tmp_path)  # type: ignore[arg-type]
        result = PromoUpliftTool(service).run(PromoUpliftInput())

        assert result.warnings
        assert "CAUSAL VALIDATION FAILED" in result.warnings[0]

    def test_a_refusal_becomes_a_tool_error(self, service: PromoUpliftService) -> None:
        result = PromoUpliftTool(service).run(PromoUpliftInput(promotion_id="NOPE"))
        assert result.status == "error"
        assert result.error is not None
        assert result.error.recoverable is True

    def test_missing_model_is_not_recoverable(self, tmp_path: Path) -> None:
        service = PromoUpliftService(None, model_dir=tmp_path / "absent")  # type: ignore[arg-type]
        result = PromoUpliftTool(service).run(PromoUpliftInput())

        assert result.status == "error"
        assert result.error is not None
        assert result.error.recoverable is False

    def test_execute_raises_a_typed_tool_error(
        self, service: PromoUpliftService
    ) -> None:
        tool = PromoUpliftTool(service)
        with pytest.raises(ToolExecutionError):
            tool._execute(PromoUpliftInput(promotion_id="NOPE"))


class TestPersistence:
    def test_round_trips_through_disk(
        self, model: FittedUpliftModel, tmp_path: Path
    ) -> None:
        model.save(tmp_path)
        loaded = FittedUpliftModel.load_from(tmp_path, None)

        assert loaded.artifact.selected == "augmented_ipw"
        assert loaded.validation_status == "passed"
        assert len(loaded.event_impact) == 3
        assert loaded.headline is not None

    def test_metadata_records_the_treatment_definition(
        self, model: FittedUpliftModel, tmp_path: Path
    ) -> None:
        import json

        model.save(tmp_path)
        metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["treatment_definition"]
        assert metadata["config_fingerprint"]

    def test_missing_artifact_raises_with_the_command_to_produce_one(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match=r"estimate_uplift\.py"):
            FittedUpliftModel.load_from(tmp_path / "absent", None)

    def test_for_promotion_is_a_lookup(self, model: FittedUpliftModel) -> None:
        result = model.for_promotion("PR01")
        assert result.promotion_id == "PR01"
        assert result.incremental_units == pytest.approx(150.0)
        assert result.roi == pytest.approx(400.0 / 200.0)

    def test_predict_uplift_without_a_cate_model_refuses(
        self, model: FittedUpliftModel
    ) -> None:
        from ml.promo_uplift.exceptions import UpliftModelUnavailableError

        with pytest.raises(UpliftModelUnavailableError, match="CATE model"):
            model.predict_uplift(pd.DataFrame({"demand_mean_28": [10.0]}))
