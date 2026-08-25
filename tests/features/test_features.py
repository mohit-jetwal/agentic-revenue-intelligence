"""Feature engineering and feature repository tests (brief sections 27, 34).

Leakage has its own suite in ``tests/leakage``. These cover everything else:
that features are produced, are named as the catalogue says, carry sane values,
and that the repository and configuration hold together.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data.repositories.point_in_time import PointInTimeView
from features.contracts import (
    FEATURE_GROUPS,
    FEATURE_SPECS,
    MODEL_REQUIREMENTS,
    FeatureGroupName,
    FeatureSpec,
    Temporality,
    requirement_for,
)
from features.contracts.config import FeatureConfigError, load_feature_config
from features.engineering import FeatureEngineer, FeatureRequest
from features.engineering.temporal import add_time_features
from features.repositories import LocalFeatureRepository

pytestmark = [pytest.mark.features, pytest.mark.data]


# --- the engineer -----------------------------------------------------------


def test_engineer_refuses_a_bare_repository(smoke_repository: object) -> None:
    """The structural half of leakage prevention.

    A bare repository would silently disable the as-of cut, and every feature
    built afterwards would be contaminated. Refusing turns a subtle correctness
    bug into an obvious TypeError.
    """
    with pytest.raises(TypeError, match="PointInTimeView"):
        FeatureEngineer(smoke_repository)  # type: ignore[arg-type]


def test_build_produces_all_families(smoke_features: pd.DataFrame) -> None:
    for probe in (
        "lag_7_units",
        "rolling_28_units",
        "price_index",
        "promotion_flag",
        "inventory_available",
        "price_gap",
        "dow_sin",
        "product_age_days",
        "store_age_days",
    ):
        assert probe in smoke_features.columns, f"{probe} missing from the panel"


def test_warmup_is_trimmed(smoke_features: pd.DataFrame, smoke_as_of: date) -> None:
    """Extra history is loaded so lags are populated, then trimmed away."""
    dates = pd.to_datetime(smoke_features["date"]).dt.date
    assert dates.max() <= smoke_as_of
    assert dates.min() >= smoke_as_of - timedelta(days=121)


def test_warmup_populates_lags_at_the_window_start(smoke_features: pd.DataFrame) -> None:
    """Without a warm-up window the first rows would have null lags, and a model
    would either drop them or train on nulls."""
    first_rows = smoke_features.sort_values("date").head(50)
    assert first_rows["lag_7_units"].notna().any()
    assert first_rows["rolling_28_units"].notna().any()


def test_panel_grain_is_unique(smoke_features: pd.DataFrame) -> None:
    """One row per product-store-date. A duplicate means a join fanned out."""
    keys = ["date", "product_id", "store_id"]
    assert not smoke_features.duplicated(subset=keys).any()


def test_empty_filter_returns_empty_not_error(smoke_view: PointInTimeView) -> None:
    panel = FeatureEngineer(smoke_view).build(
        FeatureRequest(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            product_ids=["P_DOES_NOT_EXIST"],
        )
    )
    assert panel.empty


# --- feature values ---------------------------------------------------------


def test_cyclical_encodings_are_on_the_unit_circle() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=400)})
    encoded = add_time_features(frame)

    radius = encoded["dow_sin"] ** 2 + encoded["dow_cos"] ** 2
    assert np.allclose(radius, 1.0)


def test_cyclical_encoding_wraps_the_year_boundary() -> None:
    """31 December and 1 January must be adjacent.

    The reason for sine/cosine rather than an integer: an integer day-of-year
    puts those two dates maximally far apart, and a model then cannot smooth
    across the boundary - which is where retail demand is most interesting.
    """
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-01", "2024-07-01"])})
    encoded = add_time_features(frame)

    def distance(a: int, b: int) -> float:
        return float(
            np.hypot(
                encoded["doy_sin"].iloc[a] - encoded["doy_sin"].iloc[b],
                encoded["doy_cos"].iloc[a] - encoded["doy_cos"].iloc[b],
            )
        )

    assert distance(0, 1) < distance(0, 2)


def test_indian_financial_year_starts_in_april() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-04-01", "2024-03-31"])})
    encoded = add_time_features(frame)

    assert int(encoded["financial_month"].iloc[0]) == 1
    assert int(encoded["financial_month"].iloc[1]) == 12
    assert int(encoded["financial_year"].iloc[0]) == 2024
    assert int(encoded["financial_year"].iloc[1]) == 2023


def test_price_index_centres_near_one(smoke_features: pd.DataFrame) -> None:
    """Price relative to the category mean should average around 1."""
    index = smoke_features["price_index"].dropna()
    assert not index.empty
    assert 0.5 < float(index.median()) < 2.0


def test_promotion_flag_agrees_with_discount(smoke_features: pd.DataFrame) -> None:
    promoted = smoke_features[smoke_features["promotion_flag"].astype(bool)]
    if promoted.empty:
        pytest.skip("no promotions in the sampled window")
    assert (promoted["promotion_discount"] > 0).all()


def test_unpromoted_rows_carry_zero_discount(smoke_features: pd.DataFrame) -> None:
    unpromoted = smoke_features[~smoke_features["promotion_flag"].astype(bool)]
    assert (unpromoted["promotion_discount"] == 0).all()


def test_days_since_stockout_is_non_negative(smoke_features: pd.DataFrame) -> None:
    values = smoke_features["days_since_stockout"].dropna()
    if not values.empty:
        assert (values >= 0).all()


#: Features whose nulls are meaningful rather than missing. Listed with reasons,
#: because "it's always been null" is how a genuinely broken join survives.
NULL_BY_DESIGN = {
    # Long lags need more history than a short test window contains.
    "lag_364_units": "needs a year of history",
    "lag_56_units": "needs 56 days of history",
    "rolling_56_units": "needs 56 days of history",
    # Undefined before the first occurrence: "never happened" is not the same as
    # "happened long ago", and filling it would assert the latter.
    "days_since_promotion": "null until the first promotion",
    "days_since_stockout": "null until the first stockout",
    # spend / duration, so undefined on the ~80% of rows with no active
    # promotion. Zero would claim the promotion cost nothing per day.
    "promotion_intensity": "undefined when no promotion is running",
    # No festival after the last one in the loaded calendar window.
    "days_to_festival": "null past the final festival in the window",
}


def test_null_rates_are_within_the_configured_bound(smoke_features: pd.DataFrame) -> None:
    """A feature that is mostly null is usually a broken join.

    Exemptions in :data:`NULL_BY_DESIGN` are features whose nulls carry meaning;
    each records why, so the list cannot quietly absorb a real defect.
    """
    config = load_feature_config()
    exempt = set(NULL_BY_DESIGN)

    offenders = {
        column: float(smoke_features[column].isna().mean())
        for column in smoke_features.columns
        if column in FEATURE_SPECS
        and column not in exempt
        and float(smoke_features[column].isna().mean()) > config.validation.max_null_rate
    }
    assert not offenders, f"features above the null-rate bound: {offenders}"


def test_null_exemptions_are_real_features() -> None:
    """The exemption list must not outlive the features it excuses.

    A stale entry silently exempts nothing, or worse, masks a renamed feature
    that is now genuinely broken.
    """
    unknown = [name for name in NULL_BY_DESIGN if name not in FEATURE_SPECS]
    assert not unknown, f"exemption list names features that no longer exist: {unknown}"


# --- the catalogue ----------------------------------------------------------


def test_catalogue_names_are_unique() -> None:
    names = [spec.name for group in FEATURE_GROUPS.values() for spec in group.features]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate feature names in the catalogue: {duplicates}"


def test_every_spec_records_its_provenance() -> None:
    """Section 25: name, group, source columns, transformation, version."""
    for spec in FEATURE_SPECS.values():
        assert spec.description, f"{spec.name} has no description"
        assert spec.transformation, f"{spec.name} has no transformation recorded"
        assert spec.source_tables, f"{spec.name} names no source table"
        assert spec.feature_version


def test_forward_looking_spec_requires_justification() -> None:
    """Constructing one without a reason must fail, not warn."""
    with pytest.raises(ValueError, match="justification"):
        FeatureSpec(
            name="peeks_at_tomorrow",
            group=FeatureGroupName.DEMAND,
            description="illustrative",
            temporality=Temporality.FORWARD_PLANNED,
        )


def test_configured_features_exist_in_the_catalogue() -> None:
    config = load_feature_config()
    for group, names in config.groups.items():
        unknown = [n for n in names if n not in FEATURE_SPECS]
        assert not unknown, f"group {group!r} names unknown features: {unknown}"


def test_config_rejects_an_unknown_feature(tmp_path: object) -> None:
    """A typo must fail at load, not become a silently-absent column."""
    from features.contracts.config import FeatureConfig

    with pytest.raises((FeatureConfigError, ValueError)):
        FeatureConfig.model_validate(
            {
                "feature_version": "v1.0",
                "groups": {"demand": ["lag_7_units", "not_a_real_feature"]},
                "datasets": {},
                "validation": {
                    "allowed_forward_looking": [
                        "days_until_promotion_end",
                        "days_to_next_promotion",
                        "days_to_festival",
                    ]
                },
            }
        )


# --- model requirements (section 24) ----------------------------------------


def test_all_seven_models_have_requirements() -> None:
    expected = {
        "baseline_sales",
        "demand_forecast",
        "promo_uplift",
        "trade_promo_optimization",
        "price_elasticity",
        "cross_price_elasticity",
        "price_optimization",
    }
    assert expected <= set(MODEL_REQUIREMENTS)


def test_requirements_reference_real_groups() -> None:
    for requirement in MODEL_REQUIREMENTS.values():
        for group in requirement.required_groups:
            assert group in FEATURE_GROUPS


def test_requirements_reference_real_features() -> None:
    for requirement in MODEL_REQUIREMENTS.values():
        unknown = [f for f in requirement.required_features if f not in FEATURE_SPECS]
        assert not unknown, f"{requirement.model_name} needs unknown features: {unknown}"


def test_every_requirement_records_its_hazards() -> None:
    """Each model's caveats are where the modelling traps are written down.

    Step 8 inheriting "price is endogenous" as a stated contract is worth more
    than it rediscovering it.
    """
    for requirement in MODEL_REQUIREMENTS.values():
        assert requirement.caveats, f"{requirement.model_name} records no caveats"


def test_elasticity_requirement_names_the_endogeneity_problem() -> None:
    requirement = requirement_for("price_elasticity")
    assert requirement is not None
    combined = " ".join(requirement.caveats).lower()
    assert "endogenous" in combined
    assert "instrument" in combined or "fixed effects" in combined


# --- the feature repository -------------------------------------------------


def test_repository_returns_metadata(feature_repository: LocalFeatureRepository) -> None:
    feature_set = feature_repository.get_demand_features(
        start_date=date(2024, 3, 1), end_date=date(2024, 4, 30)
    )
    metadata = feature_set.metadata
    assert metadata.feature_version
    assert metadata.dataset_version
    assert metadata.as_of_date == feature_repository.view.as_of_date
    assert metadata.source_tables


def test_training_features_separate_x_and_y(
    feature_repository: LocalFeatureRepository, smoke_panel_sample: object
) -> None:
    """Returning one combined frame is how the target ends up in X."""
    feature_set = feature_repository.get_training_features(
        dataset="forecasting",
        start_date=date(2024, 3, 1),
        end_date=date(2024, 4, 30),
        product_ids=smoke_panel_sample.product_ids[:3],  # type: ignore[attr-defined]
    )
    if len(feature_set) == 0:
        pytest.skip("no rows in the sampled window")

    assert feature_set.y is not None
    assert len(feature_set.X) == len(feature_set.y)
    assert "units" not in feature_set.X.columns


def test_cache_key_changes_with_feature_version(
    feature_repository: LocalFeatureRepository,
) -> None:
    """Bumping the version must invalidate cached sets - the definitions changed."""
    feature_set = feature_repository.get_demand_features(
        start_date=date(2024, 3, 1), end_date=date(2024, 3, 31)
    )
    original = feature_set.metadata.cache_key()
    bumped = feature_set.metadata.model_copy(update={"feature_version": "v9.9"}).cache_key()
    assert original != bumped


def test_cache_key_changes_with_as_of_date(
    feature_repository: LocalFeatureRepository,
) -> None:
    feature_set = feature_repository.get_demand_features(
        start_date=date(2024, 3, 1), end_date=date(2024, 3, 31)
    )
    original = feature_set.metadata.cache_key()
    shifted = feature_set.metadata.model_copy(update={"as_of_date": date(2023, 1, 1)}).cache_key()
    assert original != shifted


def test_materialisation_requires_a_cache_root(smoke_view: PointInTimeView) -> None:
    with pytest.raises(ValueError, match="cache_root"):
        LocalFeatureRepository(smoke_view, materialise=True)


def test_materialisation_round_trips(smoke_view: PointInTimeView, tmp_path: object) -> None:
    """With caching on, a second read must return the same rows."""
    repository = LocalFeatureRepository(
        smoke_view,
        materialise=True,
        cache_root=tmp_path,  # type: ignore[arg-type]
    )
    first = repository.get_demand_features(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31))
    second = repository.get_demand_features(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31))
    assert len(first) == len(second)
    assert list(first.X.columns) == list(second.X.columns)
