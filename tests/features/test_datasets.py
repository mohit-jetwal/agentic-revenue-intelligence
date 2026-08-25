"""Dataset builder tests (brief sections 35-40).

Each builder must produce a frame a model can actually consume: X and y
separated, lineage recorded, and - for the ones whose framing matters - the
right rows excluded.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from data.repositories.point_in_time import PointInTimeView
from data.repositories.sampling import PanelSample
from features.datasets import (
    UpliftWindows,
    create_cross_price_dataset,
    create_forecasting_dataset,
    create_price_elasticity_dataset,
    create_promo_optimization_dataset,
    create_promo_uplift_dataset,
)

pytestmark = [pytest.mark.features, pytest.mark.data]


@pytest.fixture(scope="module")
def window(smoke_as_of: date) -> tuple[date, date]:
    return smoke_as_of - timedelta(days=150), smoke_as_of


# --- 36. forecasting --------------------------------------------------------


def test_forecasting_separates_x_and_y(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_forecasting_dataset(
        smoke_view,
        train_start=start,
        train_end=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    assert len(dataset) > 0
    assert dataset.y is not None
    assert len(dataset.X) == len(dataset.y)
    assert dataset.metadata.target_name == "units"
    assert "units" not in dataset.X.columns


def test_forecasting_excludes_realised_spend(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Spend does not exist over a forecast horizon.

    Training on a feature that will be absent at inference guarantees a
    train/serve mismatch, so the config turns it off for this dataset.
    """
    start, end = window
    dataset = create_forecasting_dataset(
        smoke_view,
        train_start=start,
        train_end=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    assert "promotion_spend" not in dataset.X.columns
    assert "promotion_units" not in dataset.X.columns


def test_forecasting_carries_lag_and_rolling_features(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_forecasting_dataset(
        smoke_view,
        train_start=start,
        train_end=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    for probe in ("lag_7_units", "lag_28_units", "rolling_7_units", "rolling_28_units"):
        assert probe in dataset.X.columns


# --- 37. price elasticity ---------------------------------------------------


def test_elasticity_excludes_promotional_and_stockout_rows(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Both exclusions are load-bearing, not tidying.

    Promotional rows carry a price cut and an additive uplift at once, inflating
    the coefficient. Stockout rows report supply, not demand, and bias it toward
    zero.
    """
    start, end = window
    dataset = create_price_elasticity_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    if dataset.X.empty:
        pytest.skip("no eligible rows in the sampled window")

    if "promotion_flag" in dataset.X.columns:
        assert not dataset.X["promotion_flag"].astype(bool).any()
    if "stockout_flag" in dataset.X.columns:
        assert not dataset.X["stockout_flag"].astype(bool).any()


def test_elasticity_target_is_log_units(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """log-log, so the coefficient on log price *is* the elasticity."""
    start, end = window
    dataset = create_price_elasticity_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    if dataset.X.empty:
        pytest.skip("no eligible rows")

    assert dataset.metadata.target_name == "log_units"
    assert "log_price" in dataset.X.columns
    assert dataset.y is not None
    assert dataset.y.notna().all()


def test_elasticity_exclusions_can_be_disabled(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Step 8 needs to demonstrate the bias, which means being able to induce it."""
    start, end = window
    kwargs = dict(
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    strict = create_price_elasticity_dataset(smoke_view, **kwargs)
    loose = create_price_elasticity_dataset(
        smoke_view, exclude_promotional=False, exclude_stockouts=False, **kwargs
    )
    assert len(loose) >= len(strict)


# --- 38. promotion uplift ---------------------------------------------------


def test_uplift_labels_treatment_and_periods(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_promo_uplift_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    if dataset.X.empty:
        pytest.skip("no rows in the sampled window")

    assert "treatment" in dataset.X.columns
    assert "period" in dataset.X.columns
    assert set(dataset.X["period"].unique()) <= {"baseline", "pre", "treatment", "post"}


def test_uplift_treatment_matches_promotion_flag(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_promo_uplift_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    if dataset.X.empty or not dataset.X["treatment"].any():
        pytest.skip("no promotions in the sampled window")

    treated = dataset.X[dataset.X["treatment"]]
    assert (treated["period"] == "treatment").all()
    assert treated["promotion_id"].notna().all()


def test_uplift_marks_a_control_universe(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Controls are unpromoted days of *promoted* products.

    Never-promoted products would be a worse control: promotions are targeted,
    so those products are systematically different and the comparison would
    measure selection rather than promotional effect.
    """
    start, end = window
    dataset = create_promo_uplift_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
    )
    if dataset.X.empty or not dataset.X["treatment"].any():
        pytest.skip("no promotions in the sampled window")

    assert "eligible_for_uplift" in dataset.X.columns
    promoted_products = set(dataset.X.loc[dataset.X["treatment"], "product_id"])
    eligible = set(dataset.X.loc[dataset.X["eligible_for_uplift"], "product_id"])
    assert promoted_products <= eligible


def test_uplift_windows_are_configurable(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_promo_uplift_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        product_ids=smoke_panel_sample.product_ids,
        store_ids=smoke_panel_sample.store_ids,
        windows=UpliftWindows(pre_days=14, post_days=7),
    )
    assert dataset.metadata.feature_set_name == "promo_uplift"


# --- 39. cross-price --------------------------------------------------------


def test_cross_price_uses_declared_relationships(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Scoping by the relationship table is what keeps multiple comparisons
    tractable - N products give N(N-1) ordered pairs."""
    start, end = window
    dataset = create_cross_price_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        store_ids=smoke_panel_sample.store_ids,
        max_pairs=25,
    )
    if dataset.X.empty:
        pytest.skip("no co-listed related pairs in the sampled window")

    for column in ("product_a", "product_b", "price_b", "demand_a", "relationship_type"):
        assert column in dataset.X.columns
    assert (dataset.X["product_a"] != dataset.X["product_b"]).all()


def test_cross_price_is_store_level(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Substitution happens on a shelf. Aggregating to product-date averages away
    the store-level price variation that identifies the effect."""
    start, end = window
    dataset = create_cross_price_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        store_ids=smoke_panel_sample.store_ids,
        max_pairs=25,
    )
    if dataset.X.empty:
        pytest.skip("no pairs")
    assert "store_id" in dataset.X.columns


def test_cross_price_respects_the_pair_cap(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_cross_price_dataset(
        smoke_view,
        start_date=start,
        end_date=end,
        store_ids=smoke_panel_sample.store_ids,
        max_pairs=5,
    )
    if dataset.X.empty:
        pytest.skip("no pairs")
    pairs = dataset.X[["product_a", "product_b"]].drop_duplicates()
    assert len(pairs) <= 5


# --- 40. optimisation -------------------------------------------------------


def test_optimization_is_one_row_per_decision_cell(
    smoke_view: PointInTimeView, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_promo_optimization_dataset(smoke_view, start_date=start, end_date=end)
    if dataset.X.empty:
        pytest.skip("no trade promotions in the sampled window")

    assert not dataset.X.duplicated(subset=["product_id", "region"]).any()
    for column in ("historical_roi", "margin", "minimum_spend", "maximum_spend"):
        assert column in dataset.X.columns


def test_optimization_leaves_model_outputs_empty(
    smoke_view: PointInTimeView, window: tuple[date, date]
) -> None:
    """forecast_sales and uplift are outputs of Steps 5 and 6.

    Filling them with a naive estimate here would look complete and quietly
    become the number the optimiser trusts. An explicit null is the honest
    interface.
    """
    start, end = window
    dataset = create_promo_optimization_dataset(smoke_view, start_date=start, end_date=end)
    if dataset.X.empty:
        pytest.skip("no trade promotions")

    for column in ("forecast_sales", "baseline_sales", "uplift"):
        assert column in dataset.X.columns
        assert dataset.X[column].isna().all()


def test_optimization_spend_bounds_are_ordered(
    smoke_view: PointInTimeView, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_promo_optimization_dataset(smoke_view, start_date=start, end_date=end)
    if dataset.X.empty:
        pytest.skip("no trade promotions")
    assert (dataset.X["maximum_spend"] >= dataset.X["minimum_spend"]).all()


# --- lineage across all builders --------------------------------------------


def test_every_builder_records_lineage(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    """Section 31: dataset version, feature version, as-of date, source tables.

    This is the record that lets a Step 12 model run be reproduced.
    """
    start, end = window
    builders = [
        create_forecasting_dataset(
            smoke_view,
            train_start=start,
            train_end=end,
            product_ids=smoke_panel_sample.product_ids,
        ),
        create_price_elasticity_dataset(
            smoke_view,
            start_date=start,
            end_date=end,
            product_ids=smoke_panel_sample.product_ids,
        ),
        create_promo_uplift_dataset(
            smoke_view,
            start_date=start,
            end_date=end,
            product_ids=smoke_panel_sample.product_ids,
        ),
        create_promo_optimization_dataset(smoke_view, start_date=start, end_date=end),
    ]

    for dataset in builders:
        metadata = dataset.metadata
        assert metadata.feature_set_name
        assert metadata.feature_version
        assert metadata.dataset_version
        assert metadata.as_of_date == smoke_view.as_of_date
        assert metadata.source_tables
        assert metadata.cache_key()


def test_lineage_dataset_version_is_as_of_qualified(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, window: tuple[date, date]
) -> None:
    start, end = window
    dataset = create_forecasting_dataset(
        smoke_view,
        train_start=start,
        train_end=end,
        product_ids=smoke_panel_sample.product_ids,
    )
    assert str(smoke_view.as_of_date) in dataset.metadata.dataset_version
