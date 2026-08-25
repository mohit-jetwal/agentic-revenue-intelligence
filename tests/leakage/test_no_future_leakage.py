"""Leakage tests - the property Step 3 exists to guarantee (brief section 28).

Every model in Steps 4-11 is temporal, and a single leaked future value produces
a model that backtests beautifully and fails in production. The failure is
silent: the frame is well-formed, the metrics are excellent, and nothing
complains until real money is behind the recommendation.

The headline test here is :func:`test_features_are_identical_to_a_truncated_world`.
It builds features twice - once from the full dataset, once from a dataset
physically truncated at the as-of date - and asserts the two agree for every row
on or before that date. If any feature reaches past the cut, the two runs
diverge and the test fails.

What makes that test worth more than the targeted ones: it needs no knowledge of
*which* feature leaked. It keeps holding as Steps 4-11 add features, without
anyone remembering to extend it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data.repositories.availability import Availability, availability_of
from data.repositories.local import LocalDataRepository
from features.contracts.catalogue import FEATURE_SPECS, forward_looking_features
from features.contracts.config import load_feature_config
from features.contracts.specs import Temporality
from features.engineering import FeatureEngineer, FeatureRequest
from features.engineering.panel import rolling_on_shifted, shifted_group

pytestmark = [pytest.mark.leakage, pytest.mark.data]


# ---------------------------------------------------------------------------
# The equivalence test
# ---------------------------------------------------------------------------


def test_features_are_identical_to_a_truncated_world(
    smoke_repository: LocalDataRepository, smoke_panel_sample: object
) -> None:
    """Features built from full data must match those built from truncated data.

    The definitive leakage test. Two worlds:

    * one where the future exists but we promise not to look at it (a view over
      the full dataset);
    * one where the future does not exist at all (the same view over a dataset
      physically cut at the as-of date).

    For rows on or before the as-of date, a correct feature pipeline cannot tell
    those apart. Any difference is a feature reaching forward.
    """
    sample = smoke_panel_sample
    as_of = sample.end_date - timedelta(days=30)  # type: ignore[attr-defined]

    request = FeatureRequest(
        start_date=as_of - timedelta(days=60),
        end_date=as_of,
        product_ids=sample.product_ids[:4],  # type: ignore[attr-defined]
        store_ids=sample.store_ids[:3],  # type: ignore[attr-defined]
    )

    full_world = FeatureEngineer(smoke_repository.as_of(as_of)).build(request)

    # The truncated world: a repository that genuinely has no data past as_of.
    truncated_repo = _TruncatedRepository(smoke_repository, as_of)
    cut_world = FeatureEngineer(truncated_repo.as_of(as_of)).build(request)

    assert not full_world.empty, "no features produced; widen the sample"
    assert len(full_world) == len(cut_world), (
        f"row counts differ ({len(full_world)} vs {len(cut_world)}), so the "
        f"pipeline is reading rows beyond the as-of date"
    )

    divergent = _columns_that_differ(full_world, cut_world)
    assert not divergent, (
        f"these features differ between the full and truncated worlds, which "
        f"means they read data after {as_of}: {divergent}"
    )


class _TruncatedRepository(LocalDataRepository):
    """A repository whose observed tables genuinely end at ``cutoff``.

    Not a mock - it delegates to a real repository and hard-cuts observed data,
    so the comparison is against a world where the future is absent rather than
    merely unrequested. Known-in-advance tables pass through untouched, because
    the promotion calendar and the public holidays *do* extend past the cutoff
    in reality.
    """

    def __init__(self, inner: LocalDataRepository, cutoff: date) -> None:
        super().__init__(
            parquet_root=inner.parquet_root,
            max_result_rows=inner.max_result_rows,
        )
        self.cutoff = cutoff

    def _cut(self, table: str, frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
        if availability_of(table) is not Availability.OBSERVED:
            return frame
        if frame.empty or column not in frame.columns:
            return frame
        return frame[pd.to_datetime(frame[column]).dt.date <= self.cutoff]

    def get_sales(self, **kwargs: object) -> pd.DataFrame:  # type: ignore[override]
        return self._cut("sales_daily", super().get_sales(**kwargs))  # type: ignore[arg-type]

    def get_inventory(self, **kwargs: object) -> pd.DataFrame:  # type: ignore[override]
        return self._cut("inventory", super().get_inventory(**kwargs))  # type: ignore[arg-type]

    def get_competitor_prices(self, **kwargs: object) -> pd.DataFrame:  # type: ignore[override]
        return self._cut(
            "competitor_pricing",
            super().get_competitor_prices(**kwargs),  # type: ignore[arg-type]
        )


def _columns_that_differ(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Columns whose values disagree between two feature frames."""
    keys = ["date", "product_id", "store_id"]
    left_sorted = left.sort_values(keys).reset_index(drop=True)
    right_sorted = right.sort_values(keys).reset_index(drop=True)

    shared = [c for c in left_sorted.columns if c in right_sorted.columns]
    divergent: list[str] = []

    for column in shared:
        a, b = left_sorted[column], right_sorted[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(
                a.astype(float).fillna(-999999), b.astype(float).fillna(-999999), atol=1e-9
            ):
                divergent.append(column)
        elif not a.astype(str).equals(b.astype(str)):
            divergent.append(column)

    return divergent


# ---------------------------------------------------------------------------
# Repository-level: observed data never crosses the as-of date
# ---------------------------------------------------------------------------


def test_observed_tables_stop_at_the_as_of_date(smoke_repository: LocalDataRepository) -> None:
    """Sales, inventory and competitor prices are cut; the future is unreachable."""
    as_of = date(2024, 6, 30)
    view = smoke_repository.as_of(as_of)

    for label, frame in (
        ("sales", view.get_sales()),
        ("inventory", view.get_inventory()),
        ("competitor", view.get_competitor_prices()),
    ):
        assert not frame.empty, f"{label} returned nothing; check the fixture window"
        latest = pd.to_datetime(frame["date"]).dt.date.max()
        assert latest <= as_of, f"{label} leaked data from {latest}, past {as_of}"


def test_known_in_advance_tables_are_not_cut(smoke_repository: LocalDataRepository) -> None:
    """Calendar and promotions extend beyond as-of, because they genuinely do.

    The counterpart to the test above. A pipeline that clamped these would be
    "safe" and wrong - it would hide a promotion calendar the business has
    committed to and a holiday everybody can see coming.
    """
    as_of = date(2024, 6, 30)
    view = smoke_repository.as_of(as_of)

    calendar = view.get_calendar()
    assert pd.to_datetime(calendar["date"]).dt.date.max() > as_of

    promotions = view.get_promotions()
    assert pd.to_datetime(promotions["start_date"]).dt.date.max() > as_of


def test_future_promotion_actuals_are_masked(smoke_repository: LocalDataRepository) -> None:
    """A future promotion's schedule is visible; its realised spend is not.

    The subtle half of the known-in-advance rule. Letting realised spend through
    on a forward-dated row would hand a model a function of the demand it is
    trying to predict.
    """
    as_of = date(2024, 6, 30)
    promotions = smoke_repository.as_of(as_of).get_promotions()

    future = promotions[pd.to_datetime(promotions["start_date"]).dt.date > as_of]
    assert not future.empty, "no future promotions in the fixture; widen the window"

    assert future["promotion_spend"].isna().all(), "realised spend leaked on a future promotion"
    assert future["promotion_units"].isna().all(), "realised units leaked on a future promotion"

    past = promotions[pd.to_datetime(promotions["end_date"]).dt.date <= as_of]
    if not past.empty:
        assert past["promotion_spend"].notna().any(), (
            "historical spend was masked too; only future actuals should be hidden"
        )


# ---------------------------------------------------------------------------
# Primitive-level: the shift discipline
# ---------------------------------------------------------------------------


def test_lag_equals_the_actual_earlier_value(smoke_features: pd.DataFrame) -> None:
    """lag_7_units at t must equal units at t-7 for the same product-store."""
    series = _longest_series(smoke_features)
    assert len(series) > 40, "need a longer series to test lags"

    for position in (10, 20, 30):
        row = series.iloc[position]
        earlier = series.iloc[position - 7]
        expected_gap = (row["date"] - earlier["date"]).days
        if expected_gap != 7:
            continue  # gap in the series; skip rather than assert on a hole
        assert row["lag_7_units"] == earlier["units"], (
            f"lag_7_units at {row['date'].date()} is {row['lag_7_units']}, "
            f"but units 7 days earlier was {earlier['units']}"
        )


def test_rolling_excludes_the_current_day(smoke_features: pd.DataFrame) -> None:
    """rolling_7_units must be the mean of the *previous* seven days.

    The classic mistake: `rolling(7)` includes the current row, so the feature
    contains one seventh of the very number being predicted.
    """
    series = _longest_series(smoke_features)
    assert len(series) > 30

    position = 20
    row = series.iloc[position]
    window = series.iloc[position - 7 : position]["units"]

    assert row["rolling_7_units"] == pytest.approx(window.mean(), abs=1e-6)

    including_today = series.iloc[position - 6 : position + 1]["units"].mean()
    if abs(window.mean() - including_today) > 1e-9:
        assert row["rolling_7_units"] != pytest.approx(including_today, abs=1e-9), (
            "rolling window appears to include the current day"
        )


def test_shift_helper_refuses_a_zero_shift() -> None:
    """The primitive that enforces the discipline must not be bypassable."""
    panel = pd.DataFrame(
        {
            "product_id": ["P1"] * 5,
            "store_id": ["S1"] * 5,
            "date": pd.date_range("2024-01-01", periods=5),
            "units": [1, 2, 3, 4, 5],
        }
    )
    with pytest.raises(ValueError, match="point-in-time"):
        shifted_group(panel, "units", periods=0)

    with pytest.raises(ValueError, match="point-in-time"):
        shifted_group(panel, "units", periods=-1)


def test_rolling_helper_never_sees_the_current_row() -> None:
    """Constructed case: a spike today must not appear in today's rolling mean."""
    panel = pd.DataFrame(
        {
            "product_id": ["P1"] * 6,
            "store_id": ["S1"] * 6,
            "date": pd.date_range("2024-01-01", periods=6),
            "units": [10, 10, 10, 10, 10, 1000],
        }
    )
    rolling = rolling_on_shifted(panel, "units", window=3, statistic="mean")

    # The last row's window covers the three prior 10s, not the 1000.
    assert rolling.iloc[-1] == pytest.approx(10.0)


def test_lags_do_not_cross_product_store_boundaries() -> None:
    """A lag must never reach into a different series.

    Without grouping, the last row of one product-store becomes the lag of the
    first row of the next - a leak that is invisible in aggregate statistics.
    """
    panel = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "store_id": ["S1", "S1", "S1", "S1"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "units": [10, 20, 500, 600],
        }
    ).sort_values(["product_id", "store_id", "date"])

    lagged = shifted_group(panel, "units", periods=1)

    # First row of each group has no predecessor within its own series.
    assert pd.isna(lagged.iloc[0])
    assert pd.isna(lagged.iloc[2])
    assert lagged.iloc[1] == 10
    assert lagged.iloc[3] == 500


# ---------------------------------------------------------------------------
# Catalogue-level: forward-looking features are declared and justified
# ---------------------------------------------------------------------------


def test_forward_looking_features_match_the_allow_list() -> None:
    """Only explicitly allowed features may read beyond their row date.

    The allow-list lives in ``configs/features/features.yaml``. Adding a
    forward-looking feature therefore requires a deliberate config change, which
    is the point - it turns "I added a feature" into "I asserted this
    information is knowable in advance".
    """
    config = load_feature_config()
    declared = {spec.name for spec in forward_looking_features()}
    allowed = set(config.validation.allowed_forward_looking)

    assert declared == allowed, (
        f"forward-looking features and the allow-list disagree.\n"
        f"  declared but not allowed: {sorted(declared - allowed)}\n"
        f"  allowed but not declared: {sorted(allowed - declared)}"
    )


def test_every_forward_looking_feature_is_justified() -> None:
    """A forward-looking feature must state why the information is knowable."""
    for spec in forward_looking_features():
        assert spec.forward_justification, f"{spec.name} looks forward without justification"
        assert len(spec.forward_justification) > 60, (
            f"{spec.name}'s justification is too thin to review: {spec.forward_justification!r}"
        )


def test_forward_looking_set_is_small_and_expected() -> None:
    """Pin the exact set, so an addition is a conscious, reviewable change."""
    assert {spec.name for spec in forward_looking_features()} == {
        "days_until_promotion_end",
        "days_to_next_promotion",
        "days_to_festival",
    }


def test_demand_features_are_all_backward_looking() -> None:
    """No demand feature may be contemporaneous or forward-looking.

    Demand is the target. A demand feature that reads the current day is the
    target in disguise.
    """
    offenders = [
        spec.name
        for spec in FEATURE_SPECS.values()
        if spec.group.value == "demand" and spec.temporality is not Temporality.BACKWARD
    ]
    assert not offenders, f"demand features must be backward-looking: {offenders}"


# ---------------------------------------------------------------------------
# Frame-level: forbidden columns
# ---------------------------------------------------------------------------


def test_forbidden_columns_are_absent_from_training_features(
    smoke_features: pd.DataFrame,
) -> None:
    """Columns that are functions of the target must never reach a model.

    ``revenue`` is ``units x price``; ``closing_inventory`` is
    ``opening + received - sold``. Either one hands a model the answer.
    """
    config = load_feature_config()
    present = [c for c in config.validation.forbidden_columns if c in smoke_features.columns]
    assert not present, f"target-derived columns leaked into features: {present}"


def test_censored_inventory_columns_are_excluded(smoke_features: pd.DataFrame) -> None:
    """sold_units and closing_inventory are same-day functions of demand."""
    for column in ("sold_units", "closing_inventory", "inventory_days"):
        assert column not in smoke_features.columns, f"{column} leaked into the feature frame"


def test_shifted_inventory_is_retained(smoke_features: pd.DataFrame) -> None:
    """Excluding the censored columns must not simply discard the information.

    Yesterday's closing position is both legitimate and useful; the point is to
    shift it, not to lose it.
    """
    assert any(c.endswith("_lag_1") for c in smoke_features.columns), (
        "censored columns were dropped without retaining their shifted form"
    )


def _longest_series(features: pd.DataFrame) -> pd.DataFrame:
    """The product-store series with the most rows, sorted by date."""
    counts = features.groupby(["product_id", "store_id"]).size()
    product_id, store_id = counts.idxmax()
    return (
        features[(features["product_id"] == product_id) & (features["store_id"] == store_id)]
        .sort_values("date")
        .reset_index(drop=True)
    )
