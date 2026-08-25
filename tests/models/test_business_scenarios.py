"""Business scenarios (brief section 31).

The tests above check that the model is statistically well-behaved. These check
that it is *useful* - that in the handful of situations a category manager
actually asks about, the answer points the right way.

Framing them as scenarios rather than metrics is deliberate. "WMAPE is 0.18"
cannot be reviewed by the person who owns the promotion budget; "during a
stockout the model says demand held up and we lost sales" can be, and is the
claim the platform will ultimately be judged on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.baseline.models import build_estimator
from ml.baseline.training import (
    PromotionApproach,
    build_temporal_split,
    train_baseline,
)
from tests.model_fixtures import SEASONAL_AMPLITUDE, WEEKEND_UPLIFT

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def scenario_frame(feature_panel: pd.DataFrame, synthetic_panel: pd.DataFrame):
    """Test-window rows with predictions and truth, for scenario questions."""
    split = build_temporal_split(feature_panel)
    trained = train_baseline(
        feature_panel,
        build_estimator("lightgbm", seed=7),
        approach=PromotionApproach.EXCLUDE,
        split=split,
    )

    dates = pd.to_datetime(feature_panel["date"]).dt.date
    mask = (dates >= split.test_start) & (dates <= split.test_end)

    rows = feature_panel[mask].copy()
    rows["baseline_units"] = trained.predict_baseline(rows)
    rows["latent_units"] = synthetic_panel.loc[mask, "latent_units"].to_numpy()
    rows["true_baseline"] = synthetic_panel.loc[mask, "true_baseline"].to_numpy()
    return rows


class TestStockoutScenario:
    """"We stocked out last week - how much did we actually lose?"

    The question the baseline exists to answer, and the one the raw sales data
    cannot. Sales fell, but demand did not, and only a model that estimates
    demand can tell the difference.
    """

    def test_lost_sales_are_positive_and_material(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        stockouts = scenario_frame[scenario_frame["stockout_flag"].astype(bool)]

        lost = stockouts["baseline_units"].sum() - stockouts["units"].sum()

        assert lost > 0, "the model reports no lost sales during stockouts"

    def test_estimated_loss_is_close_to_the_true_loss(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """The number a supply-chain team would act on, checked against truth.

        Getting the direction right is necessary but not sufficient - an
        estimate that is directionally correct and 3x too large would still
        drive the wrong safety-stock decision.
        """
        stockouts = scenario_frame[scenario_frame["stockout_flag"].astype(bool)]

        estimated = stockouts["baseline_units"].sum() - stockouts["units"].sum()
        actual = stockouts["latent_units"].sum() - stockouts["units"].sum()

        assert estimated == pytest.approx(actual, rel=0.35), (
            f"estimated {estimated:,.0f} lost units against a true loss of "
            f"{actual:,.0f}"
        )

    def test_a_stockout_is_not_reported_as_a_demand_decline(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """The inversion this whole step is built to prevent.

        If the baseline dropped alongside the censored sales, a root-cause agent
        would conclude "demand fell" when the truth is "we had nothing to sell" -
        and would recommend a price cut to fix a warehouse problem.
        """
        stockouts = scenario_frame[scenario_frame["stockout_flag"].astype(bool)]
        normal = scenario_frame[~scenario_frame["stockout_flag"].astype(bool)]

        stockout_baseline = stockouts["baseline_units"].mean()
        normal_baseline = normal["baseline_units"].mean()

        assert stockout_baseline > normal_baseline * 0.8, (
            "the baseline collapses on stockout days, so the model is reporting "
            "a supply failure as a demand decline"
        )


class TestPromotionScenario:
    """"Did that promotion work?" - the platform's headline question."""

    def test_uplift_is_positive_for_a_promotion_that_worked(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        promoted = scenario_frame[
            scenario_frame["promotion_flag"].astype(bool)
            & ~scenario_frame["stockout_flag"].astype(bool)
        ]

        uplift = promoted["units"].sum() - promoted["baseline_units"].sum()

        assert uplift > 0

    def test_no_phantom_uplift_on_days_with_no_promotion(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """The mirror-image failure, and the more insidious one.

        A baseline biased low reports uplift where nothing happened. Every
        promotion then looks profitable, the model is never questioned because
        it keeps delivering good news, and budget flows toward promotions that
        do nothing.
        """
        clean = scenario_frame[
            ~scenario_frame["promotion_flag"].astype(bool)
            & ~scenario_frame["stockout_flag"].astype(bool)
        ]

        phantom = (
            clean["units"].sum() - clean["baseline_units"].sum()
        ) / clean["baseline_units"].sum()

        assert abs(phantom) < 0.05, (
            f"non-promotional days show {phantom:+.1%} apparent uplift, which "
            f"would be attributed to promotions that did not run"
        )

    def test_uplift_is_larger_on_promotion_than_off(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """Separation between the two populations, not just a positive number."""
        in_stock = scenario_frame[~scenario_frame["stockout_flag"].astype(bool)]
        promoted = in_stock[in_stock["promotion_flag"].astype(bool)]
        clean = in_stock[~in_stock["promotion_flag"].astype(bool)]

        promoted_ratio = promoted["units"].sum() / promoted["baseline_units"].sum()
        clean_ratio = clean["units"].sum() / clean["baseline_units"].sum()

        assert promoted_ratio > clean_ratio * 1.2


class TestSeasonalityScenario:
    """"Sales are up 20% - is that us, or is it just December?" """

    def test_the_baseline_carries_the_seasonal_shape(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """A flat baseline would attribute every seasonal peak to whatever
        happened to be running at the time.

        The fixture's seasonal amplitude is known, so the baseline's own
        variation across months must be of a comparable order - not merely
        non-zero.
        """
        by_month = scenario_frame.groupby("month")["baseline_units"].mean()

        spread = (by_month.max() - by_month.min()) / by_month.mean()

        assert spread > SEASONAL_AMPLITUDE * 0.3, (
            f"the baseline varies only {spread:.1%} across months against a "
            f"built-in seasonal amplitude of {SEASONAL_AMPLITUDE:.0%} - it is "
            f"too flat to separate seasonality from intervention"
        )

    def test_the_weekend_effect_is_reproduced(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """Weekend uplift is not promotional uplift.

        A baseline blind to the weekly cycle would credit every Saturday to
        whatever campaign was live.
        """
        weekend = scenario_frame[scenario_frame["is_weekend"].astype(bool)]
        weekday = scenario_frame[~scenario_frame["is_weekend"].astype(bool)]

        ratio = weekend["baseline_units"].mean() / weekday["baseline_units"].mean()

        assert ratio > 1.0 + (WEEKEND_UPLIFT - 1.0) * 0.4, (
            f"baseline weekend lift is {ratio:.2f}x against a built-in "
            f"{WEEKEND_UPLIFT}x"
        )


class TestSegmentConsistency:
    """"Does this hold for my category, or only on average?"

    An aggregate that passes while one category is badly wrong is a trap: the
    platform reports confident numbers to the manager whose category is the
    broken one.
    """

    def test_no_category_is_wildly_biased(self, scenario_frame: pd.DataFrame) -> None:
        clean = scenario_frame[
            ~scenario_frame["promotion_flag"].astype(bool)
            & ~scenario_frame["stockout_flag"].astype(bool)
        ]

        for category, rows in clean.groupby("category", observed=True):
            bias = (
                rows["units"].sum() - rows["baseline_units"].sum()
            ) / rows["baseline_units"].sum()

            assert abs(bias) < 0.10, f"category {category} is biased {bias:+.1%}"

    def test_no_channel_is_wildly_biased(self, scenario_frame: pd.DataFrame) -> None:
        clean = scenario_frame[
            ~scenario_frame["promotion_flag"].astype(bool)
            & ~scenario_frame["stockout_flag"].astype(bool)
        ]

        for channel, rows in clean.groupby("channel", observed=True):
            bias = (
                rows["units"].sum() - rows["baseline_units"].sum()
            ) / rows["baseline_units"].sum()

            assert abs(bias) < 0.10, f"channel {channel} is biased {bias:+.1%}"

    def test_every_product_gets_a_usable_baseline(
        self, scenario_frame: pd.DataFrame
    ) -> None:
        """No product may be silently unservable.

        A NaN or zero baseline for one SKU becomes an unanswerable question in
        the UI rather than an error anyone investigates.
        """
        by_product = scenario_frame.groupby("product_id")["baseline_units"].mean()

        assert by_product.notna().all()
        assert (by_product > 0).all()
