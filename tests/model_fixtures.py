"""Fixtures for the baseline sales model tests (Step 4).

Deliberately *not* built on the Step 2/3 smoke dataset, for two reasons.

**Length.** :func:`build_temporal_split` needs 390 days of history - 120 test,
90 validation, 60 calibration and at least 120 to train on. The smoke feature
panel spans 120 days total, so every split test would fail for a reason that has
nothing to do with the model.

**Knowability.** The point of most of these tests is to assert that the model
*recovers a relationship that is known to exist*. On generated data the true
coefficients are hidden inside Step 2's simulator; here they are written three
lines above the assertion. A test that says "seasonality is worth 30% and the
model must find it" is a far stronger statement than "WMAPE is under some number
that happened to pass today".

The panel below is therefore synthetic-of-synthetic: small, fast, seeded, and
constructed so that every effect a test looks for was put there on purpose.
Tests that need the *real* pipeline behaviour use the on-disk dev artifacts and
skip when they are absent - see :func:`trained_model_dir`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Known-truth parameters for the synthetic panel. Tests assert against these
# names rather than repeating the literals, so changing a value here cannot
# leave an assertion silently checking the old one.
PANEL_DAYS = 730
PANEL_PRODUCTS = 4
PANEL_STORES = 3
BASE_DEMAND = 40.0
#: Multiplicative weekend effect - Saturday and Sunday sell more.
WEEKEND_UPLIFT = 1.35
#: Amplitude of the annual seasonal cycle, as a fraction of base demand.
SEASONAL_AMPLITUDE = 0.30
#: Multiplicative demand lift applied on promotional days.
PROMO_LIFT = 1.80
#: Fraction of days a given product-store pair is on promotion.
PROMO_RATE = 0.12
#: Fraction of days a given pair is stocked out.
STOCKOUT_RATE = 0.06
#: On a stockout day, only this share of demand can actually be sold. This is
#: the censoring the model must see through rather than learn.
STOCKOUT_AVAILABILITY = 0.35
#: Promotions are scheduled toward seasonal peaks, mirroring Step 2's
#: `targeting_strength`. Without this the two promotion approaches would be
#: equivalent and the comparison test would prove nothing.
PROMO_TARGETING = 0.60


def _make_panel(seed: int = 7) -> pd.DataFrame:
    """A feature panel with known structure and hidden ground truth.

    Columns mirror the shape of the real Step 3 panel closely enough for the
    training code to run unmodified: identifiers, the target, promotion and
    stockout flags, categoricals, lags and rolling means.

    Carries two extra columns the real panel never has - ``latent_units`` (true
    uncensored demand) and ``true_baseline`` (demand with no promotion) - which
    are the analogue of Step 2's ground truth and must never reach the feature
    matrix. :data:`ml.baseline.training.EXCLUDED_FROM_FEATURES` does not know
    about them, so tests that build a feature matrix drop them explicitly and
    :func:`test_ground_truth_columns_never_become_features` guards it.
    """
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(PANEL_DAYS)]

    categories = ["beverages", "snacks"]
    channels = ["grocery", "convenience"]
    rows: list[dict[str, object]] = []

    for p in range(PANEL_PRODUCTS):
        product_id = f"P{p:03d}"
        category = categories[p % len(categories)]
        # A per-product level so the model has something to learn beyond a
        # global mean; a flat panel would let the naive benchmark tie trivially.
        product_scale = 0.7 + 0.25 * p

        for s in range(PANEL_STORES):
            store_id = f"S{s:03d}"
            channel = channels[s % len(channels)]
            store_scale = 0.8 + 0.2 * s

            for day_index, current in enumerate(dates):
                day_of_year = current.timetuple().tm_yday
                seasonal = 1.0 + SEASONAL_AMPLITUDE * np.sin(
                    2 * np.pi * day_of_year / 365.25
                )
                weekend = WEEKEND_UPLIFT if current.weekday() >= 5 else 1.0

                baseline_demand = (
                    BASE_DEMAND * product_scale * store_scale * seasonal * weekend
                )

                # Promotions targeted toward seasonal peaks, exactly the
                # selection bias Approach C inherits.
                promo_odds = PROMO_RATE * (
                    1.0 + PROMO_TARGETING * (seasonal - 1.0) / SEASONAL_AMPLITUDE
                )
                on_promo = bool(rng.random() < np.clip(promo_odds, 0.0, 0.9))

                # The true conditional mean of demand on this day, promotion
                # included. This is the analogue of Step 2's `mean_demand`, and
                # it is what `latent` is drawn around - so the gap between the
                # two is pure noise and nothing else. `true_baseline` below is a
                # different quantity: the mean with the promotion removed, which
                # is what the model is trying to estimate.
                true_mean = baseline_demand * (PROMO_LIFT if on_promo else 1.0)

                latent = max(true_mean * rng.normal(1.0, 0.10), 0.0)

                stocked_out = bool(rng.random() < STOCKOUT_RATE)
                observed = latent * STOCKOUT_AVAILABILITY if stocked_out else latent

                rows.append(
                    {
                        "date": current,
                        "product_id": product_id,
                        "store_id": store_id,
                        "units": float(round(observed)),
                        "latent_units": float(round(latent)),
                        "true_baseline": float(baseline_demand),
                        "true_mean": float(true_mean),
                        "promotion_flag": on_promo,
                        "promotion_discount": 0.25 if on_promo else 0.0,
                        "stockout_flag": stocked_out,
                        "price": 4.50 - (0.25 if on_promo else 0.0),
                        "category": category,
                        "channel": channel,
                        "day_of_week": current.weekday(),
                        "week_of_year": current.isocalendar().week,
                        "month": current.month,
                        "is_weekend": current.weekday() >= 5,
                        "day_index": day_index,
                    }
                )

    panel = pd.DataFrame(rows)

    # Lags and rolling means, computed the way Step 3 computes them: strictly
    # from prior rows within the product-store series, never including today.
    keys = [panel["product_id"], panel["store_id"]]
    grouped = panel.groupby(["product_id", "store_id"], sort=False)["units"]
    panel["lag_7_units"] = grouped.shift(7)
    panel["lag_364_units"] = grouped.shift(364)

    # Rolling means are computed on the *already shifted* series, so today's
    # value can never enter its own window - the same discipline Step 3's
    # `rolling_on_shifted` enforces in the real engineer.
    shifted = grouped.shift(1)
    panel["rolling_28_units"] = shifted.groupby(keys).transform(
        lambda s: s.rolling(28, min_periods=7).mean()
    )
    panel["rolling_7_units"] = shifted.groupby(keys).transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )

    for column in ("category", "channel"):
        panel[column] = panel[column].astype("category")

    return panel


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """Known-truth panel, generated once per session."""
    return _make_panel()


@pytest.fixture(scope="session")
def feature_panel(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """The panel with ground-truth columns stripped, as a model may see it."""
    return synthetic_panel.drop(columns=["latent_units", "true_baseline", "true_mean"])


@pytest.fixture(scope="session")
def latent_frame(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """Ground truth shaped like Step 2's ``latent_demand`` table."""
    frame = synthetic_panel[
        ["date", "product_id", "store_id", "latent_units", "units"]
    ].copy()
    frame = frame.rename(columns={"units": "observed_units"})
    frame["lost_units"] = (frame["latent_units"] - frame["observed_units"]).clip(lower=0)
    # `mean_demand` is the true conditional mean *including* promotion effects,
    # matching Step 2's semantics - not `true_baseline`, which has promotions
    # removed. Using the latter would make the noise floor look 9% biased when
    # in fact it is unbiased by construction.
    frame["mean_demand"] = synthetic_panel["true_mean"].to_numpy()
    return frame


@pytest.fixture(scope="session")
def trained_model_dir() -> Path:
    """The real trained model from ``scripts/train_baseline.py``.

    Skips rather than fails when absent: a clean checkout has no trained model,
    and a test suite that cannot pass without first running a multi-minute
    training job is a test suite people stop running.
    """
    directory = Path("data/local/models/baseline")
    if not (directory / "model.joblib").is_file():
        pytest.skip(
            "no trained baseline model; run "
            "`uv run python scripts/train_baseline.py --profile dev`"
        )
    return directory
