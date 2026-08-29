"""Fixtures for elasticity tests.

A synthetic panel with a **known** elasticity, built log-additively so a
log-log regression is the correctly specified estimator — the same structure the
platform generator uses. That is what lets these tests assert recovery of a
number rather than plausibility of one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TRUE_ELASTICITY = -1.80
TRUE_CROSS = 0.45


def make_panel(
    *,
    n_stores: int = 12,
    n_days: int = 400,
    elasticity: float = TRUE_ELASTICITY,
    seasonal_endogeneity: float = 0.0,
    idiosyncratic_endogeneity: float = 0.0,
    store_effect_sd: float = 0.05,
    seasonal_amplitude: float = 0.35,
    seed: int = 11,
    product_id: str = "P1",
) -> pd.DataFrame:
    """One product across stores and days, with a known elasticity.

    Two endogeneity channels, deliberately separate, because fixed effects
    handle them completely differently:

    ``seasonal_endogeneity``
        Price responds to a **seasonal** demand component. This is what the
        platform generator does - ``pricing_generator.py`` prices into
        anticipated demand, and anticipation is largely seasonal. Time fixed
        effects absorb it, which is why ``panel_fe`` recovers truth at r=0.99 on
        the real data.

    ``idiosyncratic_endogeneity``
        Price responds to the **daily** shock. Fixed effects cannot touch this:
        the confounder varies within every dimension being absorbed. This is the
        residual endogeneity the estimator's own warning names, and the fixture
        exists so that limitation is tested rather than merely asserted.

    ``store_effect_sd`` is small by default. With a dozen stores, large
    independent variation in base demand correlates with price by chance alone,
    and naive OLS picks that up - which would make the "clean" control panel
    fail for a reason that has nothing to do with endogeneity.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # One seasonal path shared by every store, so it lives entirely in the date
    # dimension and time fixed effects can absorb it.
    day_index = np.arange(n_days)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * day_index / 365.25)

    rows = []
    for store in range(n_stores):
        # Chosen so that base + elasticity*log(price) lands around log(60), i.e.
        # roughly 60 units a day. This is not cosmetic: at a lower intercept the
        # term `elasticity * log_price` (about -4.5 at these prices) drives
        # demand to a fraction of a unit, almost every row draws zero, and
        # `prepare_panel` then drops them - which is selection on the outcome
        # and biases every estimator toward zero. The first version of this
        # fixture did exactly that and made the estimators look broken.
        base = np.log(rng.uniform(4_000, 9_000))
        store_effect = rng.normal(0, store_effect_sd)
        # Prices move in steps, as a real price file does. Daily jitter would
        # manufacture variation that does not exist.
        n_changes = 14
        change_days = np.sort(rng.choice(n_days, n_changes, replace=False))
        levels = np.log(rng.uniform(8.0, 16.0, n_changes + 1))
        log_price = np.empty(n_days)
        reasons = np.array(["scheduled"] * n_days, dtype=object)
        start = 0
        for i, day in enumerate([*change_days.tolist(), n_days]):
            log_price[start:day] = levels[i]
            if start > 0:
                reasons[start] = "randomised_test" if i % 3 == 0 else "scheduled"
            start = day

        shock = rng.normal(0, 0.25, n_days)

        # The manager prices into anticipated demand. Which signal they respond
        # to decides whether fixed effects can rescue the estimate.
        if seasonal_endogeneity:
            log_price = log_price + seasonal_endogeneity * seasonal
        if idiosyncratic_endogeneity:
            log_price = log_price + idiosyncratic_endogeneity * shock

        log_demand = base + store_effect + seasonal + elasticity * log_price + shock
        units = rng.poisson(np.exp(np.clip(log_demand, -5, 9)))

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product_id": product_id,
                    "store_id": f"S{store:03d}",
                    "category": "Test",
                    "units": units,
                    "selling_price": np.exp(log_price).round(2),
                    "regular_price": np.exp(log_price).round(2),
                    "promotion_flag": False,
                    "stockout_flag": False,
                    "price_change_reason": reasons,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def make_cost_index(panel: pd.DataFrame, *, seed: int = 3) -> pd.DataFrame:
    """A category-level cost series, matching the platform's shape."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(panel["date"]).drop_duplicates().sort_values()
    return pd.DataFrame(
        {
            "date": dates,
            "category": "Test",
            "cost_index": np.exp(np.cumsum(rng.normal(0, 0.004, len(dates)))),
        }
    )


@pytest.fixture(scope="session")
def clean_panel() -> pd.DataFrame:
    """Exogenous prices and no seasonality: every estimator should recover truth.

    A genuine control. Seasonality is switched off deliberately - it is an
    omitted variable for naive OLS, and with 400 days and 14 price steps the
    chance correlation between a random price path and an annual sine is large
    enough to bias the naive estimate on its own. Leaving it in would make this
    fixture fail for a reason unrelated to the endogeneity it is controlling for.
    """
    return make_panel(seasonal_amplitude=0.0)


@pytest.fixture(scope="session")
def endogenous_panel() -> pd.DataFrame:
    """Price responds to *seasonal* demand, as the platform generator does.

    Naive OLS must be visibly attenuated; fixed effects must fix it, because the
    confounder lives entirely in the date dimension.
    """
    return make_panel(seasonal_endogeneity=0.8)


@pytest.fixture(scope="session")
def idiosyncratic_panel() -> pd.DataFrame:
    """Price responds to the *daily* shock - the case fixed effects cannot fix.

    Exists so the estimator's documented limitation is tested rather than
    merely asserted in a docstring.
    """
    return make_panel(idiosyncratic_endogeneity=0.6)


@pytest.fixture(scope="session")
def cost_index(clean_panel: pd.DataFrame) -> pd.DataFrame:
    return make_cost_index(clean_panel)
