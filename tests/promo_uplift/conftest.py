"""Shared fixtures for the promo uplift tests.

The expensive objects - a generated panel, cross-fitted nuisance models - are
session-scoped and built once. Fitting them per test would put the suite into
minutes, and a slow suite gets run less often, which is how correctness bugs
survive. Step 6 learned that directly.

Everything here is built with the **shipped** configuration, only smaller. A
fixture that relaxed the overlap rules or the control thresholds would validate a
system nobody deploys.
"""

from __future__ import annotations

import pytest

from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config
from ml.promo_uplift.controls import ControlPool, build_control_pool
from ml.promo_uplift.estimators import NuisanceFit, fit_nuisances
from ml.promo_uplift.features import CovariateFrame, build_covariates
from ml.promo_uplift.synthetic import SyntheticPanel, generate, scenario_config
from ml.promo_uplift.treatment import AnalysisFrame, build_analysis_frame

#: Small enough to keep the suite interactive, large enough that the estimators
#: are not dominated by sampling noise. Measured: the confounded scenario at this
#: size recovers the true ATT to within a couple of points.
TEST_SERIES = 60
TEST_DAYS = 300


@pytest.fixture(scope="session")
def base_config() -> PromoUpliftConfig:
    """The shipped configuration, shrunk. Nothing structural is relaxed."""
    return get_promo_uplift_config().smoke()


@pytest.fixture(scope="session")
def confounded_panel(base_config: PromoUpliftConfig) -> SyntheticPanel:
    """A panel with a real effect and targeted assignment - the realistic case."""
    return generate(
        "confounded",
        config=base_config,
        n_series=TEST_SERIES,
        n_days=TEST_DAYS,
        seed=7,
    )


@pytest.fixture(scope="session")
def confounded_config(base_config: PromoUpliftConfig) -> PromoUpliftConfig:
    return scenario_config("confounded", base_config)


@pytest.fixture(scope="session")
def analysis(
    confounded_panel: SyntheticPanel, confounded_config: PromoUpliftConfig
) -> AnalysisFrame:
    return build_analysis_frame(confounded_panel.observable(), config=confounded_config)


@pytest.fixture(scope="session")
def pool(analysis: AnalysisFrame, confounded_config: PromoUpliftConfig) -> ControlPool:
    return build_control_pool(analysis, config=confounded_config)


@pytest.fixture(scope="session")
def covariates(
    pool: ControlPool, analysis: AnalysisFrame, confounded_config: PromoUpliftConfig
) -> CovariateFrame:
    return build_covariates(
        pool.frame, analysis.events, config=confounded_config, history=analysis.frame
    )


@pytest.fixture(scope="session")
def nuisance(
    covariates: CovariateFrame, confounded_config: PromoUpliftConfig
) -> NuisanceFit:
    return fit_nuisances(covariates, config=confounded_config)
