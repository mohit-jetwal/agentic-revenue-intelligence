"""Fixtures for the demand forecasting tests (Step 5).

Built on the session-scoped smoke dataset from :mod:`tests.dataset_fixtures`,
which inherits ``dev``'s full 2023-01-01..2025-12-31 span at 40 products x 30
stores. That matters: the forecasting split needs train + calibration +
validation + test *plus* a 90-day embargo, so a short panel would fail every
split test for reasons that have nothing to do with the split logic.

The horizon dataset is expensive to build (a feature panel, then a self-join),
so it is session-scoped and shared. Tests must therefore treat it as read-only -
mutating it would produce failures in unrelated tests that look nothing like
their cause.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.repositories.local import LocalDataRepository
from data.repositories.point_in_time import PointInTimeView
from ml.forecasting.config import ForecastConfig, load_forecast_config
from ml.forecasting.dataset import HorizonDataset, build_history, build_horizon_dataset
from ml.forecasting.sampling import SeriesSample, sample_series


@pytest.fixture(scope="session")
def forecast_config() -> ForecastConfig:
    """The smoke variant: small enough for a test run, structurally identical.

    Everything that could be wrong - splits, embargo, buckets, censoring, the
    origin/target join - behaves the same here as at full scale, so a bug in
    that machinery surfaces in the test suite rather than only in a 20-minute
    training run.
    """
    return load_forecast_config().smoke()


@pytest.fixture(scope="session")
def forecast_sample(
    smoke_repository: LocalDataRepository, forecast_config: ForecastConfig
) -> SeriesSample:
    return sample_series(
        smoke_repository,
        n_series=forecast_config.sampling.n_series,
        seed=forecast_config.sampling.seed,
    )


@pytest.fixture(scope="session")
def forecast_history(
    smoke_repository: LocalDataRepository,
    forecast_config: ForecastConfig,
    forecast_sample: SeriesSample,
) -> pd.DataFrame:
    """The historical panel origin-side features are drawn from."""
    return build_history(smoke_repository, forecast_config, forecast_sample)


@pytest.fixture(scope="session")
def forecast_as_of(forecast_history: pd.DataFrame) -> date:
    return pd.to_datetime(forecast_history["date"]).dt.date.max()


@pytest.fixture(scope="session")
def forecast_view(
    smoke_repository: LocalDataRepository, forecast_as_of: date
) -> PointInTimeView:
    return smoke_repository.as_of(forecast_as_of)


@pytest.fixture(scope="session")
def horizon_dataset(
    forecast_history: pd.DataFrame,
    forecast_view: PointInTimeView,
    forecast_config: ForecastConfig,
    forecast_sample: SeriesSample,
) -> HorizonDataset:
    """The (origin, horizon, target) training rows. Read-only."""
    return build_horizon_dataset(
        forecast_history, forecast_view, forecast_config, forecast_sample
    )
