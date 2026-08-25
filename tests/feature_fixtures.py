"""Shared fixtures for repository, feature and leakage tests.

Built on the session-scoped smoke dataset from
:mod:`tests.dataset_fixtures`, so Step 3's tests reuse Step 2's generated data
rather than producing their own - one generation per session, not two.

The sample is deliberately **co-listed**: products and stores that genuinely
appear together in the sales fact. Sampling the two independently produces pairs
that were never stocked together, and a feature test on those fails for reasons
of sparsity rather than correctness.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from data.repositories.local import LocalDataRepository
from data.repositories.point_in_time import PointInTimeView
from data.repositories.sampling import PanelSample, build_panel_sample
from features.engineering import FeatureEngineer, FeatureRequest
from features.repositories import LocalFeatureRepository


@pytest.fixture(scope="session")
def smoke_repository(smoke_result: object) -> LocalDataRepository:
    """Repository over the session smoke dataset.

    A generous row cap: feature building over a multi-month window legitimately
    exceeds the production default, and the truncation guard would otherwise
    raise on perfectly reasonable test queries.
    """
    return LocalDataRepository(
        parquet_root=smoke_result.root / "gold",  # type: ignore[attr-defined]
        max_result_rows=5_000_000,
    )


@pytest.fixture(scope="session")
def smoke_panel_sample(smoke_repository: LocalDataRepository) -> PanelSample:
    """A dense, co-listed slice of the smoke panel."""
    return build_panel_sample(smoke_repository, n_products=6, n_stores=5, days=240, seed=42)


@pytest.fixture(scope="session")
def smoke_as_of(smoke_panel_sample: PanelSample) -> date:
    """As-of date inside the sample window, leaving future data to leak *from*.

    Set 30 days before the sample end so there is genuinely later data present -
    a cut at the very end of history would let a leaking pipeline pass by
    accident.
    """
    return smoke_panel_sample.end_date - timedelta(days=30)


@pytest.fixture(scope="session")
def smoke_view(smoke_repository: LocalDataRepository, smoke_as_of: date) -> PointInTimeView:
    return smoke_repository.as_of(smoke_as_of)


@pytest.fixture(scope="session")
def smoke_features(
    smoke_view: PointInTimeView, smoke_panel_sample: PanelSample, smoke_as_of: date
) -> pd.DataFrame:
    """A built feature panel, computed once and shared across tests."""
    engineer = FeatureEngineer(smoke_view)
    return engineer.build(
        FeatureRequest(
            start_date=smoke_as_of - timedelta(days=120),
            end_date=smoke_as_of,
            product_ids=smoke_panel_sample.product_ids,
            store_ids=smoke_panel_sample.store_ids,
        )
    )


@pytest.fixture
def feature_repository(smoke_view: PointInTimeView) -> LocalFeatureRepository:
    return LocalFeatureRepository(smoke_view)
