"""Shared fixtures for the generated dataset.

Imported into the root ``conftest`` so both ``tests/data`` and
``tests/statistical`` see them. Kept in their own module rather than inlined
into ``conftest.py`` so the dataset-specific setup stays separate from the
application fixtures (settings, container, API client) that every test uses.

The smoke dataset is generated **once per session** and shared. Generating it
per test would multiply a ~4 second cost across the whole suite; once keeps the
full run inside the sub-two-minute budget the project holds itself to.

It is written to a temporary directory rather than ``data/local`` so a test run
can never overwrite a developer's working dataset - a particularly annoying
failure to diagnose, because everything would still pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.generation.config import GenerationConfig, load_config
from data.generation.ground_truth import GroundTruth
from data.generation.pipeline import GenerationResult, generate_dataset
from data.validation.report import load_gold_tables, load_latent_demand

SMOKE_SEED = 42


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("smoke_dataset")


@pytest.fixture(scope="session")
def smoke_result(smoke_root: Path) -> GenerationResult:
    """Generate the smoke dataset once for the whole session."""
    return generate_dataset("smoke", seed=SMOKE_SEED, output_root=smoke_root)


@pytest.fixture(scope="session")
def smoke_tables(smoke_result: GenerationResult) -> dict[str, pd.DataFrame]:
    return load_gold_tables(smoke_result.root, sample_rows=None)


@pytest.fixture(scope="session")
def smoke_ground_truth(smoke_result: GenerationResult) -> GroundTruth:
    return GroundTruth.load(smoke_result.root)


@pytest.fixture(scope="session")
def smoke_latent(smoke_result: GenerationResult) -> pd.DataFrame:
    return load_latent_demand(smoke_result.root)


@pytest.fixture(scope="session")
def smoke_config() -> GenerationConfig:
    return load_config("smoke", overrides={"seed": SMOKE_SEED})


@pytest.fixture
def sales(smoke_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return smoke_tables["sales_daily"]


@pytest.fixture
def inventory(smoke_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return smoke_tables["inventory"]


@pytest.fixture
def products(smoke_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return smoke_tables["products"]


@pytest.fixture
def stores(smoke_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return smoke_tables["stores"]


@pytest.fixture
def promotions(smoke_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return smoke_tables["promotions"]


@pytest.fixture(scope="session")
def second_run(tmp_path_factory: pytest.TempPathFactory) -> GenerationResult:
    """A second generation with the same seed, for the reproducibility test."""
    root = tmp_path_factory.mktemp("smoke_repeat")
    return generate_dataset("smoke", seed=SMOKE_SEED, output_root=root)
