"""Feature engineering primitives.

Reusable, composable transformations over a product-store-date panel. No
model-specific pipelines live here - those are in ``features/datasets``, built
from these parts.

Every temporal primitive routes through :mod:`features.engineering.panel`, which
owns the shift discipline: a feature used to predict day *t* is computed from
data strictly before *t*. Concentrating that in one helper is what makes it
testable and what stops the twentieth call site from quietly getting it wrong.

Modules:

* ``panel``      - shift discipline and panel preparation. Read this first.
* ``demand``     - lags, rolling statistics, momentum, volatility
* ``temporal``   - calendar, cyclical encodings, festival proximity
* ``pricing``    - own price, price index, competitor position
* ``promotion``  - schedule features (forward-looking) and history (backward)
* ``inventory``  - availability, stockout history; censored columns excluded
* ``entity``     - product and store attributes, lifecycle age
* ``engineer``   - the facade that composes them in the correct order
"""

from features.engineering.engineer import FeatureEngineer, FeatureRequest
from features.engineering.panel import (
    PANEL_KEYS,
    prepare_panel,
    rolling_on_shifted,
    shifted_group,
)

__all__ = [
    "PANEL_KEYS",
    "FeatureEngineer",
    "FeatureRequest",
    "prepare_panel",
    "rolling_on_shifted",
    "shifted_group",
]
