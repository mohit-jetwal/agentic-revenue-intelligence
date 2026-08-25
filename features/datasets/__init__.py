"""Model-ready dataset builders (brief sections 36-40).

One builder per future model. Each takes a point-in-time view, returns a
``FeatureSet`` with ``X`` and ``y`` separated, and records lineage.

These assemble inputs; they do not fit anything. What they *do* encode is each
model's framing - which rows are eligible, what the target is, what must be
excluded to keep the estimate honest. That belongs here, because a dataset built
on the wrong framing cannot be rescued by a better model.
"""

from features.datasets.builders import (
    UpliftWindows,
    create_cross_price_dataset,
    create_forecasting_dataset,
    create_price_elasticity_dataset,
    create_promo_optimization_dataset,
    create_promo_uplift_dataset,
)

__all__ = [
    "UpliftWindows",
    "create_cross_price_dataset",
    "create_forecasting_dataset",
    "create_price_elasticity_dataset",
    "create_promo_optimization_dataset",
    "create_promo_uplift_dataset",
]
