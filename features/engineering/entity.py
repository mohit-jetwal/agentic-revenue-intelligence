"""Product and store features (brief sections 20-21).

Entity attributes are static per key, so they carry no leakage risk of their own
- with one exception, which is why ``product_age_days`` and ``store_age_days``
live here rather than being computed inline. Both are *relative to the row's
date*, so they are time-varying derivations of a static attribute, and a careless
implementation computes them against "today" instead of against the row. That
would embed the moment the feature set was built into the training data, and a
model retrained a month later would see different values for the same historical
rows.

On encoding (section 21's "avoid unnecessary high-cardinality encoding"):
identifiers and names are deliberately **not** one-hot encoded here. With 300
products and 200 stores, dummies would add 500 columns. Categorical dtype is
emitted instead - LightGBM and XGBoost consume it natively, and a model that
genuinely wants target or ordinal encoding should fit that encoder on its own
training fold, because fitting it here would leak the target across folds.
"""

from __future__ import annotations

import pandas as pd

from features.engineering.panel import DATE_KEY

#: Low-cardinality nominal attributes, safe to encode as categoricals.
PRODUCT_ATTRIBUTES: tuple[str, ...] = (
    "category",
    "subcategory",
    "brand",
    "pack_size",
    "product_status",
)
STORE_ATTRIBUTES: tuple[str, ...] = ("store_type", "channel", "region", "state")

#: High-cardinality fields deliberately left alone. Named so the omission is
#: visibly a decision rather than an oversight.
HIGH_CARDINALITY: tuple[str, ...] = ("product_name", "store_name", "city", "customer_id")


def add_product_features(
    panel: pd.DataFrame,
    products: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
    attributes: tuple[str, ...] = PRODUCT_ATTRIBUTES,
    include_economics: bool = True,
) -> pd.DataFrame:
    """Attach product attributes and lifecycle age."""
    available = [c for c in attributes if c in products.columns]
    columns = ["product_id", *available]

    if include_economics:
        # Unit cost is a business input known ahead, and margin work needs it.
        columns += [c for c in ("unit_cost", "base_price") if c in products.columns]
    if "launch_date" in products.columns:
        columns.append("launch_date")

    right = products[list(dict.fromkeys(columns))].drop_duplicates("product_id")

    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    overlapping = [c for c in right.columns if c in result.columns and c != "product_id"]
    if overlapping:
        result = result.drop(columns=overlapping)
    result = result.merge(right, on="product_id", how="left")

    if "launch_date" in result.columns:
        # Age as at the row's own date, never as at "now" - see module docstring.
        result["product_age_days"] = (
            result[date_column] - pd.to_datetime(result["launch_date"])
        ).dt.days
        # Products in their first quarter behave differently: distribution is
        # still building, so demand is suppressed for reasons unrelated to price.
        result["is_new_product"] = result["product_age_days"] < 90
        result = result.drop(columns=["launch_date"])

    for column in available:
        result[column] = result[column].astype("category")

    return result


def add_store_features(
    panel: pd.DataFrame,
    stores: pd.DataFrame,
    *,
    date_column: str = DATE_KEY,
    attributes: tuple[str, ...] = STORE_ATTRIBUTES,
) -> pd.DataFrame:
    """Attach store attributes, size and age."""
    available = [c for c in attributes if c in stores.columns]
    columns = ["store_id", *available]
    columns += [c for c in ("store_size_sqft", "opening_date") if c in stores.columns]

    right = stores[list(dict.fromkeys(columns))].drop_duplicates("store_id")

    result = panel.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    overlapping = [c for c in right.columns if c in result.columns and c != "store_id"]
    if overlapping:
        result = result.drop(columns=overlapping)
    result = result.merge(right, on="store_id", how="left")

    if "opening_date" in result.columns:
        result["store_age_days"] = (
            result[date_column] - pd.to_datetime(result["opening_date"])
        ).dt.days
        result = result.drop(columns=["opening_date"])

    for column in available:
        result[column] = result[column].astype("category")

    return result


def drop_high_cardinality(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove free-text and identifier-like columns from a feature frame.

    Panel keys (``product_id``, ``store_id``, ``date``) are kept: they identify
    rows and are needed for joins, grouping and evaluation. It is the *names*
    and free text that carry no signal a tree model can use and that bloat an
    encoder if one is fitted downstream.
    """
    present = [c for c in HIGH_CARDINALITY if c in frame.columns]
    return frame.drop(columns=present) if present else frame
