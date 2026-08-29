"""Building the elasticity panel from the repository.

One join, in one place. The model, the CLI and the validation script all call
:func:`build_elasticity_panel`, because the join decides which rows carry price
variation — and two copies of that would drift apart.

Three joins that are not optional:

* **products** for ``category``, which the cost instrument keys on and which
  scopes the cross-price candidate set.
* **pricing** for ``price_change_reason``, which is how the exogenous
  randomised-test subset is identified.
* **commodity_costs** for the instrument.

Without the second, the randomised estimator cannot run at all; without the
third, 2SLS cannot. Both then vanish from the comparison table with a reason
attached rather than silently.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.observability.logging import get_logger
from data.repositories.base import DataRepository

logger = get_logger(__name__)

#: Explicit row cap. The repository guards at 100,000 rows and raises rather
#: than truncating - right for an interactive query, wrong for a deliberate
#: panel build, so the limit is opted out of here rather than raised globally.
MAX_PANEL_ROWS = 20_000_000


def build_elasticity_panel(
    repository: DataRepository,
    *,
    product_ids: list[str] | None = None,
    store_ids: list[str] | None = None,
    region: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Sales joined to price reasons, product attributes and the cost index."""
    sales = repository.get_sales(
        product_ids=product_ids,
        store_ids=store_ids,
        region=region,
        start_date=start_date,
        end_date=end_date,
        max_rows=MAX_PANEL_ROWS,
    )
    if sales.empty:
        return sales

    sales["date"] = pd.to_datetime(sales["date"])

    products = repository.get_products()
    if not products.empty:
        columns = [
            c for c in ("product_id", "category", "brand", "unit_cost") if c in products.columns
        ]
        sales = sales.merge(products[columns], on="product_id", how="left")

    pricing = repository.get_pricing(
        product_ids=product_ids,
        store_ids=store_ids,
        start_date=start_date,
        end_date=end_date,
        max_rows=MAX_PANEL_ROWS,
    )
    if not pricing.empty and "price_change_reason" in pricing.columns:
        pricing["date"] = pd.to_datetime(pricing["date"])
        keep = ["date", "product_id", "store_id", "price_change_reason"]
        # `regular_price` may already be on the sales fact; taking it twice
        # produces _x/_y suffixes and silently breaks every downstream reference.
        if "regular_price" not in sales.columns and "regular_price" in pricing.columns:
            keep.append("regular_price")
        sales = sales.merge(pricing[keep], on=["date", "product_id", "store_id"], how="left")

    logger.info(
        "elasticity.panel_built",
        rows=len(sales),
        products=int(sales["product_id"].nunique()),
        stores=int(sales["store_id"].nunique()),
    )
    return sales.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)


def load_cost_index(
    repository: DataRepository,
    *,
    categories: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """The commodity cost index, for the 2SLS instrument.

    Returns an empty frame rather than raising when unavailable. 2SLS is one
    row of a four-row comparison, and on this data it is the row that does not
    work — its absence should not stop the other three.
    """
    try:
        return repository.get_commodity_costs(
            categories=categories,
            start_date=start_date,
            end_date=end_date,
            max_rows=MAX_PANEL_ROWS,
        )
    except (NotImplementedError, AttributeError, ValueError) as exc:
        logger.info("elasticity.cost_index_unavailable", error=str(exc))
        return pd.DataFrame()


__all__ = ["MAX_PANEL_ROWS", "build_elasticity_panel", "load_cost_index"]
