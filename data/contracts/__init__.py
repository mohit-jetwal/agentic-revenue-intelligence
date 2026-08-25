"""Data contracts - schema validation at the repository boundary.

Two validation layers exist in this project and they do different jobs:

* ``data/contracts`` (here) - the **shape of a frame crossing an interface**:
  columns present, dtypes right, values in range, keys not null, cross-column
  identities hold. Pandera. Runs at the repository boundary and at feature
  builder inputs.
* ``data/validation`` (Step 2) - **cross-row business invariants over a whole
  dataset**: does price vary enough for elasticity to be identified, do
  stockouts actually suppress sales. Hand-rolled, because those are statements
  about a body of evidence rather than about a schema.

Neither replaces the other, and neither is duplicated in the other.
"""

from data.contracts.base import (
    ContractViolationError,
    DataContract,
    check_primary_key,
    validate,
)
from data.contracts.tables import (
    CALENDAR_CONTRACT,
    COMPETITOR_PRICING_CONTRACT,
    CONTRACTS,
    CUSTOMERS_CONTRACT,
    INVENTORY_CONTRACT,
    PRICING_CONTRACT,
    PRODUCT_RELATIONSHIPS_CONTRACT,
    PRODUCTS_CONTRACT,
    PROMOTIONS_CONTRACT,
    SALES_CONTRACT,
    STORES_CONTRACT,
    TRADE_PROMOTIONS_CONTRACT,
    contract_for,
)

__all__ = [
    "CALENDAR_CONTRACT",
    "COMPETITOR_PRICING_CONTRACT",
    "CONTRACTS",
    "CUSTOMERS_CONTRACT",
    "INVENTORY_CONTRACT",
    "PRICING_CONTRACT",
    "PRODUCTS_CONTRACT",
    "PRODUCT_RELATIONSHIPS_CONTRACT",
    "PROMOTIONS_CONTRACT",
    "SALES_CONTRACT",
    "STORES_CONTRACT",
    "TRADE_PROMOTIONS_CONTRACT",
    "ContractViolationError",
    "DataContract",
    "check_primary_key",
    "contract_for",
    "validate",
]
