"""Cross-price elasticity model interface (Stage 1 Step 9).

Estimates how a change in product B's price moves product A's demand.

Sign convention, since it is the whole point: a *positive* cross-price
elasticity means the products are substitutes (B gets more expensive, A sells
more). A *negative* value means they are complements (B gets more expensive, A
sells less). Getting the sign backwards inverts every assortment and
cannibalisation conclusion drawn from it.

Scoping note: with N products there are N(N-1) ordered pairs, and estimating all
of them guarantees spurious findings at any conventional significance level.
Implementations should restrict candidates to within-category pairs and known
relationships from ``get_product_relationships``, and correct for multiple
comparisons.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date

from pydantic import BaseModel, Field

from app.schemas.domain import RelationshipType
from ml.base import AnalyticalModel


class CrossElasticityPair(BaseModel):
    """Effect of ``source_product_id``'s price on ``target_product_id``'s demand."""

    source_product_id: str
    target_product_id: str

    #: Positive => substitutes; negative => complements.
    cross_elasticity: float
    relationship_type: RelationshipType
    #: Absolute strength, for ranking. "strong" | "moderate" | "weak" | "none".
    strength: str | None = None

    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    sample_size: int = Field(ge=0)
    is_significant: bool = False


class CrossElasticityResult(BaseModel):
    """Cross-price relationships for one focal product."""

    product_id: str
    region: str | None = None
    pairs: list[CrossElasticityPair] = Field(default_factory=list)

    substitutes: list[str] = Field(default_factory=list)
    complements: list[str] = Field(default_factory=list)

    method: str | None = None
    estimation_window: tuple[date, date] | None = None
    #: Number of pairs tested, needed to interpret significance honestly.
    pairs_tested: int = 0
    multiple_testing_correction: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CrossPriceElasticityModel(AnalyticalModel[CrossElasticityResult]):
    """Cross-price elasticity estimator."""

    name = "cross_price_elasticity"

    @abstractmethod
    def predict(  # type: ignore[override]
        self,
        *,
        product_id: str,
        candidate_product_ids: list[str] | None = None,
        region: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> CrossElasticityResult:
        """Estimate cross-price relationships for the focal product."""
