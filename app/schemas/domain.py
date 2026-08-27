"""Core CPG/Retail domain vocabulary.

Shared enums used across schemas, tools, models and prompts. Keeping these in
one place stops the same concept being spelled three different ways in three
layers (``"North"`` / ``"north"`` / ``"NORTH"``), which is the usual source of
silent join failures and mis-parsed LLM output.
"""

from __future__ import annotations

from enum import StrEnum


class Region(StrEnum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"
    CENTRAL = "Central"


class Channel(StrEnum):
    HYPERMARKET = "Hypermarket"
    SUPERMARKET = "Supermarket"
    CONVENIENCE = "Convenience"
    ECOMMERCE = "E-commerce"
    PHARMACY = "Pharmacy"
    WHOLESALE = "Wholesale"


class PromotionType(StrEnum):
    PRICE_DISCOUNT = "Price Discount"
    BOGO = "Buy One Get One"
    BUNDLE = "Bundle"
    DISPLAY = "Display"
    COUPON = "Coupon"
    LOYALTY_OFFER = "Loyalty Offer"


class CustomerSegment(StrEnum):
    VALUE = "Value"
    REGULAR = "Regular"
    PREMIUM = "Premium"
    LOYAL = "Loyal"
    OCCASIONAL = "Occasional"


class RelationshipType(StrEnum):
    """Cross-price relationship between two products."""

    SUBSTITUTE = "substitute"
    COMPLEMENT = "complement"
    UNRELATED = "unrelated"


class IntentType(StrEnum):
    """What the user is asking for. Produced by the Supervisor's classifier.

    Drives the initial plan shape. A ``FORECAST`` question must not fan out to
    elasticity and optimisation tools (brief section 6); a ``TRADE_OFF``
    question legitimately may (section 7).
    """

    FORECAST = "forecast"
    PERFORMANCE_EXPLANATION = "performance_explanation"
    ROOT_CAUSE = "root_cause"
    PRICE_DECISION = "price_decision"
    PROMOTION_DECISION = "promotion_decision"
    BUDGET_ALLOCATION = "budget_allocation"
    SCENARIO_WHAT_IF = "scenario_what_if"
    TRADE_OFF = "trade_off"
    DATA_LOOKUP = "data_lookup"
    POLICY_QUESTION = "policy_question"
    OUT_OF_SCOPE = "out_of_scope"


class BusinessObjective(StrEnum):
    """The commercial objective an investigation is optimising for."""

    MAXIMISE_REVENUE = "maximise_revenue"
    MAXIMISE_PROFIT = "maximise_profit"
    MAXIMISE_VOLUME = "maximise_volume"
    PROTECT_MARGIN = "protect_margin"
    EXPLAIN_PERFORMANCE = "explain_performance"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ForecastHorizon(StrEnum):
    """Supported forecast horizons.

    ``D28`` is the retail planning horizon - four whole weeks. It matters more
    than the round-number alternatives because demand is strongly weekly: a
    28-day window contains exactly four of each weekday, so the total is not
    skewed by whichever days happen to fall inside it. A 30-day window contains
    four of some weekdays and five of others, which quietly biases the total
    toward whatever those two extra days happen to be.

    Adding it required no retraining. The model is fitted on horizon steps drawn
    from ``U{1..90}`` with the step itself as a feature, and 28 already falls
    inside the calibrated ``h15-28`` interval bucket - so the existing artifact
    serves it directly.
    """

    D7 = "7d"
    D14 = "14d"
    D28 = "28d"
    D30 = "30d"
    D90 = "90d"

    @property
    def days(self) -> int:
        return int(self.value.removesuffix("d"))
