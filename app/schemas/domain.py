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
    """Supported forecast horizons (brief section 8)."""

    D7 = "7d"
    D14 = "14d"
    D30 = "30d"
    D90 = "90d"

    @property
    def days(self) -> int:
        return int(self.value.removesuffix("d"))
