"""Analytical tools - the only route from an agent to a number.

Each tool wraps one deterministic model behind the contract in
``app.schemas.tool_contract``, enforced by ``app.tools.base.AnalyticalTool``.

Registered tools:

* ``forecast_demand``            - 7/14/28/30/90-day demand forecast
* ``estimate_promo_uplift``      - incremental sales *caused* by a promotion
* ``estimate_price_elasticity``  - own-price elasticity, plus substitutes and
                                   complements for cannibalisation questions
* ``allocate_promotion_budget``  - budget allocation under constraints
* ``optimize_price``             - recommended price and its defensible range
* ``simulate_scenario``          - composed what-if projection

``baseline_sales`` is deliberately not registered. It answers "what would normal
sales have been", which is an input to uplift rather than a question anyone
asks - exposing it would invite an agent to compute uplift itself by
subtraction, which is the naive estimate the whole causal layer exists to avoid.

Cross-price elasticity is reached through ``estimate_price_elasticity`` rather
than as its own tool: a substitute matters when you are pricing something, and a
separate tool would let an agent ask about cannibalisation without ever
establishing the own-price effect it is relative to.

Agents receive tool *names* and structured results. They never see a DataFrame,
a model object, a file path or a database connection.
"""
