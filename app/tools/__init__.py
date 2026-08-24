"""Analytical tools - the only route from an agent to a number.

Each tool wraps one deterministic model behind the contract in
``app.schemas.tool_contract``, enforced by ``app.tools.base.AnalyticalTool``.

Planned tools (registered in Stage 1 Step 13, once Steps 4-11 supply the models):

* ``baseline_sales``            - expected sales absent promotions/anomalies
* ``forecast_demand``           - 7/14/30/90-day demand forecast
* ``promo_uplift``              - incremental sales caused by a promotion
* ``trade_promo_optimization``  - budget allocation under constraints
* ``price_elasticity``          - own-price elasticity
* ``cross_price_elasticity``    - substitutes, complements, cannibalisation
* ``price_optimization``        - recommended price and its defensible range
* ``scenario_simulation``       - composed what-if projection

Agents receive tool *names* and structured results. They never see a DataFrame,
a model object, a file path or a database connection.
"""
