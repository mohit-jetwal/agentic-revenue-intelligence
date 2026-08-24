"""Deterministic analytical models.

Eight models, each behind an interface in ``<package>/interface.py``:

* ``baseline``                  - expected sales absent promotions and anomalies
* ``forecasting``               - 7/14/30/90-day demand forecast
* ``promo_uplift``              - incremental sales caused by a promotion
* ``trade_promo_optimization``  - budget allocation under constraints
* ``price_elasticity``          - own-price elasticity
* ``cross_price_elasticity``    - substitutes, complements, cannibalisation
* ``price_optimization``        - recommended price and its defensible range
* ``scenario``                  - composed what-if simulation

These are deterministic by contract: the same inputs must produce the same
numbers. That is what makes them trustworthy as the sole source of figures in a
system whose reasoning layer is not deterministic.

Every model reads through a ``DataRepository`` and carries ``ModelMetadata``, so
any number it produces is attributable to a version, a dataset and an MLflow run.

Implemented in Stage 1 Steps 4-11.
"""
