"""Feature engineering layer.

Sits between the data repositories and the ML models, so that a model receives a
standardised feature frame and never learns where the data came from:

    DataRepository -> PointInTimeView -> FeatureEngineer -> FeatureRepository -> model

Packages:

* ``engineering`` - reusable primitives (lags, rolling, price, promotion, ...)
* ``contracts``   - feature definitions, groups, model requirements, lineage
* ``repositories``- feature access, with opt-in materialisation
* ``datasets``    - the five model-ready dataset builders

The organising constraint is point-in-time correctness. Feature builders accept
a :class:`~data.repositories.point_in_time.PointInTimeView` rather than a bare
repository, so a feature cannot read future observed data - not because someone
remembered to pass a flag, but because the object has no method that would
return it.
"""
