"""Feature contracts: what each feature is, and what each model needs.

``specs`` defines the vocabulary (:class:`FeatureSpec`, :class:`FeatureGroup`,
:class:`FeatureSetMetadata`). ``catalogue`` declares every feature the platform
produces and every future model's requirements. ``config`` loads the YAML that
selects which of them a given dataset uses.

The ``temporality`` field carries the weight. It is the machine-readable form of
the availability reasoning in :mod:`data.repositories.availability`, the leakage
tests assert against it, and a ``FORWARD_PLANNED`` feature is rejected at
construction unless it states why its information is genuinely knowable at
prediction time.
"""

from features.contracts.catalogue import (
    FEATURE_GROUPS,
    FEATURE_SPECS,
    MODEL_REQUIREMENTS,
    features_in_group,
    forward_looking_features,
    requirement_for,
    spec_for,
)
from features.contracts.config import (
    DatasetSelection,
    FeatureConfig,
    FeatureConfigError,
    load_feature_config,
)
from features.contracts.specs import (
    FEATURE_VERSION,
    FeatureGroup,
    FeatureGroupName,
    FeatureSetMetadata,
    FeatureSpec,
    ModelFeatureRequirement,
    Temporality,
    current_code_version,
    hash_request,
)

__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_SPECS",
    "FEATURE_VERSION",
    "MODEL_REQUIREMENTS",
    "DatasetSelection",
    "FeatureConfig",
    "FeatureConfigError",
    "FeatureGroup",
    "FeatureGroupName",
    "FeatureSetMetadata",
    "FeatureSpec",
    "ModelFeatureRequirement",
    "Temporality",
    "current_code_version",
    "features_in_group",
    "forward_looking_features",
    "hash_request",
    "load_feature_config",
    "requirement_for",
    "spec_for",
]
