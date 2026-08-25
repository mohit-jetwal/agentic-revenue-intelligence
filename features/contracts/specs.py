"""Feature specifications, groups and lineage (brief sections 22, 25, 31).

Three things live here, and they exist for one reason: when a model in Step 8
reports an elasticity of -1.42, somebody has to be able to answer *what exactly
went into that*. Without recorded feature definitions and versions, the honest
answer is "whatever the code looked like at the time", which is not an answer.

* :class:`FeatureSpec` - what one feature is, where it comes from, and whether
  it may look forward.
* :class:`FeatureGroup` - the logical bundles from section 22.
* :class:`FeatureSetMetadata` - lineage for a materialised feature set, carrying
  the dataset version, feature version and as-of date that MLflow will need in
  Step 12 to tie a model run to its exact inputs.

The ``temporality`` field on a spec is the machine-readable form of the
reasoning in :mod:`data.repositories.availability`. It is what the leakage tests
assert against, so a feature that starts reaching forward without its spec
changing is caught.
"""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404 - used only to read the local git commit
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Bumped when a feature *definition* changes in a way that alters its values.
#: Adding a new feature does not require a bump; changing how an existing one is
#: computed does, because otherwise two incompatible feature sets share a
#: version and a model comparison becomes meaningless.
FEATURE_VERSION = "v1.0"


class Temporality(StrEnum):
    """Whether a feature may use information from on or after its row date."""

    #: Computed strictly from data before the row's date. Lags, rolling stats.
    BACKWARD = "backward"
    #: Uses the row's own date only, from data knowable in advance. Calendar,
    #: today's price, the promotion schedule.
    CONTEMPORANEOUS = "contemporaneous"
    #: Legitimately reads beyond the row date, because the information is
    #: planned and committed. `days_to_next_promotion`, `days_to_festival`.
    #: Every member of this class needs a written justification.
    FORWARD_PLANNED = "forward_planned"
    #: The prediction target. Never a feature.
    TARGET = "target"


class FeatureGroupName(StrEnum):
    """Logical bundles from brief section 22."""

    DEMAND = "demand"
    PRICE = "price"
    PROMOTION = "promotion"
    INVENTORY = "inventory"
    COMPETITOR = "competitor"
    TEMPORAL = "temporal"
    PRODUCT = "product"
    STORE = "store"


class FeatureSpec(BaseModel):
    """Definition of a single feature."""

    model_config = ConfigDict(frozen=True)

    name: str
    group: FeatureGroupName
    description: str
    #: Columns in the source tables this feature derives from.
    source_columns: tuple[str, ...] = ()
    #: Tables it reads.
    source_tables: tuple[str, ...] = ()
    #: How it is computed, in words. Deliberately prose rather than code: the
    #: point is that a reviewer can check the code against the intent.
    transformation: str = ""
    temporality: Temporality = Temporality.BACKWARD
    dtype: str = "float"
    #: Required for FORWARD_PLANNED features. Enforced below, because an
    #: unjustified forward-looking feature is exactly what leakage looks like.
    forward_justification: str | None = None
    feature_version: str = FEATURE_VERSION

    def model_post_init(self, _context: Any) -> None:
        if self.temporality is Temporality.FORWARD_PLANNED and not self.forward_justification:
            raise ValueError(
                f"feature {self.name!r} is FORWARD_PLANNED but carries no "
                f"justification. A forward-looking feature must state why the "
                f"information is genuinely knowable at prediction time."
            )


class FeatureGroup(BaseModel):
    """A named bundle of features."""

    model_config = ConfigDict(frozen=True)

    name: FeatureGroupName
    description: str
    features: tuple[FeatureSpec, ...]

    def names(self) -> list[str]:
        return [spec.name for spec in self.features]

    def by_temporality(self, temporality: Temporality) -> list[FeatureSpec]:
        return [spec for spec in self.features if spec.temporality is temporality]


class ModelFeatureRequirement(BaseModel):
    """What one future model needs (brief section 24).

    Recorded now, before any model exists, so Steps 4-11 inherit a stated
    contract rather than each inventing its own inputs and discovering the
    overlap later.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    description: str
    required_groups: tuple[FeatureGroupName, ...]
    #: Individual features the model cannot work without.
    required_features: tuple[str, ...] = ()
    target: str | None = None
    #: Notes on what the model must be careful about - usually a leakage or
    #: identification hazard specific to it.
    caveats: tuple[str, ...] = ()


class FeatureSetMetadata(BaseModel):
    """Lineage for one materialised feature set (brief section 31).

    This is the record that lets a model run in Step 12 be reproduced: same
    dataset version, same feature version, same as-of date, same filters.
    """

    model_config = ConfigDict(frozen=True)

    feature_set_name: str
    feature_version: str = FEATURE_VERSION
    #: From the Step 2 manifest, qualified by as-of date when built via a view.
    dataset_version: str = "unknown"
    as_of_date: date | None = None
    start_date: date | None = None
    end_date: date | None = None

    source_tables: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    target_name: str | None = None

    row_count: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    code_version: str | None = None
    #: Hash of the request parameters, so two feature sets built from different
    #: filters cannot be mistaken for one another.
    request_hash: str | None = None

    def cache_key(self) -> str:
        """Stable key for materialisation.

        Includes the feature version, so bumping it invalidates every cached
        set - which is the behaviour you want, since the definitions changed.
        """
        parts = [
            self.feature_set_name,
            self.feature_version,
            self.dataset_version,
            str(self.as_of_date),
            str(self.start_date),
            str(self.end_date),
            self.request_hash or "",
        ]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return f"{self.feature_set_name}__{self.feature_version}__{digest}"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True, default=str)


def current_code_version() -> str | None:
    """Short git commit for lineage, or ``None`` outside a repository.

    Best-effort by design: a missing commit should degrade the metadata, never
    fail a feature build.
    """
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, no user input
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def hash_request(payload: dict[str, Any]) -> str:
    """Stable hash of request parameters for the cache key."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
