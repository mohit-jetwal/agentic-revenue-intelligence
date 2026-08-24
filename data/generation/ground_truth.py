"""Hidden simulation parameters - the ground truth.

This is the module that makes the whole dataset falsifiable.

Every relationship the platform later claims to *estimate* is drawn here first,
recorded, and only then used to generate sales. That inverts the usual synthetic
data problem: instead of "the model produced -1.38, is that right?" with no way
to answer, Step 8 can compare -1.38 against a known -1.42 and report the error.
It also allows the sharper test - that a *naive* specification is biased in a
predictable direction, which is what proves the confounding in the generator is
doing its job.

Two rules govern this data:

1. It is written to ``ground_truth/``, structurally separate from ``gold/``.
2. ``LocalDataRepository`` has no method that can reach it. No tool, model or
   agent can read it, by construction rather than by convention.

Only tests and the validation report may load these files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generation.coerce import as_float
from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

GROUND_TRUTH_DIRNAME = "ground_truth"


@dataclass
class GroundTruth:
    """Latent parameters behind the generated data."""

    #: product_id -> true own-price elasticity.
    own_elasticity: dict[str, float]
    #: target_product -> {price_product: cross-price elasticity}.
    cross_elasticity: dict[str, dict[str, float]]
    #: product_id -> {"a": saturation ceiling, "b": curvature} per promotion type.
    promo_response: dict[str, dict[str, dict[str, float]]]
    #: product_id -> competitor cross-sensitivity (gamma).
    competitor_sensitivity: dict[str, float]
    #: Registered business scenarios A-J.
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    #: Free-form notes recorded alongside, e.g. confounder strengths in force.
    metadata: dict[str, Any] = field(default_factory=dict)

    def write(self, root: Path) -> None:
        """Persist to ``root/ground_truth/`` as JSON."""
        directory = root / GROUND_TRUTH_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)

        payloads: dict[str, Any] = {
            "elasticity.json": {
                "description": (
                    "True own-price elasticity per product. Step 8 must recover "
                    "these; naive OLS is expected to be attenuated toward zero."
                ),
                "generated_at": datetime.now(UTC).isoformat(),
                "values": self.own_elasticity,
            },
            "cross_elasticity.json": {
                "description": (
                    "d log(demand_target) / d log(price_source). Positive => "
                    "substitutes, negative => complements."
                ),
                "values": self.cross_elasticity,
            },
            "promotion_uplift.json": {
                "description": (
                    "Saturating response uplift = a * (1 - exp(-b * discount)), "
                    "additive in log space, per product and promotion type."
                ),
                "values": self.promo_response,
            },
            "competitor_sensitivity.json": {
                "description": (
                    "gamma: elasticity of our demand to competitor price. "
                    "Positive => competitor price up, our demand up."
                ),
                "values": self.competitor_sensitivity,
            },
            "scenario_config.json": {
                "description": (
                    "Injected business scenarios A-J with the exact products, "
                    "stores and windows affected, plus expected direction. Also "
                    "seeds the Step 21 golden evaluation set."
                ),
                "scenarios": self.scenarios,
            },
            "metadata.json": self.metadata,
        }

        for filename, payload in payloads.items():
            (directory / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

    @staticmethod
    def load(root: Path) -> GroundTruth:
        """Read ground truth back. For tests and validation only."""
        directory = root / GROUND_TRUTH_DIRNAME

        def read(name: str) -> dict[str, Any]:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"ground truth {name!r} not found at {path}; regenerate the dataset"
                )
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return data

        return GroundTruth(
            own_elasticity=read("elasticity.json")["values"],
            cross_elasticity=read("cross_elasticity.json")["values"],
            promo_response=read("promotion_uplift.json")["values"],
            competitor_sensitivity=read("competitor_sensitivity.json")["values"],
            scenarios=read("scenario_config.json")["scenarios"],
            metadata=read("metadata.json"),
        )


def draw_ground_truth(
    products: pd.DataFrame,
    relationships: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> GroundTruth:
    """Draw the latent parameters, before any sales exist."""
    rng = rngs.get(Stream.GROUND_TRUTH)

    # --- own-price elasticity, banded by category ---------------------------
    # Staples inelastic, discretionary elastic. Within a category, premium SKUs
    # (higher price relative to category median) are somewhat less elastic -
    # buyers of premium goods are less price-driven, which also gives Step 8 a
    # real source of within-category heterogeneity to model.
    own: dict[str, float] = {}
    for category_name, group in products.groupby("category", sort=True):
        band = config.categories[str(category_name)].elasticity_range
        median_price = float(group["base_price"].median())
        for row in group.itertuples(index=False):
            base = float(rng.uniform(*band))
            premium_ratio = as_float(row.base_price) / max(median_price, 1e-6)
            # Damp elasticity for pricier SKUs, at most ~25%.
            damping = 1.0 - 0.25 * float(np.tanh(np.log(max(premium_ratio, 1e-6))))
            elasticity = base * damping
            # Keep inside the configured band so category semantics still hold.
            own[str(row.product_id)] = round(
                float(np.clip(elasticity, band[0] * 1.15, band[1] * 0.85)), 4
            )

    # --- cross-price elasticity --------------------------------------------
    cross: dict[str, dict[str, float]] = {}
    effective = relationships[relationships["relationship_type"] != "unrelated"]
    for row in effective.itertuples(index=False):
        cross.setdefault(str(row.product_a), {})[str(row.product_b)] = as_float(
            row.cross_elasticity
        )

    # --- promotion response curves -----------------------------------------
    # Per product x promotion type. Effectiveness varies by product, which is
    # what allows Step 6 to distinguish a promotion that works from one that
    # merely coincided with demand, and Step 7 to allocate budget between them.
    promo: dict[str, dict[str, dict[str, float]]] = {}
    for product_id in products["product_id"]:
        per_type: dict[str, dict[str, float]] = {}
        # A product-level effectiveness multiplier, so some SKUs simply respond
        # better to promotion than others regardless of mechanic.
        product_factor = float(rng.uniform(0.65, 1.35))
        for type_name, type_config in config.promotions.types.items():
            a = float(rng.uniform(*type_config.a)) * product_factor
            per_type[type_name] = {
                "a": round(a, 4),
                "b": round(float(type_config.b), 4),
                "spend_per_unit": type_config.spend_per_unit,
            }
        promo[str(product_id)] = per_type

    # --- competitor sensitivity --------------------------------------------
    competitor: dict[str, float] = {
        str(pid): round(float(rng.uniform(*config.competitor.cross_sensitivity)), 4)
        for pid in products["product_id"]
    }

    metadata = {
        "dataset_version": config.dataset_version,
        "scenario_version": config.scenario_version,
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "generated_at": datetime.now(UTC).isoformat(),
        "confounders": {
            "price_endogeneity_strength": config.pricing.endogeneity_strength,
            "randomised_price_test_fraction": config.pricing.randomised_test_fraction,
            "cost_passthrough": config.pricing.cost_passthrough,
            "competitor_cost_correlation": config.competitor.cost_index_correlation,
            "promotion_targeting_strength": config.promotions.targeting_strength,
            "pull_forward_fraction": config.promotions.pull_forward_fraction,
        },
        "notes": (
            "Own-price elasticity is recoverable with store and time fixed "
            "effects, or by instrumenting price with the commodity cost index. "
            "Naive OLS without controls is expected to be attenuated toward "
            "zero because price responds to anticipated demand."
        ),
    }

    return GroundTruth(
        own_elasticity=own,
        cross_elasticity=cross,
        promo_response=promo,
        competitor_sensitivity=competitor,
        metadata=metadata,
    )
