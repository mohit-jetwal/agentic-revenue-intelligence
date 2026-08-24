"""Product relationships: substitutes, complements, unrelated.

Brief section 4.2 is emphatic that labelling products as substitutes is not
enough - the generated sales must actually behave that way. This module produces
the labels *and* the cross-price coefficients that the demand equation consumes,
from one source, so the two cannot disagree.

Sign convention, since everything downstream depends on it:

    cross_elasticity[a][b] = d log(demand_a) / d log(price_b)

    positive => substitutes  (b gets dearer, a sells more)
    negative => complements  (b gets dearer, a sells less)

Substitutes are drawn within a category (shoppers swap between shampoos, not
between shampoo and milk). Complements are drawn both within a category and
across categories - chips and cola is the classic pairing, and it gives Step 9
something interesting to find that a naive within-category search would miss.

Relationships are asymmetric on purpose. A premium SKU losing volume to a value
SKU is not matched in size by the reverse flow, because the value SKU is bigger.
Forcing symmetry would be tidier and wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.generation.coerce import as_float
from data.generation.config import GenerationConfig
from data.generation.rng import RngFactory, Stream

# Complement pairings that are plausible in a real basket.
_CROSS_CATEGORY_COMPLEMENTS: list[tuple[str, str]] = [
    ("Snacks", "Beverages"),
    ("Packaged Food", "Dairy"),
    ("Beverages", "Snacks"),
    ("Personal Care", "Home Care"),
    ("Dairy", "Packaged Food"),
    ("Health & Wellness", "Dairy"),
]


def _strength_label(coefficient: float) -> str:
    magnitude = abs(coefficient)
    if magnitude >= 0.30:
        return "strong"
    if magnitude >= 0.15:
        return "moderate"
    if magnitude > 0.0:
        return "weak"
    return "none"


def generate_product_relationships(
    products: pd.DataFrame,
    config: GenerationConfig,
    rngs: RngFactory,
) -> pd.DataFrame:
    """Draw directed product relationships with their cross-price coefficients."""
    rng = rngs.get(Stream.RELATIONSHIP)
    settings = config.relationships
    strength = settings.strength

    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, kind: str, coefficient: float) -> None:
        if a == b or (a, b) in seen:
            return
        seen.add((a, b))
        records.append(
            {
                "product_a": a,
                "product_b": b,
                "relationship_type": kind,
                "relationship_strength": _strength_label(coefficient),
                "cross_elasticity": round(float(coefficient), 4),
            }
        )

    def sample_pairs(pool: np.ndarray, count: int) -> list[tuple[str, str]]:
        """Distinct unordered pairs from a pool, clamped to what is available."""
        if len(pool) < 2 or count <= 0:
            return []
        max_pairs = len(pool) * (len(pool) - 1) // 2
        count = min(count, max_pairs)
        chosen: set[tuple[str, str]] = set()
        # Rejection sampling: cheap here because count is far below max_pairs.
        attempts = 0
        while len(chosen) < count and attempts < count * 40:
            attempts += 1
            a, b = rng.choice(pool, size=2, replace=False)
            key = tuple(sorted((str(a), str(b))))
            chosen.add((key[0], key[1]))
        return list(chosen)

    # --- within-category substitutes and complements ---
    for category in products["category"].unique():
        pool = products.loc[products["category"] == category, "product_id"].to_numpy()

        for a, b in sample_pairs(pool, settings.strong_substitute_pairs_per_category):
            forward = float(rng.uniform(*strength.strong_substitute))
            # Asymmetric reverse flow, 55-95% of the forward effect.
            reverse = forward * float(rng.uniform(0.55, 0.95))
            add(a, b, "substitute", forward)
            add(b, a, "substitute", reverse)

        for a, b in sample_pairs(pool, settings.weak_substitute_pairs_per_category):
            forward = float(rng.uniform(*strength.weak_substitute))
            reverse = forward * float(rng.uniform(0.5, 1.0))
            add(a, b, "substitute", forward)
            add(b, a, "substitute", reverse)

        for a, b in sample_pairs(pool, settings.complement_pairs_per_category):
            forward = float(rng.uniform(*strength.strong_complement))
            reverse = forward * float(rng.uniform(0.55, 0.95))
            add(a, b, "complement", forward)
            add(b, a, "complement", reverse)

    # --- cross-category complements ---
    available = set(products["category"].unique())
    pairings = [p for p in _CROSS_CATEGORY_COMPLEMENTS if p[0] in available and p[1] in available]
    if pairings:
        for _ in range(settings.cross_category_complement_pairs):
            cat_a, cat_b = pairings[int(rng.integers(0, len(pairings)))]
            pool_a = products.loc[products["category"] == cat_a, "product_id"].to_numpy()
            pool_b = products.loc[products["category"] == cat_b, "product_id"].to_numpy()
            if len(pool_a) == 0 or len(pool_b) == 0:
                continue
            a = str(rng.choice(pool_a))
            b = str(rng.choice(pool_b))
            coefficient = float(rng.uniform(*strength.weak_complement))
            add(a, b, "complement", coefficient)
            add(b, a, "complement", coefficient * float(rng.uniform(0.5, 1.0)))

    if not records:
        return pd.DataFrame(
            columns=[
                "product_a",
                "product_b",
                "relationship_type",
                "relationship_strength",
                "cross_elasticity",
            ]
        )

    frame = pd.DataFrame(records)

    # A sample of explicitly unrelated pairs. Included so Step 9 can be scored on
    # false positives too - a cross-price model that finds relationships
    # everywhere is worse than useless, and without labelled negatives there is
    # no way to measure that.
    related = set(zip(frame["product_a"], frame["product_b"], strict=True))
    all_ids = products["product_id"].to_numpy()
    unrelated_target = min(len(frame), max(10, len(frame) // 2))
    unrelated: list[dict[str, object]] = []
    attempts = 0
    while len(unrelated) < unrelated_target and attempts < unrelated_target * 30:
        attempts += 1
        a, b = rng.choice(all_ids, size=2, replace=False)
        if a == b or (str(a), str(b)) in related:
            continue
        unrelated.append(
            {
                "product_a": str(a),
                "product_b": str(b),
                "relationship_type": "unrelated",
                "relationship_strength": "none",
                "cross_elasticity": 0.0,
            }
        )

    if unrelated:
        frame = pd.concat([frame, pd.DataFrame(unrelated)], ignore_index=True)

    return frame


def build_cross_matrix(
    relationships: pd.DataFrame,
    product_ids: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Nested mapping ``target -> {source_price_product: coefficient}``.

    Sparse by design. A dense N x N matrix would be 90,000 entries at dev scale,
    almost all zero, and would make the demand loop scan the full catalogue for
    every product. The nested dict keeps the inner loop proportional to the
    number of real relationships instead.
    """
    known = set(product_ids.tolist())
    matrix: dict[str, dict[str, float]] = {}
    effective = relationships[relationships["relationship_type"] != "unrelated"]

    for row in effective.itertuples(index=False):
        target = str(row.product_a)
        source = str(row.product_b)
        if target not in known or source not in known:
            continue
        matrix.setdefault(target, {})[source] = as_float(row.cross_elasticity)

    return matrix
