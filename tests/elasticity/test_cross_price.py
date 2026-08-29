"""Cross-price elasticity: sign, multiple testing, and the promotion trap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.schemas.domain import RelationshipType
from ml.cross_price_elasticity.estimator import (
    PairEstimate,
    adjust_p_values,
    candidate_pairs,
    estimate_cross_elasticities,
    estimate_pair,
)
from ml.price_elasticity.estimator import prepare_panel
from tests.elasticity.conftest import TRUE_CROSS, make_panel

pytestmark = pytest.mark.models


def make_pair_panel(
    *, cross: float = TRUE_CROSS, n_stores: int = 14, n_days: int = 400, seed: int = 21
) -> pd.DataFrame:
    """Two products where B's price moves A's demand by a known amount."""
    rng = np.random.default_rng(seed)
    source = make_panel(
        n_stores=n_stores, n_days=n_days, seed=seed, product_id="B",
        seasonal_amplitude=0.0,
    )

    focal_rows = []
    for store, group in source.groupby("store_id", observed=True):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        own_log_price = np.log(rng.uniform(9.0, 15.0, n))
        source_log_price = np.log(group["selling_price"].to_numpy())

        log_demand = (
            np.log(rng.uniform(4_000, 9_000))
            - 1.5 * own_log_price
            + cross * source_log_price
            + rng.normal(0, 0.2, n)
        )
        focal_rows.append(
            pd.DataFrame(
                {
                    "date": group["date"],
                    "product_id": "A",
                    "store_id": store,
                    "category": "Test",
                    "units": rng.poisson(np.exp(np.clip(log_demand, -5, 9))),
                    "selling_price": np.exp(own_log_price).round(2),
                    "regular_price": np.exp(own_log_price).round(2),
                    "promotion_flag": False,
                    "stockout_flag": False,
                    "price_change_reason": "scheduled",
                }
            )
        )

    return pd.concat([pd.concat(focal_rows, ignore_index=True), source], ignore_index=True)


def panels_from(frame: pd.DataFrame, *, drop_promotions: bool = True) -> dict:
    prepared = prepare_panel(frame, drop_promotions=drop_promotions)
    return {str(p): g for p, g in prepared.groupby("product_id", observed=True)}


class TestRecovery:
    def test_recovers_a_known_substitute(self) -> None:
        panels = panels_from(make_pair_panel(cross=TRUE_CROSS))
        estimate = estimate_pair(
            panels["A"], panels["B"], focal_product="A", source_product="B"
        )
        assert estimate.cross_elasticity == pytest.approx(TRUE_CROSS, abs=0.15)

    def test_positive_means_substitute(self) -> None:
        """Sign convention is the whole point: getting it backwards inverts
        every assortment and cannibalisation conclusion."""
        panels = panels_from(make_pair_panel(cross=0.5))
        estimate = estimate_pair(
            panels["A"], panels["B"], focal_product="A", source_product="B"
        )
        estimate.adjusted_p_value = 0.001
        assert estimate.relationship is RelationshipType.SUBSTITUTE

    def test_negative_means_complement(self) -> None:
        panels = panels_from(make_pair_panel(cross=-0.5, seed=33))
        estimate = estimate_pair(
            panels["A"], panels["B"], focal_product="A", source_product="B"
        )
        assert estimate.cross_elasticity < 0
        estimate.adjusted_p_value = 0.001
        assert estimate.relationship is RelationshipType.COMPLEMENT

    def test_no_relationship_is_reported_as_unrelated(self) -> None:
        panels = panels_from(make_pair_panel(cross=0.0, seed=44))
        estimate = estimate_pair(
            panels["A"], panels["B"], focal_product="A", source_product="B"
        )
        assert abs(estimate.cross_elasticity) < 0.15
        assert estimate.relationship is RelationshipType.UNRELATED

    def test_own_price_is_controlled_for(self) -> None:
        """Category cost shocks move both prices together. Omitting the own
        price lets that shared movement load onto the cross coefficient."""
        panels = panels_from(make_pair_panel())
        estimate = estimate_pair(
            panels["A"], panels["B"], focal_product="A", source_product="B"
        )
        assert estimate.own_price_coefficient is not None
        assert estimate.own_price_coefficient < 0


class TestPromotionTrap:
    def test_keeping_source_promotions_preserves_the_signal(self) -> None:
        """The bug this fixture exists to prevent.

        A candidate's promotional price cut is the largest single source of the
        price variation that identifies the cross effect. Dropping it deletes
        the experiment: measured on real data it cut the identifying variation
        by 29% and lost a true +0.44 substitute entirely.
        """
        frame = make_pair_panel()
        source_mask = (frame["product_id"] == "B") & (
            frame["date"] < frame["date"].min() + pd.Timedelta(days=150)
        )
        frame.loc[source_mask, "promotion_flag"] = True
        frame.loc[source_mask, "selling_price"] = (
            frame.loc[source_mask, "selling_price"] * 0.7
        ).round(2)

        kept = panels_from(frame, drop_promotions=False)
        dropped = panels_from(frame, drop_promotions=True)

        assert kept["B"]["log_price"].std() > dropped["B"]["log_price"].std()
        assert len(kept["B"]) > len(dropped["B"])


class TestMultipleTesting:
    def test_adjusted_p_values_are_never_smaller_than_raw(self) -> None:
        pairs = [
            PairEstimate("B", "A", 0.4, 0.1, p) for p in (0.001, 0.01, 0.04, 0.2, 0.6)
        ]
        adjust_p_values(pairs)
        for pair in pairs:
            assert pair.adjusted_p_value is not None
            assert pair.adjusted_p_value >= pair.p_value

    def test_adjustment_is_monotone(self) -> None:
        """A larger raw p-value must never adjust to a smaller one."""
        pairs = [
            PairEstimate("B", "A", 0.4, 0.1, p) for p in (0.001, 0.01, 0.04, 0.2, 0.6)
        ]
        adjust_p_values(pairs)
        adjusted = [p.adjusted_p_value for p in sorted(pairs, key=lambda x: x.p_value)]
        assert adjusted == sorted(adjusted)  # type: ignore[type-var]

    def test_significance_uses_the_adjusted_value(self) -> None:
        """Testing 40 candidates at the 5% level produces two spurious findings
        by construction. Raw p-values would report them as real."""
        pairs = [PairEstimate("B", "A", 0.4, 0.1, 0.04)] + [
            PairEstimate(f"C{i}", "A", 0.01, 0.1, 0.9) for i in range(39)
        ]
        adjust_p_values(pairs)
        assert pairs[0].p_value == 0.04
        assert not pairs[0].is_significant

    def test_empty_input_is_handled(self) -> None:
        assert adjust_p_values([]) == []


class TestCandidateSelection:
    def _products(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "product_id": [f"P{i}" for i in range(6)],
                "category": ["Snacks"] * 3 + ["Dairy"] * 3,
            }
        )

    def test_restricts_to_the_same_category(self) -> None:
        """A cross-price effect between shampoo and frozen peas is not a
        finding, it is a coincidence that survived a t-test."""
        candidates = candidate_pairs("P0", self._products())
        assert set(candidates) == {"P1", "P2"}

    def test_includes_declared_relationships(self) -> None:
        relationships = pd.DataFrame(
            {
                "product_a": ["P0"],
                "product_b": ["P5"],
                "relationship_type": ["substitute"],
            }
        )
        candidates = candidate_pairs("P0", self._products(), relationships)
        assert "P5" in candidates

    def test_excludes_the_focal_product(self) -> None:
        assert "P0" not in candidate_pairs("P0", self._products())

    def test_respects_the_cap(self) -> None:
        products = pd.DataFrame(
            {"product_id": [f"P{i}" for i in range(100)], "category": ["Snacks"] * 100}
        )
        assert len(candidate_pairs("P0", products, max_candidates=10)) == 10


class TestEstimationLoop:
    def test_reports_pairs_tested_not_just_survivors(self) -> None:
        """A result that reports only the survivors has hidden its denominator."""
        panels = panels_from(make_pair_panel())
        pairs, tested = estimate_cross_elasticities(panels, "A", ["B", "MISSING"])
        assert tested == 1
        assert len(pairs) == 1

    def test_unknown_focal_product_raises(self) -> None:
        panels = panels_from(make_pair_panel())
        with pytest.raises(ValueError, match="no panel for focal"):
            estimate_cross_elasticities(panels, "NOPE", ["B"])

    def test_too_few_overlapping_days_is_skipped(self) -> None:
        panels = panels_from(make_pair_panel(n_days=400))
        panels["B"] = panels["B"].head(20)
        pairs, tested = estimate_cross_elasticities(panels, "A", ["B"])
        assert tested == 1
        assert pairs == []


class TestStrength:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, "strong"), (0.2, "moderate"), (0.02, "none")],
    )
    def test_strength_bands(self, value: float, expected: str) -> None:
        estimate = PairEstimate("B", "A", value, 0.01, 0.0001)
        estimate.adjusted_p_value = 0.0001
        assert estimate.strength == expected
