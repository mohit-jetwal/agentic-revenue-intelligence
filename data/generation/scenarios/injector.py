"""Scenario injection.

Scenarios are applied by mutating the driver matrices *before* demand is
simulated - a price scenario raises the price path, a stockout scenario throttles
the supply cap - rather than by editing sales afterwards. That ordering is what
makes them genuine: the effect propagates through the same causal chain as
everything else, including onto substitutes and complements, so Scenario A
automatically produces Scenario F without either being hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from data.generation.config import GenerationConfig
from data.generation.generators.competitor_generator import CompetitorPaths
from data.generation.generators.pricing_generator import PricePaths
from data.generation.generators.promotion_generator import PromotionPaths
from data.generation.rng import RngFactory, Stream


@dataclass
class ScenarioRecord:
    """One injected scenario, as registered in the ground truth."""

    scenario_id: str
    label: str
    description: str
    expected_effect: str
    product_ids: list[str] = field(default_factory=list)
    related_product_ids: list[str] = field(default_factory=list)
    store_ids: list[str] = field(default_factory=list)
    region: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    magnitude: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "label": self.label,
            "description": self.description,
            "expected_effect": self.expected_effect,
            "product_ids": self.product_ids,
            "related_product_ids": self.related_product_ids,
            # Store lists can be large; the count is what validation needs.
            "store_count": len(self.store_ids),
            "store_ids": self.store_ids[:25],
            "region": self.region,
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "magnitude": self.magnitude,
            "detail": self.detail,
        }


class ScenarioInjector:
    """Applies scenarios A-J to the driver matrices and records what it did."""

    def __init__(
        self,
        listings: pd.DataFrame,
        products: pd.DataFrame,
        stores: pd.DataFrame,
        calendar: pd.DataFrame,
        relationships: pd.DataFrame,
        config: GenerationConfig,
        rngs: RngFactory,
    ) -> None:
        self.listings = listings
        self.products = products
        self.stores = stores
        self.calendar = calendar
        self.relationships = relationships
        self.config = config
        self.rng = rngs.get(Stream.SCENARIO)
        self.records: list[ScenarioRecord] = []

        self._dates = calendar["date"].to_numpy()
        self.n_days = len(calendar)
        self.n_pairs = len(listings)

        # Pair lookup by product and by store, so a scenario can select rows
        # without scanning the listing table repeatedly.
        self._pairs_by_product: dict[str, np.ndarray] = {
            str(pid): group.to_numpy()
            for pid, group in listings.groupby("product_id", sort=False).groups.items()
        }
        store_region = stores.set_index("store_id")["region"]
        self._pair_region = listings["store_id"].map(store_region).to_numpy()

    # -- helpers ------------------------------------------------------------

    def _window(self, duration_days: int, *, earliest: float = 0.25) -> tuple[int, int]:
        """Pick a date window, biased away from the very start of the history.

        Every scenario needs a pre-period to be measured against. Starting one on
        day 3 would leave nothing to compare with, and a validation check for it
        would fail for a reason that has nothing to do with the effect itself.
        """
        low = int(self.n_days * earliest)
        high = max(low + 1, self.n_days - duration_days - 30)
        start = int(self.rng.integers(low, high))
        return start, min(start + duration_days, self.n_days)

    def _products_with_relationships(self) -> set[str]:
        """Products that appear as a price driver in at least one relationship."""
        effective = self.relationships[self.relationships["relationship_type"] != "unrelated"]
        return {str(p) for p in effective["product_b"]}

    def _pick_products(
        self,
        count: int,
        *,
        exclude: set[str] | None = None,
        prefer: set[str] | None = None,
    ) -> list[str]:
        """Choose well-observed products, so the effect is measurable.

        Deliberately not uniform over the catalogue: a scenario applied to a SKU
        listed in two stores would be invisible under noise, and the resulting
        validation failure would be about sample size, not about the generator.

        ``prefer`` narrows the pool when it can still satisfy the requested
        count, and is ignored otherwise - a preference should shape the choice,
        not make the scenario impossible on a small profile.
        """
        excluded = exclude or set()
        counts: dict[str, int] = {
            str(pid): int(n) for pid, n in self.listings["product_id"].value_counts().items()
        }
        threshold = max(float(np.median(list(counts.values()))), 3.0)
        eligible = [p for p, n in counts.items() if n >= threshold and p not in excluded]
        if not eligible:
            eligible = [p for p in counts if p not in excluded]

        if prefer:
            preferred = [p for p in eligible if p in prefer]
            if len(preferred) >= count:
                eligible = preferred

        count = min(count, len(eligible))
        if count <= 0:
            return []
        chosen = self.rng.choice(np.array(eligible, dtype=object), size=count, replace=False)
        return [str(c) for c in np.atleast_1d(chosen)]

    def _pair_rows(self, product_id: str) -> np.ndarray:
        return self._pairs_by_product.get(product_id, np.array([], dtype=int))

    # -- scenarios ----------------------------------------------------------

    def inject_price_increase(self, price_paths: PricePaths) -> list[str]:
        """Scenario A: a sustained price rise on selected products.

        Because cross-price effects run through the same price matrix, this also
        produces Scenario F (substitutes gain) and Scenario G (complements lose)
        without either being separately fabricated.
        """
        spec = self.config.scenarios.price_increase
        # Prefer products that actually have declared substitutes or
        # complements. Scenarios F and G are meant to fall out of A via
        # cross-price effects, and a focal product with no relationships would
        # register the scenario while producing nothing to observe.
        chosen = self._pick_products(spec.products, prefer=self._products_with_relationships())
        for n, product_id in enumerate(chosen):
            start, end = self._window(spec.duration_days)
            rows = self._pair_rows(product_id)
            if rows.size == 0:
                continue
            price_paths.regular_price[np.ix_(rows, np.arange(start, end))] *= np.float32(
                1.0 + spec.magnitude
            )
            price_paths.change_flag[rows, start] = True
            price_paths.change_reason[rows, start] = 1

            related = self.relationships[
                (self.relationships["product_b"] == product_id)
                & (self.relationships["relationship_type"] != "unrelated")
            ]
            substitutes = related[related["relationship_type"] == "substitute"]["product_a"]
            complements = related[related["relationship_type"] == "complement"]["product_a"]

            self.records.append(
                ScenarioRecord(
                    scenario_id=f"A{n + 1}",
                    label="price_increase",
                    description=(
                        f"Product {product_id} regular price raised "
                        f"{spec.magnitude:.0%} for {end - start} days."
                    ),
                    expected_effect=(
                        "Own demand falls; revenue direction depends on elasticity. "
                        "Substitutes gain volume (Scenario F); complements lose "
                        "volume (Scenario G)."
                    ),
                    product_ids=[product_id],
                    related_product_ids=[
                        *(str(p) for p in substitutes.head(5)),
                        *(str(p) for p in complements.head(5)),
                    ],
                    start_date=self._dates[start],
                    end_date=self._dates[end - 1],
                    magnitude=spec.magnitude,
                    detail={
                        "substitutes": [str(p) for p in substitutes.head(5)],
                        "complements": [str(p) for p in complements.head(5)],
                    },
                )
            )
        return chosen

    def inject_promotions(self, promotion_paths: PromotionPaths, ground_truth_promo: dict) -> None:
        """Scenarios B and C: a clearly good promotion and a clearly bad one.

        The bad promotion is not bad because uplift is absent - it is bad because
        the discount is deep enough that incremental margin cannot cover it.
        Step 6 should still measure positive uplift; Step 7 should still decline
        to fund it. Those are different judgements and the data must separate them.
        """
        good = self.config.scenarios.successful_promo
        bad = self.config.scenarios.bad_promo

        for label, spec, scenario_letter, effect in (
            (
                "successful_promo",
                good,
                "B",
                "Clear incremental uplift with positive ROI: discount is shallow "
                "enough that incremental margin exceeds spend.",
            ),
            (
                "bad_promo",
                bad,
                "C",
                "Sales rise but gross margin falls sharply; ROI is poor. Uplift is "
                "positive while the promotion still destroys value.",
            ),
        ):
            chosen = self._pick_products(spec.products)
            for n, product_id in enumerate(chosen):
                start, end = self._window(spec.duration_days)
                rows = self._pair_rows(product_id)
                if rows.size == 0:
                    continue
                window = np.arange(start, end)
                grid = np.ix_(rows, window)

                promotion_paths.discount[grid] = np.float32(spec.discount)
                response = ground_truth_promo[product_id]["Price Discount"]
                lift = float(response["a"]) * (1.0 - np.exp(-float(response["b"]) * spec.discount))
                promotion_paths.lift[grid] = np.float32(lift)

                self.records.append(
                    ScenarioRecord(
                        scenario_id=f"{scenario_letter}{n + 1}",
                        label=label,
                        description=(
                            f"Product {product_id} promoted at {spec.discount:.0%} "
                            f"for {end - start} days across {rows.size} stores."
                        ),
                        expected_effect=effect,
                        product_ids=[product_id],
                        start_date=self._dates[start],
                        end_date=self._dates[end - 1],
                        magnitude=spec.discount,
                        detail={"expected_log_lift": round(lift, 4)},
                    )
                )

    def inject_stockouts(self, supply_cap: np.ndarray, exclude: set[str]) -> None:
        """Scenario D: supply failure with demand intact.

        The critical property: only ``supply_cap`` is touched. Latent demand is
        untouched, so ground truth records what shoppers wanted while the sales
        table records only what was available to sell. That gap is exactly what
        Step 4 must learn to detect.
        """
        spec = self.config.scenarios.stockout
        chosen = self._pick_products(spec.products, exclude=exclude)
        for n, product_id in enumerate(chosen):
            start, end = self._window(spec.duration_days)
            rows = self._pair_rows(product_id)
            if rows.size == 0:
                continue
            affected_count = max(1, int(len(rows) * spec.stores_fraction))
            affected = self.rng.choice(rows, size=affected_count, replace=False)
            supply_cap[np.ix_(affected, np.arange(start, end))] = np.float32(0.12)

            store_ids = self.listings["store_id"].to_numpy()[affected]
            self.records.append(
                ScenarioRecord(
                    scenario_id=f"D{n + 1}",
                    label="stockout",
                    description=(
                        f"Product {product_id} supply constrained to ~12% of normal "
                        f"in {affected_count} of {len(rows)} stores for {end - start} days."
                    ),
                    expected_effect=(
                        "Observed sales fall sharply while latent demand is "
                        "unchanged. Root-cause analysis must attribute this to "
                        "availability, not to a demand decline."
                    ),
                    product_ids=[product_id],
                    store_ids=[str(s) for s in store_ids],
                    start_date=self._dates[start],
                    end_date=self._dates[end - 1],
                    magnitude=-0.88,
                    detail={"affected_stores": affected_count, "total_stores": len(rows)},
                )
            )

    def inject_competitor_price_cut(self, competitor: CompetitorPaths) -> None:
        """Scenario E: a competitor undercuts us."""
        spec = self.config.scenarios.competitor_price_cut
        chosen = self._pick_products(spec.products)
        positions = {str(pid): i for i, pid in enumerate(self.products["product_id"])}

        for n, product_id in enumerate(chosen):
            row = positions.get(product_id)
            if row is None:
                continue
            start, end = self._window(spec.duration_days)
            competitor.mean_price[row, start:end] *= np.float32(1.0 + spec.magnitude)

            self.records.append(
                ScenarioRecord(
                    scenario_id=f"E{n + 1}",
                    label="competitor_price_cut",
                    description=(
                        f"Competitor price for {product_id} cut {abs(spec.magnitude):.0%} "
                        f"for {end - start} days."
                    ),
                    expected_effect=(
                        "Our demand falls while our own price is unchanged - a "
                        "decline that own-price elasticity alone cannot explain."
                    ),
                    product_ids=[product_id],
                    start_date=self._dates[start],
                    end_date=self._dates[end - 1],
                    magnitude=spec.magnitude,
                )
            )

    def inject_regional_shock(self, scenario_term: np.ndarray, supply_cap: np.ndarray) -> None:
        """Scenario H: regional decline driven by distribution loss.

        Implemented as stores in the region dropping the product entirely, not as
        shoppers wanting less. Both look like "North is down" in an aggregate
        report; only one is a demand problem, and telling them apart is the point.
        """
        spec = self.config.scenarios.regional_shock
        region = spec.region or "North"
        start, end = self._window(spec.duration_days)

        in_region = np.flatnonzero(self._pair_region == region)
        if in_region.size == 0:
            return

        # A share of listings in the region lose distribution outright.
        lost_count = max(1, int(in_region.size * abs(spec.magnitude)))
        lost = self.rng.choice(in_region, size=lost_count, replace=False)
        window = np.arange(start, end)
        supply_cap[np.ix_(lost, window)] = np.float32(0.0)

        # The remainder see a milder genuine demand softening, so the region is
        # not explained by distribution alone - the agent has to weigh both.
        remaining = np.setdiff1d(in_region, lost)
        if remaining.size:
            scenario_term[np.ix_(remaining, window)] += np.float32(-0.06)

        self.records.append(
            ScenarioRecord(
                scenario_id="H1",
                label="regional_shock",
                description=(
                    f"{region} region: {lost_count} of {in_region.size} product-store "
                    f"listings lost distribution for {end - start} days, with a mild "
                    f"demand softening across the remainder."
                ),
                expected_effect=(
                    "Regional sales decline is driven mainly by lost distribution, "
                    "not by a fall in per-store demand. Attributing it entirely to "
                    "demand would be incorrect."
                ),
                region=region,
                store_ids=[str(s) for s in self.listings["store_id"].to_numpy()[lost]],
                start_date=self._dates[start],
                end_date=self._dates[end - 1],
                magnitude=spec.magnitude,
                detail={
                    "listings_lost": lost_count,
                    "listings_total": int(in_region.size),
                    "residual_demand_effect": -0.06,
                },
            )
        )

    def register_seasonal_and_launch(self) -> None:
        """Scenarios I and J: recorded rather than injected.

        Both are already produced by the base simulation - festival multipliers
        create the seasonal peak, and the launch ramp creates the gradual build.
        Injecting them again would double-count. They are registered here so the
        scenario registry is complete and validation can still check them.
        """
        festivals = self.calendar[self.calendar["festival_flag"]]
        if not festivals.empty:
            peak = festivals.groupby("festival_name").size().idxmax()
            occurrence = festivals[festivals["festival_name"] == peak]
            self.records.append(
                ScenarioRecord(
                    scenario_id="I1",
                    label="seasonal_peak",
                    description=(f"Festival demand peaks, most prominently around {peak}."),
                    expected_effect=(
                        "Demand rises materially on festival and pre-festival days "
                        "relative to comparable non-festival days."
                    ),
                    start_date=occurrence["date"].min(),
                    end_date=occurrence["date"].max(),
                    detail={"festival": str(peak)},
                )
            )

        launched = self.products[self.products["product_status"] == "Launched"]
        if not launched.empty:
            sample = launched.head(5)
            self.records.append(
                ScenarioRecord(
                    scenario_id="J1",
                    label="product_launch",
                    description=(
                        f"{len(launched)} products launch mid-history and build "
                        f"distribution over {self.config.lifecycle.launch_ramp_days} days."
                    ),
                    expected_effect=(
                        "Demand ramps up gradually from launch rather than starting "
                        "at full rate; pre-launch sales are zero."
                    ),
                    product_ids=[str(p) for p in sample["product_id"]],
                    start_date=sample["launch_date"].min(),
                    detail={"launched_products": len(launched)},
                )
            )

    def registry(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]
