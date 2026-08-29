"""Synthetic panels with an exactly known treatment effect (brief section 32).

**Why this module is the most important one in the step.**

A forecasting model can be validated by holding out data: the truth is in the
test set, and a wrong prediction is visibly wrong. A causal estimate has no such
luxury. The counterfactual - what this promoted store-day *would* have sold - is
missing from every dataset that will ever exist. So a promo uplift model can
score perfectly on every predictive metric and still return an effect that is
wrong by a factor of three, and nothing in the data will say so.

The only way out is to generate data where the answer is known by construction.
Here the effect is applied by hand, so the true ATT is not estimated, inferred or
approximated - it is *recorded*. An estimator either recovers it or does not.

**What is deliberately NOT reproduced.** This is not a second copy of
``data/generation``. The platform generator has price elasticity, cross-price
substitution, pull-forward, inventory censoring and negative-binomial
over-dispersion, and validating against it is a separate exercise (see
``validate_against_ground_truth`` in :mod:`ml.promo_uplift.diagnostics`). This
module's job is narrower and more exacting: a DGP simple enough that the true
effect is a single number, so a failure to recover it is unambiguously the
estimator's fault.

**The scenario that matters most is ``confounded_null``.** Treatment assignment
depends on demand, and the true effect is exactly zero. A naive comparison finds
a large, confident, entirely spurious uplift. Any method that reports anything
other than zero here would, on real data, invent effects for promotions that did
nothing - which is the specific failure this whole capability exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ml.promo_uplift.config import PromoUpliftConfig, get_promo_uplift_config

#: Columns holding simulation truth. Never features, never inputs to an
#: estimator - the same rule ``data/generation/ground_truth.py`` enforces for the
#: platform dataset. :meth:`SyntheticPanel.observable` strips them, and the tests
#: assert that an estimator fitted on the observable frame cannot see them.
GROUND_TRUTH_COLUMNS: frozenset[str] = frozenset(
    {
        "true_lambda_untreated",
        "true_lambda_treated",
        "true_effect_units",
        "true_uplift_pct",
        "true_segment_uplift",
    }
)

#: Segment labels for the heterogeneous scenario, and their true uplifts. Chosen
#: far enough apart that a CATE model ranking them wrongly is a real failure
#: rather than sampling noise at this panel size.
_SEGMENT_UPLIFTS: dict[str, float] = {"A": 0.35, "B": 0.15, "C": -0.05}

_CATEGORIES = ("Beverages", "Snacks", "Dairy", "Household")
_REGIONS = ("North", "South", "East", "West")

#: Separate stream for promotional spend, so adding spend does not shift the
#: demand draws and change every previously measured result.
_SPEND_SEED = 20260829


@dataclass(frozen=True)
class SyntheticScenario:
    """One configuration of the data-generating process.

    ``mechanic_uplift`` is only *half* the treatment effect. A promotion here, as
    in the platform generator, moves demand two ways: the mechanic itself, and
    the price cut acting through own-price elasticity. The counterfactual is "no
    promotion at all", which means no discount either - so the true effect is the
    sum of both channels.

    That is why tests assert against :attr:`SyntheticPanel.true_att_pct`, which
    is computed from the generated data, rather than against this field. Naming
    the mechanic ``true_uplift`` and asserting on it would have quietly measured
    the smaller half.
    """

    name: str
    #: Multiplicative mechanic effect during treatment. Not the total.
    mechanic_uplift: float
    #: Discount depth range for events. ``(0, 0)`` is a mechanic-only promotion
    #: with no price cut, which is what makes a genuinely null scenario possible.
    discount_range: tuple[float, float]
    #: Whether assignment depends on demand. When True, a naive comparison is
    #: biased and the bias direction is known.
    confounded: bool
    #: Whether the effect varies by store segment.
    heterogeneous: bool
    description: str
    #: What a *correct* method must conclude. Read by the report and the tests so
    #: the expectation lives with the scenario rather than being restated.
    expectation: str


SCENARIOS: dict[str, SyntheticScenario] = {
    "positive": SyntheticScenario(
        name="positive",
        mechanic_uplift=0.15,
        discount_range=(0.08, 0.32),
        confounded=False,
        heterogeneous=False,
        description=(
            "Randomly assigned promotions with a genuine effect: a +15% mechanic "
            "on top of the demand the price cut buys."
        ),
        expectation=(
            "every method recovers the true ATT; the naive one too, since random "
            "assignment leaves nothing to confound it"
        ),
    ),
    "negative": SyntheticScenario(
        name="negative",
        mechanic_uplift=-0.28,
        discount_range=(0.08, 0.20),
        confounded=False,
        heterogeneous=False,
        description=(
            "A promotion that destroys volume - the mechanic is negative enough "
            "to outweigh its own price cut. Cheapened brand, wrong timing, "
            "shoppers who stocked up elsewhere. Rare, real, and the estimator "
            "must not flinch from it."
        ),
        expectation="the estimate stays negative; nothing clips it at zero",
    ),
    "null": SyntheticScenario(
        name="null",
        mechanic_uplift=0.0,
        # No price cut, so the promotion genuinely does nothing at all. A
        # discounted "null" promotion would still move volume through
        # elasticity, and its true effect would not be zero.
        discount_range=(0.0, 0.0),
        confounded=False,
        heterogeneous=False,
        description="Randomly assigned mechanic-only promotions with no effect at all.",
        expectation="the confidence interval covers zero",
    ),
    "confounded": SyntheticScenario(
        name="confounded",
        mechanic_uplift=0.15,
        discount_range=(0.08, 0.32),
        confounded=True,
        heterogeneous=False,
        description=(
            "A genuine effect, but promotions are targeted at high-demand series "
            "and seasonal peaks - as they are in the real generator."
        ),
        expectation="the naive estimate is biased UPWARD; adjustment recovers the true ATT",
    ),
    "confounded_null": SyntheticScenario(
        name="confounded_null",
        mechanic_uplift=0.0,
        discount_range=(0.0, 0.0),
        confounded=True,
        heterogeneous=False,
        description=(
            "The sharpest test in the suite. Promotions are targeted at exactly "
            "the days that would have sold well anyway, and they do nothing."
        ),
        expectation=(
            "the naive estimate finds a large spurious uplift; a correct method "
            "returns zero and its interval covers zero"
        ),
    ),
    "heterogeneous": SyntheticScenario(
        name="heterogeneous",
        mechanic_uplift=0.15,
        discount_range=(0.08, 0.32),
        confounded=True,
        heterogeneous=True,
        description=(
            "Mechanic varies by store segment: +35% in A, +15% in B, -5% in C. "
            "The average is positive while one segment is losing money."
        ),
        expectation="CATE ranks A > B > C; the aggregate ATT hides the negative segment",
    ),
}


@dataclass
class SyntheticPanel:
    """A generated panel together with the truth behind it."""

    frame: pd.DataFrame
    scenario: SyntheticScenario

    #: True average treatment effect on the treated, in units per treated day.
    #: The estimand every estimator here targets.
    true_att_units: float
    #: True ATT as a fraction of untreated demand on the treated days. This is
    #: the number that equals ``scenario.true_uplift`` by construction, which is
    #: what makes it a usable assertion.
    true_att_pct: float
    #: Total incremental units across all treated rows.
    true_incremental_units: float
    treated_rows: int
    control_rows: int

    def observable(self) -> pd.DataFrame:
        """The frame as an estimator may see it, with truth removed.

        Dropping rather than trusting callers to ignore the columns. A
        ground-truth column left in a feature frame is the one bug that makes
        every result look excellent, and it is invisible in the output.
        """
        return self.frame.drop(columns=list(GROUND_TRUTH_COLUMNS), errors="ignore")

    def truth(self) -> pd.DataFrame:
        """Just the keys and the ground-truth columns, for validation."""
        keys = ["date", "product_id", "store_id", "treatment"]
        columns = keys + [c for c in GROUND_TRUTH_COLUMNS if c in self.frame.columns]
        return self.frame[columns].copy()

    def summary(self) -> str:
        return (
            f"{self.scenario.name}: true ATT {self.true_att_pct:+.1%} "
            f"({self.true_att_units:+.3f} units/day), "
            f"{self.treated_rows:,} treated / {self.control_rows:,} control rows"
        )


def generate(
    scenario: str | SyntheticScenario,
    *,
    config: PromoUpliftConfig | None = None,
    n_series: int | None = None,
    n_days: int | None = None,
    seed: int | None = None,
    censoring_rate: float = 0.0,
    burn_in_days: int = 70,
) -> SyntheticPanel:
    """Generate a panel whose treatment effect is known exactly.

    The demand equation is log-additive, matching the platform generator's
    structure so the estimators face the same functional form they will meet in
    production:

    .. code-block:: text

        log lambda[i,t] = log base[i] + season[t] + dow[t] + trend[i]*t
                          + beta_price[i] * log(price[i,t] / ref[i])
                          + log(1 + u[i]) * treated[i,t]
                          + noise

        units[i,t] ~ Poisson(lambda[i,t])

    ``u[i]`` is the whole causal story. Because it enters multiplicatively, the
    true counterfactual mean for a treated row is available in closed form -
    ``lambda / (1 + u)`` - so the ATT is computed rather than approximated.

    ``censoring_rate`` optionally caps some outcomes to exercise the stockout
    path. It is applied *after* the counts are drawn, so the recorded truth
    still describes the demand that existed rather than what survived.
    """
    settings = config or get_promo_uplift_config()
    spec = SCENARIOS[scenario] if isinstance(scenario, str) else scenario

    n_series = n_series or settings.synthetic.n_series
    n_days = n_days or settings.synthetic.n_days
    rng = np.random.default_rng(seed if seed is not None else settings.synthetic.seed)

    start = date(2024, 1, 1)
    dates = pd.to_datetime([start + timedelta(days=int(d)) for d in range(n_days)])
    day_index = np.arange(n_days, dtype=float)

    # --- series attributes ---------------------------------------------------
    # Log-normal base demand: a few fast movers, a long slow tail. A uniform
    # draw would make every series equally important and hide the fact that
    # volume weighting matters.
    log_base = rng.normal(2.4, 0.8, size=n_series)
    base_demand = np.exp(log_base)
    trend = rng.normal(0.0, 0.15, size=n_series) / max(n_days, 1)
    beta_price = -rng.uniform(0.8, 2.2, size=n_series)
    reference_price = np.round(rng.uniform(30.0, 350.0, size=n_series), 2)

    product_ids = np.array([f"SP{i // 4 + 1:04d}" for i in range(n_series)])
    store_ids = np.array([f"SS{i % 25 + 1:04d}" for i in range(n_series)])
    categories = np.array([_CATEGORIES[i % len(_CATEGORIES)] for i in range(n_series)])
    regions = np.array([_REGIONS[(i // 3) % len(_REGIONS)] for i in range(n_series)])

    # Segments carry the heterogeneous effect. Assigned by store so the segment
    # is a genuine store attribute a CATE model can condition on, rather than a
    # per-row label that would leak the effect directly.
    segment_labels = np.array(list(_SEGMENT_UPLIFTS))
    segments = segment_labels[np.arange(n_series) % len(segment_labels)]

    if spec.heterogeneous:
        series_uplift = np.array([_SEGMENT_UPLIFTS[s] for s in segments], dtype=float)
    else:
        series_uplift = np.full(n_series, spec.mechanic_uplift, dtype=float)

    # --- calendar terms ------------------------------------------------------
    day_of_week = np.array([d.weekday() for d in dates])
    dow_term = np.log(np.array([0.92, 0.90, 0.95, 1.00, 1.15, 1.30, 1.05]))[day_of_week]
    # One annual cycle, phase-shifted per category so the seasonal peak is not
    # simultaneous everywhere. A shared peak would make season and calendar-date
    # indistinguishable, and the confounder would collapse into a date effect.
    phase = {name: i * 0.6 for i, name in enumerate(_CATEGORIES)}
    seasonal = np.vstack(
        [
            0.30 * np.sin(2 * np.pi * (day_index / 365.25) + phase[str(c)])
            for c in categories
        ]
    )

    calendar_term = seasonal + dow_term[np.newaxis, :] + np.outer(trend, day_index)

    # --- treatment assignment ------------------------------------------------
    treated, discount = _assign_treatment(
        rng=rng,
        n_series=n_series,
        n_days=n_days,
        seasonal=seasonal,
        base_demand=base_demand,
        confounded=spec.confounded,
        min_duration=settings.treatment.min_duration_days,
        discount_range=spec.discount_range,
        burn_in=min(burn_in_days, max(n_days // 3, 1)),
    )

    # --- prices --------------------------------------------------------------
    regular_price = np.repeat(reference_price[:, np.newaxis], n_days, axis=1)
    # A little independent price variation so the price coefficient is
    # identified from something other than the promotion itself. Without it,
    # price and treatment are collinear and the outcome model cannot separate
    # the discount from the mechanic.
    regular_price = np.round(regular_price * np.exp(rng.normal(0.0, 0.03, (n_series, n_days))), 2)
    selling_price = np.round(regular_price * (1.0 - discount), 2)

    reference = reference_price[:, np.newaxis]
    log_ratio_actual = np.log(selling_price / reference)
    # The counterfactual price is the *regular* price: had the promotion not run,
    # there would have been no discount. Building the counterfactual at the
    # discounted price would hold the price cut fixed across both arms and
    # silently redefine the estimand as "the mechanic alone", which is the
    # smaller half of what a promotion does.
    log_ratio_counterfactual = np.log(regular_price / reference)

    # --- demand --------------------------------------------------------------
    # One noise draw, shared by both potential outcomes. That is what makes them
    # potential outcomes for the *same* unit rather than two different worlds:
    # the only difference between the arms is the treatment.
    noise = rng.normal(0.0, 0.18, size=(n_series, n_days))
    structural = np.log(base_demand)[:, np.newaxis] + calendar_term + noise

    lambda_untreated = np.exp(
        np.clip(structural + beta_price[:, np.newaxis] * log_ratio_counterfactual, -10.0, 10.0)
    )
    lambda_treated = np.exp(
        np.clip(structural + beta_price[:, np.newaxis] * log_ratio_actual, -10.0, 10.0)
    ) * np.where(treated, 1.0 + series_uplift[:, np.newaxis], 1.0)

    # Untreated rows have no discount, so both arms coincide there by
    # construction. Enforced rather than assumed - a rounding difference in the
    # price path would otherwise show as a phantom effect on control rows.
    lambda_treated = np.where(treated, lambda_treated, lambda_untreated)

    units = rng.poisson(np.clip(lambda_treated, 0.0, 1e6)).astype(np.int64)

    # --- optional censoring --------------------------------------------------
    stockout = np.zeros((n_series, n_days), dtype=bool)
    if censoring_rate > 0.0:
        units, stockout = _apply_censoring(units, lambda_treated, rng, censoring_rate)

    # --- assemble ------------------------------------------------------------
    frame = _build_frame(
        dates=dates,
        product_ids=product_ids,
        store_ids=store_ids,
        categories=categories,
        regions=regions,
        segments=segments,
        units=units,
        treated=treated,
        discount=discount,
        regular_price=regular_price,
        selling_price=selling_price,
        stockout=stockout,
        lambda_untreated=lambda_untreated,
        lambda_treated=lambda_treated,
        series_uplift=series_uplift,
    )

    treated_mask = frame["treatment"].to_numpy()
    treated_baseline = frame.loc[treated_mask, "true_lambda_untreated"]
    incremental = frame.loc[treated_mask, "true_effect_units"]

    # The ATT is the mean effect over treated rows, and its percentage form is
    # the ratio of totals, not the mean of per-row ratios. Averaging ratios
    # would weight a slow-moving store-day as heavily as a hero SKU and would
    # not equal `true_uplift` even when the DGP is exactly right.
    true_att_units = float(incremental.mean()) if len(incremental) else 0.0
    true_att_pct = (
        float(incremental.sum() / treated_baseline.sum())
        if len(treated_baseline) and treated_baseline.sum() > 0
        else 0.0
    )

    return SyntheticPanel(
        frame=frame,
        scenario=spec,
        true_att_units=true_att_units,
        true_att_pct=true_att_pct,
        true_incremental_units=float(incremental.sum()),
        treated_rows=int(treated_mask.sum()),
        control_rows=int((~treated_mask).sum()),
    )


def _assign_treatment(
    *,
    rng: np.random.Generator,
    n_series: int,
    n_days: int,
    seasonal: np.ndarray,
    base_demand: np.ndarray,
    confounded: bool,
    min_duration: int,
    discount_range: tuple[float, float],
    burn_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place promotion blocks, optionally targeted at demand.

    Two channels of confounding when ``confounded``, mirroring how real
    merchandisers behave and how ``data/generation`` implements it:

    * **Cross-sectional** - high-volume series get more events, because they
      carry the category and attract trade investment.
    * **Temporal** - within a series, events land on seasonal peaks, because a
      promotion is planned for when shoppers are already in the aisle.

    Both make treated rows systematically higher-demand than control rows before
    any promotion runs, which is precisely the bias a naive comparison reports
    as uplift. Both are also *observable*, so adjustment can remove them - the
    honest position, and the same one the platform generator takes.
    """
    treated = np.zeros((n_series, n_days), dtype=bool)
    discount = np.zeros((n_series, n_days), dtype=float)

    volume_rank = (np.argsort(np.argsort(base_demand)) / max(n_series - 1, 1)) - 0.5

    # A fifth of listings are never promoted. Real assortments look like this -
    # plenty of SKUs never get trade investment - and two things here need it:
    # difference-in-differences has no control group without never-treated
    # units, and the cross-sectional control pool is empty without them, leaving
    # every comparison within-series.
    #
    # Which listings are skipped is NOT random when confounded: the lowest-volume
    # ones are, because that is who gets passed over in practice. That makes the
    # never-treated group systematically different from the treated one, which is
    # exactly the compositional problem cross-sectional controls have in reality.
    if confounded:
        never_treated = set(np.argsort(base_demand)[: int(0.20 * n_series)].tolist())
    else:
        never_treated = set(
            rng.choice(n_series, size=int(0.20 * n_series), replace=False).tolist()
        )

    for i in range(n_series):
        if i in never_treated:
            continue
        if confounded:
            # 2 to 10 events, rising with volume.
            expected = 4.0 + 6.0 * (volume_rank[i] + 0.5)
            weights = np.exp(1.6 * seasonal[i])
        else:
            expected = 6.0
            weights = np.ones(n_days)
        weights = weights / weights.sum()

        # No promotion inside the burn-in. Two reasons, both structural rather
        # than cosmetic: the trailing covariates need history before the first
        # event or every treated row is dropped for an incomplete adjustment
        # set, and difference-in-differences needs a genuinely untreated
        # pre-period to test parallel trends against. Without it DiD cannot run
        # at all and the comparison table loses a row it is meant to have.
        weights = weights.copy()
        weights[:burn_in] = 0.0
        weights = weights / weights.sum()

        n_events = int(rng.poisson(expected))
        n_events = int(np.clip(n_events, 1, 14))

        for _ in range(n_events):
            duration = int(rng.integers(min_duration, min_duration + 12))
            start = int(rng.choice(n_days, p=weights))
            end = min(start + duration, n_days)
            # Skip overlaps rather than merging them. Two promotions on one
            # listing on one day makes the treatment indicator ambiguous, and an
            # ambiguous treatment makes the estimand undefined.
            if treated[i, start:end].any():
                continue
            treated[i, start:end] = True
            discount[i, start:end] = float(rng.uniform(*discount_range))

    return treated, discount


def _apply_censoring(
    units: np.ndarray,
    lambda_treated: np.ndarray,
    rng: np.random.Generator,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap a share of outcomes, biased toward high-demand rows.

    Censoring is *not* applied at random. In the platform generator a stockout
    happens when demand outruns the reorder policy, so it strikes the busiest
    days - and because promotions raise demand, treated rows censor more often
    than control rows. Reproducing that correlation is the whole point: uniform
    random censoring would be ignorable and would make the stockout handling
    look unnecessary.
    """
    threshold = np.quantile(lambda_treated, 1.0 - rate * 2.5)
    at_risk = lambda_treated >= threshold
    stockout = at_risk & (rng.random(units.shape) < 0.5)
    capped = np.where(stockout, np.floor(units * rng.uniform(0.4, 0.8, units.shape)), units)
    return capped.astype(np.int64), stockout


def _build_frame(
    *,
    dates: pd.DatetimeIndex,
    product_ids: np.ndarray,
    store_ids: np.ndarray,
    categories: np.ndarray,
    regions: np.ndarray,
    segments: np.ndarray,
    units: np.ndarray,
    treated: np.ndarray,
    discount: np.ndarray,
    regular_price: np.ndarray,
    selling_price: np.ndarray,
    stockout: np.ndarray,
    lambda_untreated: np.ndarray,
    lambda_treated: np.ndarray,
    series_uplift: np.ndarray,
) -> pd.DataFrame:
    """Flatten the (series, day) matrices into the analysis frame.

    Column names match the platform's gold ``sales`` table so the same treatment
    construction, feature code and estimators run over both without a
    translation layer. A synthetic panel the real code cannot read would
    validate a different pipeline than the one that ships.
    """
    n_series, n_days = units.shape

    promotion_id = np.full(units.shape, None, dtype=object)
    if treated.any():
        # One id per contiguous block per series, so an event is addressable the
        # way a real promotion_id is.
        block_starts = treated & ~np.pad(treated[:, :-1], ((0, 0), (1, 0)))
        block_number = np.cumsum(block_starts.reshape(-1)).reshape(units.shape)
        labels = np.char.add("SPR", np.char.zfill(block_number[treated].astype(str), 6))
        promotion_id[treated] = labels

    unit_cost = np.round(regular_price * 0.62, 2)
    revenue = np.round(units * selling_price, 2)
    cost = np.round(units * unit_cost, 2)

    # Promotional spend, so ROI is exercised rather than reported as
    # unavailable. Deliberately *not* proportional to the uplift: spend is
    # decided when the promotion is planned, before anyone knows whether it
    # worked. Making it track the outcome would build a positive ROI into the
    # data by construction and make the profitability finding circular.
    spend = np.zeros(units.shape, dtype=float)
    if treated.any():
        spend_rng = np.random.default_rng(_SPEND_SEED)
        per_event = {
            # Spread wide enough that some events clear break-even and some do
            # not. A range where every promotion is profitable would never
            # exercise the value-destroying path, which is the finding Step 8
            # most needs to be able to act on.
            label: float(spend_rng.uniform(150.0, 2500.0))
            for label in np.unique(promotion_id[treated])
        }
        # Charged once, on the event's first day. Broadcasting the total across
        # every day of the window would multiply the spend by the duration when
        # the event table is summed.
        first_day = treated & ~np.pad(treated[:, :-1], ((0, 0), (1, 0)))
        rows, cols = np.nonzero(first_day)
        for r, c in zip(rows, cols, strict=True):
            spend[r, c] = per_event[promotion_id[r, c]]

    frame = pd.DataFrame(
        {
            "date": np.tile(dates.to_numpy(), n_series),
            "product_id": np.repeat(product_ids, n_days),
            "store_id": np.repeat(store_ids, n_days),
            "category": np.repeat(categories, n_days),
            "region": np.repeat(regions, n_days),
            "store_segment": np.repeat(segments, n_days),
            "units": units.reshape(-1),
            "regular_price": regular_price.reshape(-1),
            "selling_price": selling_price.reshape(-1),
            "discount_percentage": (discount.reshape(-1) * 100.0).round(2),
            "revenue": revenue.reshape(-1),
            "cost": cost.reshape(-1),
            "gross_profit": (revenue - cost).reshape(-1),
            "promotion_id": promotion_id.reshape(-1),
            "promotion_flag": treated.reshape(-1),
            "promotion_spend": spend.reshape(-1),
            "treatment": treated.reshape(-1),
            "stockout_flag": stockout.reshape(-1),
            # --- ground truth, stripped by `observable()` --------------------
            "true_lambda_untreated": lambda_untreated.reshape(-1).round(6),
            "true_lambda_treated": lambda_treated.reshape(-1).round(6),
            "true_effect_units": (lambda_treated - lambda_untreated).reshape(-1).round(6),
            "true_uplift_pct": np.where(
                treated, np.repeat(series_uplift[:, np.newaxis], n_days, axis=1), 0.0
            ).reshape(-1),
            "true_segment_uplift": np.repeat(
                np.array([_SEGMENT_UPLIFTS.get(str(s), 0.0) for s in segments])[:, np.newaxis],
                n_days,
                axis=1,
            ).reshape(-1),
        }
    )
    return frame.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)


def scenario_config(
    scenario: str | SyntheticScenario, base: PromoUpliftConfig | None = None
) -> PromoUpliftConfig:
    """The configuration a scenario needs to be analysable.

    Only one thing is adjusted, and it is a definitional collision rather than a
    convenience. The null scenarios use **mechanic-only** promotions with no
    price cut - that is what makes their true effect exactly zero, since a
    discounted promotion moves volume through elasticity whatever the mechanic
    does. But the default treatment definition requires a depth of at least 5%
    to screen out trivial price noise, which would filter every one of those
    events out and leave nothing to estimate.

    So the depth floor drops to zero for those scenarios. Nothing else moves:
    the estimators, the covariates, the overlap rules and the fold structure are
    the shipped ones, because a validation that ran on a softened configuration
    would validate a system nobody deploys.
    """
    settings = base or get_promo_uplift_config()
    spec = SCENARIOS[scenario] if isinstance(scenario, str) else scenario

    if spec.discount_range == (0.0, 0.0):
        return settings.model_copy(
            update={
                "treatment": settings.treatment.model_copy(
                    update={"min_discount_depth": 0.0}
                )
            }
        )
    return settings


def generate_all(
    *,
    config: PromoUpliftConfig | None = None,
    seed: int | None = None,
) -> dict[str, SyntheticPanel]:
    """Every scenario, for the validation report."""
    return {name: generate(name, config=config, seed=seed) for name in SCENARIOS}


__all__ = [
    "GROUND_TRUTH_COLUMNS",
    "SCENARIOS",
    "SyntheticPanel",
    "SyntheticScenario",
    "generate",
    "generate_all",
]
