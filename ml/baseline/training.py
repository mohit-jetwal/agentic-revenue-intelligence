"""Training pipeline: splits, promotion handling, backtesting (sections 5-6, 11-12).

Three decisions live here, and each one is the difference between a defensible
baseline and a plausible-looking wrong one.

**Splits are temporal, never random.** A random split lets the model see 2025
while predicting 2024, which inflates every metric and produces a model that
looks excellent and forecasts badly. Section 11 is explicit.

**Promotional rows are handled two ways and compared.** Neither is obviously
right, so both are built and scored - see :class:`PromotionApproach`.

**Stockout rows are excluded and not lagged forward.** Observed sales during a
stockout measure availability, not demand. Training on them teaches the model
that a supply failure predicts low demand, which is exactly backwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger
from ml.baseline.conformal import ConformalCalibration, calibrate, measure_coverage
from ml.baseline.evaluation import BaselineMetrics, compute_metrics
from ml.baseline.models import IDENTIFIER_COLUMNS, BaselineEstimator

logger = get_logger(__name__)

TARGET = "units"

#: Never features. Identifiers, the target, and columns that are functions of
#: the target on the same row. Step 3's engineer already drops the
#: target-derived ones; this is the second line of defence, because a leak here
#: produces a model with excellent metrics and no value.
EXCLUDED_FROM_FEATURES: frozenset[str] = frozenset(
    {
        *IDENTIFIER_COLUMNS,
        TARGET,
        "revenue",
        "cost",
        "gross_profit",
        "sold_units",
        "closing_inventory",
        "inventory_days",
        "promotion_units",
        "promotion_id",
        "units_uncensored",
        # Free text and high-cardinality labels: no signal a tree can use, and
        # they bloat any encoder fitted downstream.
        "product_name",
        "store_name",
        "city",
        "state",
        "promotion_channel",
        "price_change_reason",
    }
)

#: Supply-side columns. **Never** features, under either approach.
#:
#: This exclusion is not about leakage in the usual sense - these are all known
#: in advance and none is derived from today's target. It is about *what the
#: baseline is defined to be*: demand under normal conditions, with stock
#: available. Conditioning on inventory answers a different question - "what
#: would sell given this stock level" - and that question is circular for every
#: use this model has.
#:
#: The practical consequence was measured rather than assumed. With these
#: columns present, ``closing_inventory_lag_1`` became the single most important
#: feature in the LightGBM baseline, and the model recovered only 0.30 of true
#: demand during stockouts against a theoretically-correct ~0.64 - it had
#: learned "low stock predicts low sales" and therefore reported a supply
#: failure as a demand collapse. Both LightGBM candidates were disqualified by
#: the stockout check because of it.
#:
#: Note that excluding stockout *rows* from training does not prevent this. The
#: model learns the relationship from the many partially-depleted rows just
#: below the stockout threshold, then extrapolates it to zero stock.
#:
#: Accepted cost: a recent stockout can genuinely depress future demand, as
#: customers switch brand or store. That is a real effect this model now cannot
#: see. It is given up deliberately, because the effect is inseparable here from
#: the censoring artefact and far smaller than it.
SUPPLY_FEATURES: frozenset[str] = frozenset(
    {
        "inventory_available",
        "opening_inventory",
        "closing_inventory_lag_1",
        "inventory_days_lag_1",
        "inventory_days_cover",
        "inventory_ratio",
        "stockout_flag",
        "stockout_yesterday",
        "days_since_stockout",
        "stockouts_last_28d",
        "stockouts_last_90d",
    }
)

#: Promotion columns. Present as features only under Approach B.
PROMOTION_FEATURES: frozenset[str] = frozenset(
    {
        "promotion_flag",
        "promotion_discount",
        "promotion_duration",
        "days_into_promotion",
        "days_until_promotion_end",
        "promotion_type",
        "display_flag",
        "bundle_flag",
        "promotion_spend",
        "promotion_intensity",
        "days_to_next_promotion",
    }
)


class PromotionApproach(StrEnum):
    """How promotional contamination is handled (brief section 5).

    ``EXCLUDE`` (Approach C)
        Train only on non-promotional rows; predict everywhere. Counterfactual
        semantics are unambiguous - the model has literally never seen a
        promotion, so its prediction *is* the no-promotion expectation.

        The cost is selection bias. Step 2 sets
        ``promotions.targeting_strength: 0.40``, weighting promotion timing
        toward seasonal peaks, so dropping promotional rows under-represents
        high season. The baseline then underestimates peaks and **overstates
        uplift** - a bias that flows straight into Step 6.

    ``CONTROL`` (Approach B)
        Train on all rows with promotion features included, then predict with
        those features zeroed. Uses every row, so no selection bias.

        The cost is extrapolation. Asking a gradient-boosted tree to predict
        ``promotion_flag=0`` for a product-store that was almost always
        promoted is asking it to extrapolate to a region of feature space it
        never saw, which is where tree models are weakest.

    Neither dominates on argument, so both are fitted and scored against
    ``latent_units``. The comparison is a deliverable in its own right.
    """

    EXCLUDE = "exclude"
    CONTROL = "control"


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train / calibration / validation / test boundaries.

    Four folds rather than three, because conformal calibration needs data the
    model did not train on *and* that is not the test set. Reusing test data for
    calibration would make the measured coverage meaningless - it would be
    reporting the calibration set's own quantile back to itself.
    """

    train_start: date
    train_end: date
    calibration_start: date
    calibration_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date

    def describe(self) -> str:
        return (
            f"train {self.train_start}..{self.train_end} | "
            f"calib {self.calibration_start}..{self.calibration_end} | "
            f"valid {self.valid_start}..{self.valid_end} | "
            f"test {self.test_start}..{self.test_end}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "train_start": str(self.train_start), "train_end": str(self.train_end),
            "calibration_start": str(self.calibration_start),
            "calibration_end": str(self.calibration_end),
            "valid_start": str(self.valid_start), "valid_end": str(self.valid_end),
            "test_start": str(self.test_start), "test_end": str(self.test_end),
        }


def build_temporal_split(
    panel: pd.DataFrame,
    *,
    test_days: int = 120,
    valid_days: int = 90,
    calibration_days: int = 60,
    date_column: str = "date",
) -> TemporalSplit:
    """Carve chronological folds from the back of the available history.

    Ordered train -> calibration -> validation -> test, oldest to newest, so
    every evaluation is forward in time from the data that produced the model.

    Calibration sits *before* validation deliberately: validation drives early
    stopping, so it must be the fold closest to test for the stopping point to
    reflect the most recent regime.
    """
    dates = pd.to_datetime(panel[date_column]).dt.date
    first, last = dates.min(), dates.max()

    required = test_days + valid_days + calibration_days + 120
    available = (last - first).days
    if available < required:
        raise ValueError(
            f"only {available} days of history but the split needs {required} "
            f"(test {test_days} + valid {valid_days} + calibration "
            f"{calibration_days} + at least 120 for training)"
        )

    test_end = last
    test_start = test_end - timedelta(days=test_days - 1)
    valid_end = test_start - timedelta(days=1)
    valid_start = valid_end - timedelta(days=valid_days - 1)
    calibration_end = valid_start - timedelta(days=1)
    calibration_start = calibration_end - timedelta(days=calibration_days - 1)
    train_end = calibration_start - timedelta(days=1)

    split = TemporalSplit(
        train_start=first, train_end=train_end,
        calibration_start=calibration_start, calibration_end=calibration_end,
        valid_start=valid_start, valid_end=valid_end,
        test_start=test_start, test_end=test_end,
    )
    logger.info("baseline.split_built", **split.to_dict())
    return split


def select_features(
    panel: pd.DataFrame, *, approach: PromotionApproach
) -> list[str]:
    """Feature columns for the given approach.

    Under ``EXCLUDE`` the promotion columns are removed entirely: the model
    never sees a promotional row, so a promotion feature would be constant and
    carry no information while inviting the misreading that promotions were
    modelled.
    """
    columns = [
        c for c in panel.columns
        if c not in EXCLUDED_FROM_FEATURES and c not in SUPPLY_FEATURES
    ]
    if approach is PromotionApproach.EXCLUDE:
        columns = [c for c in columns if c not in PROMOTION_FEATURES]
    return columns


def prepare_training_rows(
    panel: pd.DataFrame, *, approach: PromotionApproach
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter the panel down to rows whose target is trustworthy.

    Two exclusions, for different reasons:

    * **Stockout rows always.** The target is censored - it records what was
      available to sell, not what customers wanted. Training on it is training
      on the wrong quantity.
    * **Promotional rows under ``EXCLUDE`` only.** The target contains
      promotional lift, which is not baseline demand.
    """
    excluded: dict[str, int] = {"total": len(panel)}
    working = panel

    if "stockout_flag" in working.columns:
        censored = working["stockout_flag"].astype(bool)
        excluded["stockout"] = int(censored.sum())
        working = working[~censored]

    if approach is PromotionApproach.EXCLUDE and "promotion_flag" in working.columns:
        promoted = working["promotion_flag"].astype(bool)
        excluded["promotional"] = int(promoted.sum())
        working = working[~promoted]

    # A row with no target cannot train anything.
    if TARGET in working.columns:
        missing = working[TARGET].isna()
        excluded["missing_target"] = int(missing.sum())
        working = working[~missing]

    excluded["retained"] = len(working)
    return working.reset_index(drop=True), excluded


def neutralise_promotions(X: pd.DataFrame) -> pd.DataFrame:
    """Zero the promotion features, for Approach B prediction.

    This is the counterfactual step: "what would this row have sold with no
    promotion running?" The model was trained with promotion signal present, so
    asking for the baseline means presenting a row that says no promotion.

    Flags and depths go to zero; the "days until" style columns go to ``-1``,
    matching the sentinel Step 3's promotion primitives use for "no active
    promotion". Using 0 there would read as "the promotion ends today", which is
    a different and quite wrong statement.
    """
    neutral = X.copy()

    for column in ("promotion_flag", "display_flag", "bundle_flag"):
        if column in neutral.columns:
            neutral[column] = False if neutral[column].dtype == bool else 0

    for column in ("promotion_discount", "promotion_duration", "promotion_spend",
                   "promotion_intensity"):
        if column in neutral.columns:
            neutral[column] = 0.0

    for column in ("days_into_promotion", "days_until_promotion_end"):
        if column in neutral.columns:
            neutral[column] = -1

    if "promotion_type" in neutral.columns:
        promotion_type = neutral["promotion_type"]
        # Set to *missing*, not to a "none" label. On an unpromoted row the
        # training data had NaN here - the promotion calendar simply did not
        # join - so missing is the value the model actually learned to associate
        # with no promotion. Introducing a new "none" level would be an unseen
        # category at prediction time.
        #
        # The existing category set is preserved so LightGBM's native codes stay
        # aligned with training; rebuilding the dtype from scratch would
        # renumber them and silently map every row to the wrong level.
        if isinstance(promotion_type.dtype, pd.CategoricalDtype):
            missing: list[Any] = [None] * len(neutral)
            neutral["promotion_type"] = pd.Categorical(
                missing, categories=promotion_type.cat.categories
            )
        else:
            neutral["promotion_type"] = None

    return neutral


@dataclass
class TrainedBaseline:
    """A fitted estimator with everything needed to use and audit it."""

    estimator: BaselineEstimator
    approach: PromotionApproach
    split: TemporalSplit
    feature_names: tuple[str, ...]
    calibration: ConformalCalibration | None
    metrics: dict[str, BaselineMetrics] = field(default_factory=dict)
    coverage: Any = None
    excluded_rows: dict[str, int] = field(default_factory=dict)
    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.estimator.name}__{self.approach.value}"

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """Baseline prediction, applying the approach's counterfactual.

        The one place the two approaches differ at inference: ``CONTROL`` must
        neutralise the promotion features first, ``EXCLUDE`` never saw them.
        Putting it here rather than at each call site means a caller cannot get
        it wrong.
        """
        frame = neutralise_promotions(X) if self.approach is PromotionApproach.CONTROL else X
        return self.estimator.predict(frame[list(self.feature_names)])

    def summary(self) -> str:
        lines = [f"{self.name}: {self.split.describe()}"]
        for label, metric in self.metrics.items():
            lines.append(f"  {label:<22} {metric.summary()}")
        if self.coverage is not None:
            lines.append(f"  interval               {self.coverage.summary()}")
        return "\n".join(lines)


def train_baseline(
    panel: pd.DataFrame,
    estimator: BaselineEstimator,
    *,
    approach: PromotionApproach,
    split: TemporalSplit,
    alpha: float = 0.1,
    date_column: str = "date",
) -> TrainedBaseline:
    """Fit one estimator under one promotion approach, then calibrate and score.

    The full lifecycle for a single candidate: filter, fit with early stopping
    against a temporally-later fold, calibrate intervals on unseen data, measure
    coverage on data used for neither, and evaluate.
    """
    dates = pd.to_datetime(panel[date_column]).dt.date

    def window(start: date, end: date) -> pd.DataFrame:
        return panel[(dates >= start) & (dates <= end)]

    features = select_features(panel, approach=approach)

    train_rows, excluded = prepare_training_rows(
        window(split.train_start, split.train_end), approach=approach
    )
    valid_rows, _ = prepare_training_rows(
        window(split.valid_start, split.valid_end), approach=approach
    )

    if train_rows.empty:
        raise ValueError(f"{estimator.name}/{approach.value}: no training rows survived filtering")

    logger.info(
        "baseline.training_started",
        model=estimator.name, approach=approach.value,
        train_rows=len(train_rows), features=len(features), **excluded,
    )

    started = time.perf_counter()
    estimator.fit(
        train_rows[features],
        train_rows[TARGET],
        X_valid=valid_rows[features] if not valid_rows.empty else None,
        y_valid=valid_rows[TARGET] if not valid_rows.empty else None,
    )
    train_seconds = time.perf_counter() - started

    trained = TrainedBaseline(
        estimator=estimator,
        approach=approach,
        split=split,
        feature_names=tuple(features),
        calibration=None,
        excluded_rows=excluded,
        train_seconds=train_seconds,
    )

    # --- calibrate on clean, unseen rows ---------------------------------
    # Clean rows only: the interval should describe the model's error on normal
    # demand. Calibrating on promotional rows would widen it by the uplift and
    # then nothing would ever look significant.
    calibration_rows, _ = prepare_training_rows(
        window(split.calibration_start, split.calibration_end),
        approach=PromotionApproach.EXCLUDE,
    )
    if len(calibration_rows) >= 100:
        trained.calibration = calibrate(
            calibration_rows[TARGET],
            trained.predict_baseline(calibration_rows),
            alpha=alpha,
        )
    else:
        logger.warning(
            "baseline.calibration_skipped",
            model=estimator.name, rows=len(calibration_rows),
            reason="too few clean calibration rows for a meaningful quantile",
        )

    # --- evaluate on test -------------------------------------------------
    test_rows = window(split.test_start, split.test_end)
    started = time.perf_counter()
    test_predictions = trained.predict_baseline(test_rows)
    trained.predict_seconds = time.perf_counter() - started

    clean_test, _ = prepare_training_rows(test_rows, approach=PromotionApproach.EXCLUDE)
    if not clean_test.empty:
        trained.metrics["test_clean"] = compute_metrics(
            clean_test[TARGET], trained.predict_baseline(clean_test)
        )
        if trained.calibration is not None:
            trained.coverage = measure_coverage(
                clean_test[TARGET],
                trained.predict_baseline(clean_test),
                trained.calibration,
            )

    # Reported for completeness, but on promotional and stockout rows the
    # baseline is *supposed* to differ from the observed target - so these are
    # diagnostic, not accuracy.
    if "promotion_flag" in test_rows.columns:
        promoted = test_rows[test_rows["promotion_flag"].astype(bool)]
        if len(promoted) > 30:
            trained.metrics["test_promotional"] = compute_metrics(
                promoted[TARGET], trained.predict_baseline(promoted)
            )

    if "stockout_flag" in test_rows.columns:
        censored = test_rows[test_rows["stockout_flag"].astype(bool)]
        if len(censored) > 30:
            trained.metrics["test_stockout"] = compute_metrics(
                censored[TARGET], trained.predict_baseline(censored)
            )

    del test_predictions
    logger.info(
        "baseline.training_completed",
        model=estimator.name, approach=approach.value,
        train_seconds=round(train_seconds, 2),
        wmape=round(trained.metrics["test_clean"].wmape, 4)
        if "test_clean" in trained.metrics else None,
    )
    return trained


# ---------------------------------------------------------------------------
# Expanding-window backtest (brief section 12)
# ---------------------------------------------------------------------------


@dataclass
class BacktestFold:
    """One expanding-window fold."""

    index: int
    train_end: date
    valid_start: date
    valid_end: date
    metrics: BaselineMetrics
    train_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train_end": str(self.train_end),
            "valid_start": str(self.valid_start),
            "valid_end": str(self.valid_end),
            "train_rows": self.train_rows,
            **self.metrics.to_dict(),
        }


@dataclass
class BacktestResult:
    """Stability of accuracy across time."""

    folds: list[BacktestFold]

    @property
    def mean_wmape(self) -> float:
        return float(np.mean([f.metrics.wmape for f in self.folds])) if self.folds else float("nan")

    @property
    def std_wmape(self) -> float:
        return float(np.std([f.metrics.wmape for f in self.folds])) if self.folds else float("nan")

    @property
    def is_stable(self) -> bool:
        """Accuracy consistent across folds.

        Dispersion under a quarter of the mean. A model whose WMAPE swings from
        8% to 25% between quarters is not a model you can put behind a
        recommendation, even if its average looks respectable - the average is
        then describing something that never happens.
        """
        if not self.folds or not np.isfinite(self.mean_wmape) or self.mean_wmape <= 0:
            return False
        return self.std_wmape / self.mean_wmape < 0.25

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.folds])

    def summary(self) -> str:
        verdict = "stable" if self.is_stable else "UNSTABLE"
        return (
            f"{len(self.folds)} folds: mean WMAPE {self.mean_wmape:.1%} "
            f"+/- {self.std_wmape:.1%} - {verdict}"
        )


def expanding_window_backtest(
    panel: pd.DataFrame,
    estimator_factory: Any,
    *,
    approach: PromotionApproach,
    n_folds: int = 4,
    fold_days: int = 90,
    min_train_days: int = 365,
    date_column: str = "date",
) -> BacktestResult:
    """Expanding-window validation (brief section 12).

    Each fold trains on everything up to a cut-off and validates on the quarter
    after it, with the training window growing. Expanding rather than sliding
    because that is how the model would actually be retrained in production -
    you do not throw away last year's data when this quarter arrives.

    ``estimator_factory`` is called per fold so each gets a fresh, unfitted
    model. Reusing one instance would leak fold *n*'s fit into fold *n+1* and
    the "stability over time" reading would be meaningless.
    """
    dates = pd.to_datetime(panel[date_column]).dt.date
    first, last = dates.min(), dates.max()

    folds: list[BacktestFold] = []
    for index in range(n_folds):
        # Work backwards from the end so the last fold is the most recent data.
        valid_end = last - timedelta(days=fold_days * (n_folds - index - 1))
        valid_start = valid_end - timedelta(days=fold_days - 1)
        train_end = valid_start - timedelta(days=1)

        if (train_end - first).days < min_train_days:
            logger.warning(
                "baseline.backtest_fold_skipped",
                fold=index, reason="insufficient training history",
                available_days=(train_end - first).days, required=min_train_days,
            )
            continue

        train_rows, _ = prepare_training_rows(
            panel[(dates >= first) & (dates <= train_end)], approach=approach
        )
        valid_rows, _ = prepare_training_rows(
            panel[(dates >= valid_start) & (dates <= valid_end)],
            approach=PromotionApproach.EXCLUDE,
        )
        if train_rows.empty or valid_rows.empty:
            continue

        features = select_features(panel, approach=approach)
        estimator = estimator_factory()
        estimator.fit(train_rows[features], train_rows[TARGET])

        predict_frame = (
            neutralise_promotions(valid_rows)
            if approach is PromotionApproach.CONTROL
            else valid_rows
        )
        predictions = estimator.predict(predict_frame[features])

        folds.append(
            BacktestFold(
                index=index,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                metrics=compute_metrics(valid_rows[TARGET], predictions),
                train_rows=len(train_rows),
            )
        )
        logger.info(
            "baseline.backtest_fold",
            fold=index, wmape=round(folds[-1].metrics.wmape, 4), train_rows=len(train_rows),
        )

    return BacktestResult(folds=folds)
