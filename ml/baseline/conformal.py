"""Split conformal prediction intervals (brief section 18).

Section 18 is emphatic: do not return ``confidence = 0.92`` unless it is
statistically justified. This module is what makes the number justified.

**How it works.** Hold out a calibration fold the model never trained on.
Predict it, collect the absolute residuals, take their empirical
``(1-alpha)`` quantile. That width, added either side of a point prediction,
gives an interval with a finite-sample coverage guarantee - no distributional
assumption about the errors, no assumption that the model is well specified.

**Why conformal over the alternatives.** Quantile regression means training
three models instead of one and its quantiles are only as calibrated as the fit.
Bootstrap needs many refits. Conformal needs one extra prediction pass and comes
with a proof.

**The assumption it does make** is exchangeability between calibration and test
data. A temporal split violates that mildly - the world drifts - so the
guarantee is approximate here. Which is exactly why
:func:`measure_coverage` exists and why the achieved coverage is reported on
test data rather than assumed. A 90% interval that covers 71% is a finding, and
the only way to have that finding is to look.

**Scaled residuals.** Absolute residuals produce a constant-width interval, which
is wrong for count data: a hero SKU selling 500/day and a slow mover selling
3/day do not deserve the same +/- 40. Normalising the residual by the prediction
before taking the quantile gives a multiplicative interval that widens with
volume - the same reasoning that makes WMAPE the right headline metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.observability.logging import get_logger

logger = get_logger(__name__)

#: Floor on the prediction used as a scaling denominator. Without it a near-zero
#: prediction produces an enormous normalised residual and one slow-moving SKU
#: sets the interval width for the entire catalogue.
_SCALE_FLOOR = 1.0


@dataclass(frozen=True)
class ConformalCalibration:
    """A fitted conformal calibration.

    Immutable, and carries the numbers needed to audit it: how many points it
    was calibrated on, and what the residual distribution looked like.
    """

    #: Nominal miscoverage. 0.1 gives a 90% interval.
    alpha: float
    #: Quantile of the normalised absolute residual.
    quantile: float
    #: Points the quantile was estimated from.
    n_calibration: int
    scaled: bool
    #: Distribution summary, for the model card and for diagnosing a bad fit.
    residual_median: float
    residual_p90: float

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def interval(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds for point predictions.

        The lower bound is clipped at zero: demand cannot be negative, and a
        baseline whose lower bound is -20 units would let the "is this gap
        significant?" test in Step 17 pass on nonsense.
        """
        point = np.asarray(predictions, dtype=float)
        width = self.quantile * np.maximum(point, _SCALE_FLOOR) if self.scaled else self.quantile
        return np.clip(point - width, 0.0, None), point + width

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": round(self.nominal_coverage, 4),
            "quantile": round(self.quantile, 6),
            "n_calibration": self.n_calibration,
            "scaled": self.scaled,
            "residual_median": round(self.residual_median, 6),
            "residual_p90": round(self.residual_p90, 6),
        }


def calibrate(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    alpha: float = 0.1,
    scaled: bool = True,
) -> ConformalCalibration:
    """Fit a conformal calibration on a held-out fold.

    The fold must be one the model did **not** train on. Calibrating on training
    residuals produces intervals that are far too narrow, because training
    residuals understate the error the model makes on data it has not seen -
    and the resulting interval would then be exactly the fabricated confidence
    section 18 forbids.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    usable = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[usable], yhat[usable]

    if y.size < 100:
        raise ValueError(
            f"conformal calibration needs a meaningful sample; got {y.size} usable "
            f"points. An empirical quantile from a handful of residuals is noise."
        )

    residuals = np.abs(y - yhat)
    if scaled:
        residuals = residuals / np.maximum(yhat, _SCALE_FLOOR)

    # The finite-sample correction. Using the plain (1-alpha) quantile
    # under-covers slightly; ceil((n+1)(1-alpha))/n is the level that carries
    # the guarantee. It matters at small n and costs nothing at large n.
    n = residuals.size
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    quantile = float(np.quantile(residuals, level))

    calibration = ConformalCalibration(
        alpha=alpha,
        quantile=quantile,
        n_calibration=int(n),
        scaled=scaled,
        residual_median=float(np.median(residuals)),
        residual_p90=float(np.quantile(residuals, 0.9)),
    )
    logger.info("conformal.calibrated", **calibration.to_dict())
    return calibration


@dataclass(frozen=True)
class CoverageReport:
    """Achieved coverage, measured rather than claimed."""

    nominal: float
    empirical: float
    n: int
    mean_width: float
    #: Mean width as a share of the mean actual - an interval nobody can act on
    #: is not useful even when it covers.
    relative_width: float

    @property
    def gap(self) -> float:
        """Empirical minus nominal. Negative means under-covering."""
        return self.empirical - self.nominal

    @property
    def is_calibrated(self) -> bool:
        """Within 5 percentage points of nominal.

        A tolerance rather than exactness because the temporal split breaks
        exchangeability, so some drift is expected and honest.
        """
        return abs(self.gap) <= 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_coverage": round(self.nominal, 4),
            "empirical_coverage": round(self.empirical, 4),
            "coverage_gap": round(self.gap, 4),
            "is_calibrated": self.is_calibrated,
            "n": self.n,
            "mean_width": round(self.mean_width, 3),
            "relative_width": round(self.relative_width, 4),
        }

    def summary(self) -> str:
        verdict = "calibrated" if self.is_calibrated else "MISCALIBRATED"
        return (
            f"{self.nominal:.0%} interval covers {self.empirical:.1%} on {self.n:,} "
            f"test rows ({self.gap:+.1%}) - {verdict}. Mean width "
            f"{self.mean_width:.1f} units ({self.relative_width:.0%} of mean actual)."
        )


def measure_coverage(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    calibration: ConformalCalibration,
) -> CoverageReport:
    """Measure how often the interval actually contains the truth.

    The step that separates a justified interval from a decorative one. Run on
    **test** data - a fold used neither for training nor for calibration -
    otherwise this measures the calibration set's own quantile and returns the
    nominal level by construction.
    """
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    usable = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[usable], yhat[usable]

    if y.size == 0:
        return CoverageReport(
            nominal=calibration.nominal_coverage, empirical=float("nan"),
            n=0, mean_width=float("nan"), relative_width=float("nan"),
        )

    lower, upper = calibration.interval(yhat)
    covered = (y >= lower) & (y <= upper)
    widths = upper - lower
    mean_actual = float(np.mean(np.abs(y)))

    report = CoverageReport(
        nominal=calibration.nominal_coverage,
        empirical=float(covered.mean()),
        n=int(y.size),
        mean_width=float(widths.mean()),
        relative_width=float(widths.mean() / mean_actual) if mean_actual > 1e-9 else float("nan"),
    )

    if not report.is_calibrated:
        # Surfaced, not swallowed. An under-covering interval makes the
        # "is this gap significant?" question in Step 17 answer yes too often.
        logger.warning("conformal.miscalibrated", **report.to_dict())
    else:
        logger.info("conformal.coverage_measured", **report.to_dict())

    return report


def add_intervals(
    frame: pd.DataFrame,
    calibration: ConformalCalibration,
    *,
    predicted_column: str = "baseline_units",
    actual_column: str | None = "actual_units",
) -> pd.DataFrame:
    """Attach interval bounds and, where actuals exist, a significance flag.

    ``is_significant`` means something precise: the actual falls outside the
    prediction interval, so the gap is larger than the model's normal error.
    That is the test Step 17's root-cause agent needs before claiming a decline
    is real rather than noise - and it is only meaningful because the coverage
    was measured.
    """
    result = frame.copy()
    lower, upper = calibration.interval(result[predicted_column].to_numpy(dtype=float))
    result["baseline_lower"] = lower
    result["baseline_upper"] = upper
    result["interval_coverage"] = calibration.nominal_coverage

    if actual_column and actual_column in result.columns:
        actual = result[actual_column].to_numpy(dtype=float)
        result["is_significant"] = (actual < lower) | (actual > upper)

    return result
