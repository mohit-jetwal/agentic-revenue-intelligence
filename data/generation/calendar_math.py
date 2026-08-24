"""Vectorised time-shape helpers shared by several generators.

Kept separate from ``calendar_generator`` so that the pricing and demand
simulations can use the same seasonal shape without importing the calendar
builder, and so these functions can be unit-tested on their own.

That sharing is not incidental. The pricing generator uses the *same* seasonal
curve the demand equation uses, because that is precisely what creates price
endogeneity: a manager who raises prices into the seasonal peak makes price
correlate with the demand shock, which is the bias Step 8 has to overcome.
"""

from __future__ import annotations

import numpy as np


def annual_seasonality_series(
    day_of_year: np.ndarray,
    amplitude: float,
    peak_month: int,
) -> np.ndarray:
    """Smooth annual seasonal factor, additive in log space.

    One sine harmonic rather than twelve monthly dummies: smooth across month
    boundaries (no artificial jump on the 1st), described by two interpretable
    parameters, and still recoverable by a model using month features or Fourier
    terms.
    """
    peak_day = (peak_month - 1) * 30.44 + 15.0
    phase = 2.0 * np.pi * (np.asarray(day_of_year, dtype=float) - peak_day) / 365.25
    return amplitude * np.cos(phase)


def linear_trend(day_index: np.ndarray, annual_rate: float) -> np.ndarray:
    """Compound annual trend expressed additively in log space."""
    years = np.asarray(day_index, dtype=float) / 365.25
    return np.log1p(annual_rate) * years


def launch_ramp(days_since_launch: np.ndarray, ramp_days: int) -> np.ndarray:
    """Distribution build-up after a product launch, in [0, 1].

    Saturating rather than linear: a new SKU gains listings and shelf presence
    quickly at first, then plateaus. Scenario J depends on this curve being
    visibly gradual rather than a step change.
    """
    x = np.clip(np.asarray(days_since_launch, dtype=float) / max(ramp_days, 1), 0.0, None)
    return 1.0 - np.exp(-3.0 * x)


def exponential_decay(days_since_event: np.ndarray, half_life_days: float) -> np.ndarray:
    """Decay weight used for post-promotion pull-forward."""
    days = np.asarray(days_since_event, dtype=float)
    weight = np.exp(-np.log(2.0) * days / max(half_life_days, 1e-6))
    return np.where(days >= 0, weight, 0.0)
