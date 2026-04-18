"""
scaling.py — State normalisation and denormalisation utilities.

The 7 state variables span very different numerical ranges:
  - debt_to_gdp:   0 – 300 %
  - interest_rate: -80 – 85 % (winsorised)
  - climate_shock: 0 – 10 % of GDP
  - adaptation:    0 – 1 (ND-GAIN readiness index)
  ...

Neural networks (DQN/PPO) work poorly when inputs are on wildly different scales.
Robust scaling (subtract median, divide by IQR) is preferred over min–max or
z-score normalisation because it is resistant to the extreme outliers present in
this dataset (e.g. Iran's hyperinflation-era interest rates, Indonesia's 1997
crisis growth collapse).

Critical clipping to [-10, 10]:
Without clipping, outlier countries produce normalised values in the thousands,
which cause gradient explosions in neural networks. Clipping at ±10 IQRs from the
median retains over 99% of meaningful economic variation while bounding inputs to
a range where neural networks are numerically stable.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Canonical ordering of state variables.
# This order is used consistently throughout the environment and must match
# the order in which arrays are assembled in sovereign_risk_env.py.
# ---------------------------------------------------------------------------
STATE_VARIABLES: list[str] = [
    "state_output_growth",      # index 0 — real GDP growth rate (%)
    "state_debt_to_gdp",        # index 1 — gross government debt (% GDP)
    "state_primary_balance",    # index 2 — primary balance (% GDP, + = surplus)
    "state_interest_rate",      # index 3 — real interest rate (%)
    "state_climate_shock",      # index 4 — climate damage (% GDP)
    "state_adaptation_capital", # index 5 — ND-GAIN readiness index [0, 1]
    "state_risk_premium",       # index 6 — interest–growth differential (r − g)
]

# Clip bound: normalised values are capped at this magnitude to prevent
# outlier-driven numerical instability in neural network training.
_CLIP_BOUND: float = 10.0


def normalise_state(
    raw_state: np.ndarray,
    scaling_params: dict,
) -> np.ndarray:
    """Convert raw state values to normalised values using robust (median/IQR) scaling.

    For each of the 7 state variables:
        normalised_i = (raw_i − median_i) / IQR_i

    The result is clipped to [−10, 10] to prevent extreme outlier values
    from destabilising neural network training.

    Parameters
    ----------
    raw_state:
        1-D array of shape (7,) containing un-normalised state values, ordered
        according to STATE_VARIABLES.
    scaling_params:
        Dict with one key per STATE_VARIABLES entry. Each value is a sub-dict
        with "median" (float) and "iqr" (float).

    Returns
    -------
    np.ndarray of shape (7,) with dtype float32.

    Raises
    ------
    ValueError
        If raw_state does not have shape (7,).
    """
    if raw_state.shape != (7,):
        raise ValueError(
            f"raw_state must have shape (7,), got {raw_state.shape}"
        )

    normalised = np.empty(7, dtype=np.float32)
    for i, var in enumerate(STATE_VARIABLES):
        params = scaling_params[var]
        median = float(params["median"])
        iqr = float(params["iqr"])
        # Guard against zero IQR (constant variable in a very sparse cell)
        if iqr < 1e-8:
            normalised[i] = 0.0
        else:
            normalised[i] = (raw_state[i] - median) / iqr

    # Clip to prevent outlier-driven instability
    np.clip(normalised, -_CLIP_BOUND, _CLIP_BOUND, out=normalised)

    # NaN guard: if any upstream computation produced NaN, raise immediately
    if np.any(np.isnan(normalised)):
        raise RuntimeError(
            f"NaN detected in normalised state. Raw state was: {raw_state}"
        )

    return normalised


def denormalise_state(
    norm_state: np.ndarray,
    scaling_params: dict,
) -> np.ndarray:
    """Convert a normalised state back to raw (interpretable) values.

    Inverse of normalise_state:
        raw_i = normalised_i × IQR_i + median_i

    Parameters
    ----------
    norm_state:
        1-D array of shape (7,) of normalised state values.
    scaling_params:
        Same format as in normalise_state.

    Returns
    -------
    np.ndarray of shape (7,) with dtype float64.
    """
    if norm_state.shape != (7,):
        raise ValueError(
            f"norm_state must have shape (7,), got {norm_state.shape}"
        )

    raw = np.empty(7, dtype=np.float64)
    for i, var in enumerate(STATE_VARIABLES):
        params = scaling_params[var]
        median = float(params["median"])
        iqr = float(params["iqr"])
        raw[i] = norm_state[i] * iqr + median

    return raw
