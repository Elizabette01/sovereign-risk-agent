import numpy as np


def discretize_state(
    obs: np.ndarray,
    n_bins: int = 3,
) -> tuple[int, int, int, int]:
    """
    Convert continuous observation into discrete bins.

    obs = [debt_ratio, gdp_growth, inflation, climate_shock]
    Returns a tuple of 4 integers, each in [0, n_bins-1]
    """

    debt, growth, infl, shock = obs.tolist()

    # Define bin edges 
    debt_bins = np.linspace(0.0, 2.0, n_bins + 1)
    growth_bins = np.linspace(-0.10, 0.10, n_bins + 1)
    infl_bins = np.linspace(0.0, 0.20, n_bins + 1)
    shock_bins = np.linspace(0.0, 1.0, n_bins + 1)

    # np.digitize returns 1..n_bins, so we subtract 1 to get 0..n_bins-1
    debt_i = int(np.clip(np.digitize(debt, debt_bins) - 1, 0, n_bins - 1))
    growth_i = int(np.clip(np.digitize(growth, growth_bins) - 1, 0, n_bins - 1))
    infl_i = int(np.clip(np.digitize(infl, infl_bins) - 1, 0, n_bins - 1))
    shock_i = int(np.clip(np.digitize(shock, shock_bins) - 1, 0, n_bins - 1))

    return debt_i, growth_i, infl_i, shock_i
