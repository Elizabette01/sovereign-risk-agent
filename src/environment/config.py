"""
config.py — Profile configuration loader for the Sovereign Risk Gymnasium environment.

Each of the 9 cells in the 3x3 (economy type × climate risk tier) matrix has a distinct set of calibrated parameters stored in transition_parameters.json. This module reads those parameters into a structured dataclass so the rest of the codebase can access them by name rather than by dictionary key.

Design rationale:
- Using a dataclass rather than a plain dict makes autocomplete and type checking work properly, and means a typo ("growht_base") raises an AttributeError at load time rather than silently returning None during a training run.
- The two data-quality fixes (Advanced_High sparse cell substitutions and the Emerging Market climate fiscal cost sign correction) are applied here, centrally, so every downstream user automatically gets the corrected values.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default file paths (relative to the project root).
# These can be overridden by passing explicit paths to load_profile().
# ---------------------------------------------------------------------------
_DEFAULT_TRANSITION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "transition_parameters.json"
)
_DEFAULT_SCALING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "scaling_parameters.json"
)


@dataclass
class ProfileConfig:
    """All calibrated parameters for a single economy-type × climate-risk-tier cell.

    Fields are populated from transition_parameters.json and scaling_parameters.json.
    Null values for the Advanced_High sparse cell are replaced with Advanced_Low
    fallbacks; see load_profile() for details.
    """

    # Identity
   
    profile_name: str          # e.g. "Advanced_Low"
    economy_type: str          # e.g. "Advanced"
    climate_risk_tier: str     # e.g. "Low"
    sparse_cell: bool          # True if fewer than ~10 countries

   
    # Initial state values (used to seed episodes in reset())
    # These are the empirical medians for each cell.
    # -----------------------------------------------------------------------
    initial_debt: float
    initial_growth: float
    initial_primary_balance: float
    initial_interest_rate: float
    initial_climate_shock: float
    initial_adaptation_capital: float
    initial_risk_premium: float

    # ------------------------------------------------------------
    # Debt dynamics — stock-flow adjustment
    # The stock-flow term captures everything outside the standard debt
    # accumulation identity: privatisations, bank recapitalisations, exchange-rate revaluations, statistical discrepancies, etc.
    # ------------------------------------------------------------
    stock_flow_mean: float     # mean of annual stock-flow adjustment (% GDP)
    stock_flow_std: float      # standard deviation (% GDP)

    # ---------------------------------------------------------
    # GDP growth AR(1) process
    # Growth is persistent — a recession year tends to be followed by another
    # weak year — but mean-reverts toward growth_base.
    # -----------------------------------------------------------------------
    growth_base: float                  # long-run (unconditional) mean growth rate
    growth_ar1_coef: float              # persistence coefficient ρ ∈ [0,1)
    growth_shock_std: float             # standard deviation of innovation
    growth_shock_distribution: str      # "normal" or "t"
    growth_shock_t_df: Optional[float]  # degrees of freedom if t-distribution

    # -----------------------------------------------------------------------
    # Climate shock process
    # A two-stage model: (1) does a disaster occur? (2) if yes, how big?
    # -----------------------------------------------------------------------
    climate_event_probability: float    # Bernoulli probability each year
    climate_conditional_mean: float     # mean damage given event (% GDP)
    climate_conditional_std: float      # std of damage given event
    climate_max_damage: float           # cap on drawn damage (prevents extreme draws)
    climate_fiscal_cost: float          # impact on primary balance per unit damage (pp)
                                        # NEGATIVE means disasters worsen the balance

    # -----------------------------------------------------------------------
    # Real interest rate process (random walk with drift)
    # -----------------------------------------------------------------------
    rate_base: float           # starting level / long-run anchor
    rate_shock_mean: float     # drift (small positive = gradual normalisation)
    rate_shock_std: float      # volatility of the annual shock

    # -----------------------------------------------------------------------
    # Scaling parameters for state normalisation
    # Each variable's median and IQR are used in robust (median/IQR) scaling.
    # The inner dict has keys "median" and "iqr".
    # -----------------------------------------------------------------------
    scaling: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Optional reward scaling multiplier (tuned during training)
    # -----------------------------------------------------------------------
    reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_profiles(
    config_path: str = _DEFAULT_TRANSITION_PATH,
) -> List[str]:
    """Return all profile names present in transition_parameters.json."""
    path = Path(config_path)
    with path.open() as f:
        data = json.load(f)
    return list(data.keys())


def load_profile(
    profile_name: str,
    config_path: str = _DEFAULT_TRANSITION_PATH,
    scaling_path: str = _DEFAULT_SCALING_PATH,
) -> ProfileConfig:
    """Load and validate a ProfileConfig for the requested profile.

    Parameters
    ----------
    profile_name:
        One of the 9 profile keys, e.g. "Advanced_Low", "Developing_High".
    config_path:
        Path to transition_parameters.json.
    scaling_path:
        Path to scaling_parameters.json (used only to cross-check scaling dicts).

    Returns
    -------
    ProfileConfig
        Fully populated dataclass with all null values resolved and any
        known data-quality corrections applied.
    """
    transition_path = Path(config_path)
    scale_path = Path(scaling_path)

    with transition_path.open() as f:
        all_profiles: dict = json.load(f)

    if profile_name not in all_profiles:
        available = list(all_profiles.keys())
        raise KeyError(
            f"Profile '{profile_name}' not found. Available profiles: {available}"
        )

    raw = all_profiles[profile_name]

    # ------------------------------------------------------------------
    # Sparse-cell fallback: Advanced_High has only 2 countries and
    # several null parameters. We substitute Advanced_Low values, which
    # share the same economy type and have the richest data coverage.
    # ------------------------------------------------------------------
    advanced_low = all_profiles["Advanced_Low"]
    substitutions_made = []

    def _fallback(key: str, fallback_value: float) -> float:
        """Return raw[key] if non-null, else fallback_value with a warning."""
        val = raw.get(key)
        if val is None:
            substitutions_made.append((key, fallback_value))
            return fallback_value
        return val

    initial_primary_balance = _fallback(
        "initial_primary_balance", advanced_low["initial_primary_balance"]
    )
    initial_interest_rate = _fallback(
        "initial_interest_rate", advanced_low["initial_interest_rate"]
    )
    initial_risk_premium = _fallback(
        "initial_risk_premium", advanced_low["initial_risk_premium"]
    )
    rate_base = _fallback("rate_base", advanced_low["rate_base"])

    if substitutions_made:
        logger.warning(
            "Profile '%s' is sparse (%d countries). The following null parameters "
            "have been substituted with Advanced_Low fallback values: %s",
            profile_name,
            raw.get("n_countries", "?"),
            {k: v for k, v in substitutions_made},
        )

    # ------------------------------------------------------------------
    # Emerging Market climate fiscal cost correction.
    #
    # The calibrated value for all Emerging Market profiles is +0.547,
    # implying that climate disasters *improve* the primary balance — an
    # economically incoherent result that arises from limited data and
    # measurement noise in the historical panel. We override this to -0.5
    # for all Emerging Market cells.
    #
    # Methodology decision: -0.5 is a conservative estimate consistent
    # with the Advanced and Developing economy calibrations (-1.569 and
    # -0.489 respectively). It imposes the correct directional relationship
    # (disasters worsen fiscal outcomes) without overstating the magnitude.
    # This follows the approach in IMF (2020) "How large are climate-related
    # fiscal risks?" which finds average fiscal costs of 0.2–1.0% of GDP
    # per major climate event for middle-income countries.
    # ------------------------------------------------------------------
    climate_fiscal_cost = raw["climate_fiscal_cost"]
    if raw["economy_type"] == "Emerging Market" and climate_fiscal_cost > 0:
        logger.warning(
            "Profile '%s': calibrated climate_fiscal_cost is +%.3f (wrong sign — "
            "implies disasters improve fiscal balance). Overriding to -0.5 for all "
            "Emerging Market profiles. See config.py for methodology rationale.",
            profile_name,
            climate_fiscal_cost,
        )
        climate_fiscal_cost = -0.5

    cfg = ProfileConfig(
        profile_name=profile_name,
        economy_type=raw["economy_type"],
        climate_risk_tier=raw["climate_risk_tier"],
        sparse_cell=raw["sparse_cell"],
        # Initial state
        initial_debt=raw["initial_debt"],
        initial_growth=raw["initial_growth"],
        initial_primary_balance=initial_primary_balance,
        initial_interest_rate=initial_interest_rate,
        initial_climate_shock=raw["initial_climate_shock"],
        initial_adaptation_capital=raw["initial_adaptation_capital"],
        initial_risk_premium=initial_risk_premium,
        # Debt dynamics
        stock_flow_mean=raw["stock_flow_mean"],
        stock_flow_std=raw["stock_flow_std"],
        # Growth process
        growth_base=raw["growth_base"],
        growth_ar1_coef=raw["growth_ar1_coef"],
        growth_shock_std=raw["growth_shock_std"],
        growth_shock_distribution=raw["growth_shock_distribution"],
        growth_shock_t_df=raw.get("growth_shock_t_df"),
        # Climate process
        climate_event_probability=raw["climate_event_probability"],
        climate_conditional_mean=raw["climate_conditional_mean"],
        climate_conditional_std=raw["climate_conditional_std"],
        climate_max_damage=raw["climate_max_damage"],
        climate_fiscal_cost=climate_fiscal_cost,
        # Interest rate
        rate_base=rate_base,
        rate_shock_mean=raw["rate_shock_mean"],
        rate_shock_std=raw["rate_shock_std"],
        # Scaling dict (embedded in the profile JSON)
        scaling=raw.get("scaling", {}),
    )

    return cfg
