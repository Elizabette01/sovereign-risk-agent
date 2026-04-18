"""
calibrated_bohn.py — Empirically calibrated Bohn (1998) fiscal reaction function.

This module provides `CalibratedBohn`, a subclass of `BohnFiscalReaction` that:
1. Uses OLS-estimated coefficients (α, β, γ) from the historical panel data
   (2000–2024, IMF WEO + World Bank) rather than placeholder values.
2. Uses the profile-specific long-run growth rate (growth_base) as the
   trend-growth anchor when computing the output gap, rather than the
   hardcoded 2.5% approximation in the base class.

Calibration methodology (Notebook 07a):
    pb(t) = α + β × d(t-1) + γ × output_gap(t) + ε(t)

    Estimated via pooled OLS within each economy type (Advanced / Emerging Market
    / Developing) over the period 2000–2024 on non-projection observations.
    Sample filtered to |primary balance| ≤ 50% of GDP and debt_lag1 ∈ [0, 300%]
    to remove data-quality outliers in the WEO panel.

Economic interpretation of key coefficients:
    β > 0 (Bohn's sustainability condition): the government tightens fiscal
        policy when debt rises, ensuring intertemporal solvency. Met for
        Advanced and Emerging Market; mixed for Developing (see Notebook 07a).
    γ > 0: counter-cyclical adjustment — primary balance improves when the
        economy is above trend (positive output gap), as automatic stabilisers
        operate and revenue increases.

Usage:
    from src.benchmarks.calibrated_bohn import CalibratedBohn, BOHN_COEFFICIENTS

    policy = CalibratedBohn.for_profile("Advanced_Low")
    action = policy.select_action(obs, info)

Reference:
    Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits."
    Quarterly Journal of Economics, 113(3), 949–963.
"""

from __future__ import annotations

from typing import Dict

from .bohn_fiscal_reaction import BohnFiscalReaction

# ---------------------------------------------------------------------------
# OLS-estimated Bohn coefficients from Notebook 07a
# Estimated on 2000–2024 non-projection panel, filtered for data quality.
# ---------------------------------------------------------------------------
BOHN_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    # OLS estimated in Notebook 07a on 2000-2024 WEO/World Bank panel.
    # Sample: non-projection observations, |pb| ≤ 50%, debt_lag1 ∈ [0%, 300%].
    # HC3 heteroskedasticity-robust standard errors.
    "Advanced": {
        "alpha":     0.0142,
        "beta":      0.0040,   # β > 0: Bohn sustainability condition MET
        "gamma":     0.4804,
        "r_squared": 0.0148,
        "n_obs":     1119,
        "se_beta":   0.0088,   # p=0.6454 — positive but not statistically significant
        "se_gamma":  0.1498,   # p=0.0013 — output gap highly significant
        "pv_beta":   0.6454,
        "pv_gamma":  0.0013,
    },
    "Emerging Market": {
        "alpha":    -1.8499,
        "beta":      0.0132,   # β > 0: Bohn sustainability condition MET
        "gamma":     0.4064,
        "r_squared": 0.0162,
        "n_obs":     809,
        "se_beta":   0.0101,   # p=0.1909 — positive but marginal significance
        "se_gamma":  0.1220,   # p=0.0009 — output gap highly significant
        "pv_beta":   0.1909,
        "pv_gamma":  0.0009,
    },
    "Developing": {
        "alpha":    -2.5346,
        "beta":     -0.0144,   # β < 0: sustainability condition NOT met.
        "gamma":     0.1828,   # Empirically common for developing economies with
        "r_squared": 0.0037,   # limited fiscal space (see dissertation Section 4).
        "n_obs":     966,
        "se_beta":   0.0077,   # p=0.0615 — marginally significant negative coefficient
        "se_gamma":  0.1521,   # p=0.2294 — output gap not significant
        "pv_beta":   0.0615,
        "pv_gamma":  0.2294,
    },
}


class CalibratedBohn(BohnFiscalReaction):
    """Bohn (1998) fiscal reaction function calibrated from historical data.

    Inherits the discrete action selection logic from `BohnFiscalReaction` but
    uses profile-specific trend growth for the output gap calculation and
    empirically estimated coefficients from Notebook 07a.

    Parameters
    ----------
    alpha, beta, gamma:
        OLS-estimated coefficients from Notebook 07a.
    debt_target:
        Debt-to-GDP threshold (default 60% — Maastricht criterion).
    growth_base:
        Profile-specific long-run growth rate, used as the trend for computing
        the output gap: output_gap = actual_growth − growth_base.
    economy_type:
        Economy type label (for reporting).
    profile_name:
        Full profile name (for reporting).
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        debt_target: float,
        growth_base: float,
        economy_type: str = "",
        profile_name: str = "",
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, debt_target=debt_target)
        self._growth_base = growth_base
        self.economy_type = economy_type
        self.profile_name = profile_name

    def select_action(self, observation, info: dict) -> int:
        """Select the nearest discrete action to the Bohn-implied adjustment.

        Uses profile-specific growth_base for the output gap, rather than
        the hardcoded 2.5% approximation in the parent class.
        """
        raw = info.get("raw_state", {})
        debt = raw.get("debt", 60.0)
        growth = raw.get("growth", self._growth_base)
        pb_current = raw.get("primary_balance", 0.0)

        # Output gap: how far is actual growth from trend growth?
        output_gap = growth - self._growth_base

        # Bohn's rule: target primary balance
        debt_gap = debt - self.debt_target
        pb_target = self.alpha + self.beta * debt_gap + self.gamma * output_gap

        # Implied annual change in primary balance
        implied_delta = pb_target - pb_current

        # Select the discrete action whose pb_change is closest to implied_delta
        best_action = min(
            self._ACTION_CHANGES.keys(),
            key=lambda a: abs(self._ACTION_CHANGES[a] - implied_delta),
        )
        return best_action

    def __repr__(self) -> str:
        return (
            f"CalibratedBohn({self.profile_name}, "
            f"α={self.alpha:.4f}, β={self.beta:.4f}, γ={self.gamma:.4f}, "
            f"g_base={self._growth_base:.2f})"
        )

    @classmethod
    def for_profile(
        cls,
        profile_name: str,
        config_path: str = "data/processed/transition_parameters.json",
        scaling_path: str = "data/processed/scaling_parameters.json",
        debt_target: float = 60.0,
    ) -> "CalibratedBohn":
        """Factory method: create a CalibratedBohn for any profile by name.

        Looks up the economy type, fetches OLS coefficients from
        BOHN_COEFFICIENTS, and reads growth_base from the profile config.

        Parameters
        ----------
        profile_name:
            One of the 9 profile keys, e.g. "Advanced_Low".
        config_path, scaling_path:
            Paths to the data files (relative to project root).
        debt_target:
            Debt target for the rule (default 60%).
        """
        from src.environment.config import load_profile

        cfg = load_profile(profile_name, config_path, scaling_path)
        coeffs = BOHN_COEFFICIENTS[cfg.economy_type]

        return cls(
            alpha=coeffs["alpha"],
            beta=coeffs["beta"],
            gamma=coeffs["gamma"],
            debt_target=debt_target,
            growth_base=cfg.growth_base,
            economy_type=cfg.economy_type,
            profile_name=profile_name,
        )
