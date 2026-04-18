"""
bohn_fiscal_reaction.py — Bohn (1998) fiscal reaction function benchmark.

The Bohn (1998) fiscal reaction function is the standard empirical benchmark
for debt sustainability. Bohn showed that if the government's primary balance
responds positively to the debt level (β > 0), this is a sufficient condition
for intertemporal solvency. It is therefore the natural benchmark against which
to evaluate RL agent performance: an agent that outperforms the Bohn rule is
doing something economically meaningful.

Reference:
    Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits."
    Quarterly Journal of Economics, 113(3), 949–963.

This file also defines:
    - FiscalPolicy: the abstract base class for all policy strategies.
    - RandomPolicy: a uniform-random baseline for environment testing.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# Base class — all benchmark strategies and RL wrappers share this interface
# ===========================================================================

class FiscalPolicy:
    """Abstract base class for fiscal policy strategies.

    All benchmarks (Bohn, SGP, passive) and RL agent wrappers must implement
    this interface so they can be evaluated on the same environment in a
    uniform loop in Notebook 07.
    """

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select an action given the current normalised observation.

        Parameters
        ----------
        observation:
            Normalised state vector of shape (7,).
        info:
            The info dict from the last reset()/step() call, which contains
            'raw_state' with the un-normalised values for rule-based policies
            that operate on interpretable values (e.g. debt level in % GDP).

        Returns
        -------
        int: action index in {0, 1, 2, 3, 4, 5}.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state at the start of a new episode."""
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


# ===========================================================================
# Random policy — uniform random action selection
# ===========================================================================

class RandomPolicy(FiscalPolicy):
    """Selects actions uniformly at random.

    Used for:
    1. Sanity-checking that the environment produces varied outcomes.
    2. Establishing the minimum baseline that any purposeful policy should beat.
    3. Initial exploration during RL training (via ε-greedy with high ε).

    Performance expectation: random policy should produce moderate debt
    accumulation with high variance, and frequent debt crises when applied to
    high-stress profiles (Developing_High).
    """

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        return int(np.random.randint(0, 6))


# ===========================================================================
# Bohn (1998) fiscal reaction function
# ===========================================================================

class BohnFiscalReaction(FiscalPolicy):
    """Implements the Bohn (1998) fiscal reaction function.

    The government adjusts its primary balance in response to the debt level:

        pb(t) = α + β × d(t−1) + γ × output_gap(t)

    where:
        α = constant (average primary balance)
        β = debt-responsiveness coefficient (must be > 0 for sustainability)
        γ = cyclical adjustment (output gap sensitivity)

    Bohn's key insight: if β > 0, the government "leans against" rising debt —
    it tightens fiscal policy when debt is high. This is both an empirical
    regularity and a theoretical condition for solvency.

    We map the desired primary balance to a discrete action by finding the
    action whose pb_change best implements the Bohn-implied adjustment:
    - Compute the target primary balance pb*(t) from the rule.
    - Compute the implied change: Δpb = pb*(t) − pb(t−1)
    - Select the action whose pb_change is closest to Δpb.

    Placeholder: α, β, and γ coefficients will be calibrated from the
    historical panel data in Notebook 07 using OLS (following Bohn, 1998).
    For now, we use plausible benchmark values from the empirical literature:
        α = 0.0   (balanced budget on average)
        β = 0.03  (3pp tightening per 100pp of excess debt — Bohn's US estimate)
        γ = 0.4   (40% of output gap offset by fiscal policy)

    Reference:
        Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits."
        Quarterly Journal of Economics, 113(3), 949–963.
    """

    # Action → (name, primary balance change in pp of GDP)
    _ACTION_CHANGES = {
        0: +3.0,   # Severe austerity
        1: +1.5,   # Moderate austerity
        2:  0.0,   # Maintain
        3: -1.5,   # Moderate stimulus
        4: -3.0,   # Large stimulus
        5: -1.0,   # Climate adaptation
    }

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.03,
        gamma: float = 0.4,
        debt_target: float = 60.0,
    ) -> None:
        """
        Parameters
        ----------
        alpha:
            Intercept of the fiscal reaction function (pp of GDP).
        beta:
            Debt-responsiveness coefficient. Must be > 0 for sustainability.
        gamma:
            Output gap sensitivity.
        debt_target:
            The debt level at which no adjustment is required (default: 60%).
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.debt_target = debt_target

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select the action closest to the Bohn-implied fiscal adjustment.

        Placeholder implementation — coefficients to be calibrated in Notebook 07.
        """
        raw = info.get("raw_state", {})
        debt = raw.get("debt", 60.0)
        growth = raw.get("growth", 2.0)
        growth_base = 2.5  # approximate trend — will be profile-specific in Notebook 07
        output_gap = growth - growth_base  # positive = above trend

        # Bohn's rule: target primary balance
        debt_gap = debt - self.debt_target
        pb_target = self.alpha + self.beta * debt_gap + self.gamma * output_gap

        # Current primary balance
        pb_current = raw.get("primary_balance", 0.0)

        # Implied change
        implied_delta = pb_target - pb_current

        # Find the discrete action whose pb_change is closest
        best_action = min(
            self._ACTION_CHANGES.keys(),
            key=lambda a: abs(self._ACTION_CHANGES[a] - implied_delta),
        )
        return best_action

    def __repr__(self) -> str:
        return (
            f"BohnFiscalReaction(α={self.alpha}, β={self.beta}, "
            f"γ={self.gamma}, debt_target={self.debt_target})"
        )
