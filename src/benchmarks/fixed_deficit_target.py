"""
fixed_deficit_target.py — EU Stability and Growth Pact (SGP) benchmark.

The EU SGP imposes two numerical fiscal rules:
    1. Overall deficit ≤ 3% of GDP (deficit criterion)
    2. Debt-to-GDP ≤ 60%, or declining sufficiently toward it (debt criterion)

A government following the SGP adjusts its fiscal stance to move toward these
targets each year. This is a widely used institutional benchmark in the sovereign
risk literature and is the natural comparator for rule-based fiscal policy.

Reference:
    European Commission (2020). "Vade Mecum on the Stability and Growth Pact."
    European Economy Institutional Paper 129.

Placeholder: the full SGP rule (including the structural balance methodology,
cyclical adjustments, and medium-term objective) will be implemented in Notebook 07.
"""

from __future__ import annotations

import numpy as np
from .bohn_fiscal_reaction import FiscalPolicy


class FixedDeficitTarget(FiscalPolicy):
    """Implements simplified fixed deficit targeting based on the EU SGP.

    The government targets:
        - Structural deficit ≤ deficit_target (default: 3% of GDP)
        - If debt > debt_target (60%), also aim for annual debt reduction

    Strategy:
        1. If the current primary balance implies a deficit > deficit_target,
           tighten fiscal policy (choose austerity actions).
        2. If debt is above the debt_target, apply additional tightening.
        3. If targets are met, maintain current policy.
        4. Allow moderate stimulus when well within targets.

    The mapping to discrete actions mirrors the Bohn benchmark: compute a
    target primary balance implied by the SGP rule, then select the nearest
    discrete action.

    Placeholder — to be fully calibrated and validated in Notebook 07.
    """

    _ACTION_CHANGES = {
        0: +3.0,
        1: +1.5,
        2:  0.0,
        3: -1.5,
        4: -3.0,
        5: -1.0,
    }

    def __init__(
        self,
        deficit_target: float = -3.0,
        debt_target: float = 60.0,
        adjustment_speed: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        deficit_target:
            The maximum tolerated deficit as a percentage of GDP
            (negative = deficit). Default: −3.0 (EU SGP criterion).
        debt_target:
            The debt-to-GDP target. Default: 60.0 (EU Maastricht criterion).
        adjustment_speed:
            Fraction of the gap to close in one year (0 = no adjustment,
            1 = close the full gap in one year).
        """
        self.deficit_target = deficit_target
        self.debt_target = debt_target
        self.adjustment_speed = adjustment_speed

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select action to move toward SGP compliance.

        Placeholder implementation — full SGP logic in Notebook 07.
        """
        raw = info.get("raw_state", {})
        pb = raw.get("primary_balance", 0.0)
        debt = raw.get("debt", 60.0)

        # Target primary balance: must at minimum cover the deficit criterion.
        # For debt above 60%, add an additional consolidation requirement.
        pb_target = self.deficit_target
        if debt > self.debt_target:
            # 1/20th rule: debt above 60% should decrease by 1/20th per year
            excess_debt = debt - self.debt_target
            pb_target += self.adjustment_speed * (excess_debt / 20.0)

        implied_delta = (pb_target - pb) * self.adjustment_speed

        best_action = min(
            self._ACTION_CHANGES.keys(),
            key=lambda a: abs(self._ACTION_CHANGES[a] - implied_delta),
        )
        return best_action

    def __repr__(self) -> str:
        return (
            f"FixedDeficitTarget(deficit_target={self.deficit_target}%, "
            f"debt_target={self.debt_target}%)"
        )
