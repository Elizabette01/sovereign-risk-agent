"""
passive_stress_test.py — Bank of England CBES / NGFS passive stress test benchmark.

The Bank of England Climate Biennial Exploratory Scenario (CBES) and the Network
for Greening the Financial System (NGFS) scenarios assess climate-related financial
risk by applying a predefined stress scenario to a *passive* institution that does
not adapt its behaviour in response to the scenario.

In the fiscal policy context, this translates to a government that:
    1. Maintains its current fiscal policy stance (no active adjustment).
    2. Faces heightened climate shocks in specific years of the episode.
    3. Does not invest in adaptation.

This benchmark is important because it establishes the cost of *inaction* — how
bad would outcomes be if the government simply did nothing? An RL agent that
outperforms this benchmark under climate stress is demonstrably doing better
than inaction.

References:
    Bank of England (2021). "Results of the 2021 Climate Biennial Exploratory
    Scenario (CBES)."
    NGFS (2023). "NGFS Climate Scenarios for Central Banks and Supervisors."

Placeholder: the climate stress path (which years get elevated shocks and by
how much) will be implemented in Notebook 07 using the NGFS Disorderly
Transition scenario parameters.
"""

from __future__ import annotations

import numpy as np
from .bohn_fiscal_reaction import FiscalPolicy


class PassiveStressTest(FiscalPolicy):
    """Implements a passive (no-response) fiscal policy during a climate stress scenario.

    The government always selects action 2 (Maintain current policy), meaning
    it makes no active fiscal adjustment regardless of the state. This creates
    a clear counterfactual: what happens if the government ignores climate risk?

    In Notebook 07, this will be extended to:
        1. Apply elevated climate shock probabilities in specified years
           (following NGFS Disorderly Transition scenario).
        2. Report debt trajectories under stress, comparable to BoE CBES outputs.

    Placeholder — to be implemented in Notebook 07.
    """

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Always maintain current policy (action 2 — no fiscal adjustment)."""
        return 2  # Maintain current policy

    def __repr__(self) -> str:
        return "PassiveStressTest(action=2 [maintain current policy])"
