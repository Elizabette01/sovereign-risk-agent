"""
sovereign_risk_env.py — Core Gymnasium environment for the Sovereign Risk Agent.

This environment simulates a government managing its fiscal policy over a 30-year
horizon in the presence of stochastic economic and climate shocks. It is the
central training arena for DQN and PPO agents developed in Notebooks 07–09.

Architecture:
    The class is intentionally thin. All economic mechanics live in separate,
    independently testable modules:
        dynamics.py  — growth, rate, climate, and debt accumulation
        reward.py    — multi-objective reward function
        config.py    — parameter loading and data-quality fixes

    The environment class is responsible only for:
        1. Orchestrating calls to the above modules in the correct sequence.
        2. Managing the Gymnasium API (reset/step/close lifecycle).
        3. Normalising observations before returning them to the agent.

Gymnasium compliance:
    - observation_space: Box(7,) float32, clipped to [−10, 10]
    - action_space: Discrete(6)
    - reset() returns (obs, info)
    - step() returns (obs, reward, terminated, truncated, info)
    - check_env() passes (verified in tests/test_environment.py)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ProfileConfig, load_profile
from .policy_dynamics import (
    compute_next_debt,
    generate_climate_shock,
    generate_growth,
    generate_interest_rate,
    update_adaptation_capital,
)
from .reward import compute_reward
from ..utils.scaling import STATE_VARIABLES, normalise_state

logger = logging.getLogger(__name__)


class SovereignRiskEnv(gym.Env):
    """Gymnasium environment simulating sovereign fiscal dynamics under climate risk.

    The agent plays the role of a fiscal policymaker choosing annual fiscal
    adjustments over a 30-year horizon. Each episode represents one possible
    30-year future for a given economy-type × climate-risk-tier cell.

    State space (7-dimensional, normalised to [−10, 10]):
        0. output_growth        — real GDP growth rate (%)
        1. debt_to_gdp          — gross government debt (% of GDP)
        2. primary_balance      — primary fiscal balance (% of GDP)
        3. interest_rate        — real interest rate (%)
        4. climate_shock        — climate damage this period (% of GDP)
        5. adaptation_capital   — ND-GAIN readiness index [0, 1]
        6. risk_premium         — interest–growth differential r − g (pp)

    Action space (6 discrete actions):
        0. Severe austerity             (+3.0 pp improvement in primary balance)
        1. Moderate austerity           (+1.5 pp improvement)
        2. Maintain current policy      (no change)
        3. Moderate stimulus            (−1.5 pp, worsens balance)
        4. Large stimulus               (−3.0 pp, worsens balance)
        5. Climate adaptation invest.   (−1.0 pp, but raises adaptation capital)

    Episode termination:
        - After max_steps years (default 30)
        - Early termination if debt > crisis_threshold (default 200% of GDP)
          with a large one-off penalty of −50 applied to the final reward.

    Args:
        profile:          Name of the calibration profile, e.g. "Advanced_Low".
        config_path:      Path to transition_parameters.json.
        scaling_path:     Path to scaling_parameters.json.
        seed:             Random seed. Passed to np.random.default_rng().
        max_steps:        Episode length in years.
        crisis_threshold: Debt level (% GDP) that triggers early termination.
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Action definitions
    # Each entry: (human-readable name, change in primary balance pp of GDP)
    # Literature references:
    #   Actions 0–1: consolidation episodes, Alesina & Ardagna (2010)
    #     "Large Changes in Fiscal Policy: Taxes vs Spending",
    #     typical size 1–3% of GDP per year.
    #   Action 2: status quo — no active policy change.
    #   Actions 3–4: stimulus, Blanchard & Leigh (2013) "Growth Forecast
    #     Errors and Fiscal Multipliers", typical 1–3% of GDP per year.
    #   Action 5: climate adaptation investment, Hallegatte et al. (2019)
    #     "Lifelines: The Resilient Infrastructure Opportunity",
    #     typical 0.5–1.5% of GDP for climate-vulnerable countries.
    # ------------------------------------------------------------------
    ACTION_MAP: Dict[int, Tuple[str, float]] = {
        0: ("Severe austerity",              +3.0),
        1: ("Moderate austerity",            +1.5),
        2: ("Maintain current policy",        0.0),
        3: ("Moderate stimulus",             -1.5),
        4: ("Large stimulus",                -3.0),
        5: ("Climate adaptation investment", -1.0),
    }

    def __init__(
        self,
        profile: str = "Advanced_Low",
        config_path: str = "data/processed/transition_parameters.json",
        scaling_path: str = "data/processed/scaling_parameters.json",
        seed: int = 42,
        max_steps: int = 30,
        crisis_threshold: float = 200.0,
    ) -> None:
        super().__init__()

        self.profile_name = profile
        self.max_steps = max_steps
        self.crisis_threshold = crisis_threshold

        # Load calibrated parameters
        self.config: ProfileConfig = load_profile(
            profile_name=profile,
            config_path=config_path,
            scaling_path=scaling_path,
        )

        # ------------------------------------------------------------------
        # Gymnasium spaces
        # ------------------------------------------------------------------
        # Observation: 7 normalised state variables, each clipped to [−10, 10]
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Action: 6 discrete fiscal policy choices
        self.action_space = spaces.Discrete(6)

        # ------------------------------------------------------------------
        # Internal state (initialised in reset())
        # ------------------------------------------------------------------
        self.raw_state: Dict[str, float] = {}
        self.current_step: int = 0
        self.episode_history: List[Dict[str, Any]] = []

        # Seed the RNG at construction time (reset() can override)
        self._seed = seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        logger.info(
            "SovereignRiskEnv created: profile=%s, max_steps=%d, crisis_threshold=%.1f",
            profile,
            max_steps,
            crisis_threshold,
        )

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial conditions for a new episode.

        The initial state is drawn from the calibrated profile medians, with a
        small random perturbation to create variety across episodes. Without
        this perturbation, every episode would start identically, which slows
        RL training because the agent never encounters diverse starting conditions.

        The perturbation is drawn from N(0, 0.1 × IQR) for each variable —
        small enough that the initial state always looks like a realistic economy
        of the given profile type, but large enough to meaningfully vary the
        challenge across episodes.

        Parameters
        ----------
        seed:
            Optional override for the random seed. If None, the seed passed at
            construction time is used.
        options:
            Unused (required by Gymnasium API).

        Returns
        -------
        observation: np.ndarray of shape (7,) — normalised initial state.
        info: dict — contains raw_state, profile name, step count (0),
              and empty episode_history list.
        """
        # Gymnasium's super().reset() sets self.np_random correctly
        super().reset(seed=seed)

        cfg = self.config
        scaling = cfg.scaling

        # ------------------------------------------------------------------
        # Build the initial raw state with small perturbations
        # For each variable, we perturb by ± a small fraction of the IQR.
        # ------------------------------------------------------------------
        def _perturb(initial_val: float, scaling_key: str) -> float:
            iqr = scaling[scaling_key]["iqr"]
            noise = self.np_random.normal(0.0, 0.1 * iqr)
            return initial_val + noise

        self.raw_state = {
            "growth":             _perturb(cfg.initial_growth,            "state_output_growth"),
            "debt":               _perturb(cfg.initial_debt,              "state_debt_to_gdp"),
            "primary_balance":    _perturb(cfg.initial_primary_balance,   "state_primary_balance"),
            "interest_rate":      _perturb(cfg.initial_interest_rate,     "state_interest_rate"),
            "climate_damage":     max(0.0, _perturb(cfg.initial_climate_shock,  "state_climate_shock")),
            "adaptation_capital": float(np.clip(
                _perturb(cfg.initial_adaptation_capital, "state_adaptation_capital"), 0.0, 1.0
            )),
            "risk_premium":       _perturb(cfg.initial_risk_premium,      "state_risk_premium"),
        }

        # Ensure debt is non-negative
        self.raw_state["debt"] = max(self.raw_state["debt"], 0.0)

        self.current_step = 0
        self.episode_history = []

        observation = self._get_observation()

        info = {
            "raw_state": self.raw_state.copy(),
            "profile": self.profile_name,
            "step": 0,
            "episode_history": self.episode_history,
        }

        return observation, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step (one fiscal year).

        Sequence of operations within a single step:
            1. Apply the chosen fiscal policy action (modify primary balance).
            2. Generate stochastic shocks: growth, interest rate, climate.
            3. Apply climate fiscal impact to the primary balance.
            4. Evolve debt using the government budget constraint.
            5. Update adaptation capital.
            6. Compute the risk premium (r − g).
            7. Compute the multi-objective reward.
            8. Check termination conditions (crisis or max steps).
            9. Assemble observation, info dict, and return.

        Parameters
        ----------
        action:
            Integer in {0, 1, 2, 3, 4, 5}.

        Returns
        -------
        observation: np.ndarray (7,) — normalised state.
        reward: float — scalar reward signal.
        terminated: bool — True if debt crisis triggered early termination.
        truncated: bool — True if episode reached max_steps.
        info: dict — full step metadata including raw_state and reward_components.
        """
        assert self.action_space.contains(action), (
            f"Invalid action {action}. Must be in 0–{self.action_space.n - 1}."
        )

        action_name, pb_change = self.ACTION_MAP[action]
        is_adaptation = (action == 5)

        # ------------------------------------------------------------------
        # 1. Apply fiscal policy: modify the primary balance
        # ------------------------------------------------------------------
        new_primary_balance = self.raw_state["primary_balance"] + pb_change

        # ------------------------------------------------------------------
        # 2. Generate stochastic shocks
        # All randomness flows through self.np_random for seed control.
        # ------------------------------------------------------------------
        new_growth = generate_growth(
            self.raw_state["growth"], self.config, self.np_random
        )
        new_rate = generate_interest_rate(
            self.raw_state["interest_rate"], self.config, self.np_random
        )
        climate_damage, climate_fiscal_cost = generate_climate_shock(
            self.config, self.np_random
        )
        stock_flow_shock = float(self.np_random.normal(
            self.config.stock_flow_mean, self.config.stock_flow_std
        ))

        # ------------------------------------------------------------------
        # 3. Apply climate fiscal impact
        # climate_fiscal_cost is negative (damages worsen the balance).
        # ------------------------------------------------------------------
        new_primary_balance += climate_fiscal_cost

        # ------------------------------------------------------------------
        # 4. Evolve debt using the budget constraint
        # ------------------------------------------------------------------
        new_debt = compute_next_debt(
            prev_debt=self.raw_state["debt"],
            growth=new_growth,
            interest_rate=new_rate,
            primary_balance=new_primary_balance,
            stock_flow_shock=stock_flow_shock,
        )

        # ------------------------------------------------------------------
        # 5. Update adaptation capital
        # ------------------------------------------------------------------
        new_adaptation = update_adaptation_capital(
            self.raw_state["adaptation_capital"],
            investment_action=is_adaptation,
            damage=climate_damage,
        )

        # ------------------------------------------------------------------
        # 6. Risk premium: interest–growth differential
        # ------------------------------------------------------------------
        new_risk_premium = new_rate - new_growth

        # ------------------------------------------------------------------
        # 7. Update state
        # ------------------------------------------------------------------
        prev_state = self.raw_state.copy()
        self.raw_state = {
            "growth":             new_growth,
            "debt":               new_debt,
            "primary_balance":    new_primary_balance,
            "interest_rate":      new_rate,
            "climate_damage":     climate_damage,
            "adaptation_capital": new_adaptation,
            "risk_premium":       new_risk_premium,
        }

        # ------------------------------------------------------------------
        # 8. Compute reward
        # ------------------------------------------------------------------
        reward, reward_components = compute_reward(
            self.raw_state, prev_state, action, self.config
        )

        # ------------------------------------------------------------------
        # 9. Check termination conditions
        # ------------------------------------------------------------------
        self.current_step += 1
        terminated = bool(new_debt > self.crisis_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        # Apply a large one-off crisis penalty to signal that debt crises are
        # catastrophic, not just a bad-reward ending.
        if terminated:
            reward -= 50.0
            logger.debug(
                "Debt crisis at step %d: debt=%.1f%% (threshold=%.1f%%)",
                self.current_step, new_debt, self.crisis_threshold,
            )

        # ------------------------------------------------------------------
        # 10. Build observation and info
        # ------------------------------------------------------------------
        observation = self._get_observation()

        info = {
            "raw_state":          self.raw_state.copy(),
            "prev_state":         prev_state,
            "action_name":        action_name,
            "pb_change":          pb_change,
            "reward_components":  reward_components,
            "climate_event":      climate_damage > 0,
            "climate_damage":     climate_damage,
            "stock_flow_shock":   stock_flow_shock,
            "step":               self.current_step,
            "terminated_by_crisis": terminated,
        }

        # Accumulate for post-episode analysis
        self.episode_history.append(info)

        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # render() — minimal human-readable output
    # ------------------------------------------------------------------

    def render(self) -> None:
        if not self.raw_state:
            print("Environment not yet reset.")
            return
        s = self.raw_state
        print(
            f"Step {self.current_step:3d} | "
            f"Growth={s['growth']:+6.2f}%  "
            f"Debt={s['debt']:6.1f}%GDP  "
            f"PB={s['primary_balance']:+6.2f}%  "
            f"Rate={s['interest_rate']:+6.2f}%  "
            f"Climate={s['climate_damage']:5.2f}%  "
            f"Adapt={s['adaptation_capital']:.3f}  "
            f"r-g={s['risk_premium']:+6.2f}"
        )

    # ------------------------------------------------------------------
    # close() — nothing to clean up for this environment
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Assemble the raw state array and normalise it for the agent."""
        s = self.raw_state
        raw_array = np.array([
            s["growth"],
            s["debt"],
            s["primary_balance"],
            s["interest_rate"],
            s["climate_damage"],
            s["adaptation_capital"],
            s["risk_premium"],
        ], dtype=np.float64)

        return normalise_state(raw_array, self.config.scaling)
