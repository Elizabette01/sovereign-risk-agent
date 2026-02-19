# Import neccessary libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SovereignRiskEnv(gym.Env):
    """
    A tiny toy environment for sovereign fiscal stress.

    State (observation): 4 numbers
      0) debt_ratio      (0.0 to 2.0)   e.g., 1.0 = 100% debt/GDP
      1) gdp_growth      (-0.10 to 0.10) per step
      2) inflation       (0.0 to 0.20)
      3) climate_shock   (0.0 to 1.0)

    Actions (discrete):
      0) do_nothing
      1) austerity
      2) stimulus
      3) adaptation_investment
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int | None = None):
        super().__init__()

        #Action government can take, in this case 4
        self.action_space = spaces.Discrete(4)

        #Observation space - debt/GDP, gdp growth rate, inflation and climate_shock
        low = np.array([0.0, -0.10, 0.0, 0.0], dtype=np.float32)
        high = np.array([2.0, 0.10, 0.20, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        #Generating controlled randomness for the climate and economic conditions
        self.rng = np.random.default_rng(seed)

        #initialize the current economic state and simulate 36 months
        self.state = None
        self.t = 0
        self.max_steps = 36  # e.g., 36 months

    #start a new government episode
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        debt_ratio = float(self.rng.uniform(0.5, 1.5))
        gdp_growth = float(self.rng.uniform(-0.02, 0.03))
        inflation = float(self.rng.uniform(0.02, 0.08))
        climate_shock = float(self.rng.uniform(0.0, 0.3))

        self.state = np.array([debt_ratio, gdp_growth, inflation, climate_shock], dtype=np.float32)
        self.t = 0

        info = {}
        return self.state, info

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."

        debt, growth, infl, shock = [float(x) for x in self.state]

        # --- exogenous climate shock dynamics ---
        # shock drifts a bit + random noise, clipped to [0, 1]
        shock = float(np.clip(shock + self.rng.normal(0.01, 0.03), 0.0, 1.0))

        # --- action effects (toy, not realistic yet) ---
        if action == 0:  # do nothing
            growth += -0.01 * shock
            infl += 0.005 * shock
            debt += 0.02 + 0.05 * shock

        elif action == 1:  # austerity
            growth += -0.01 - 0.01 * shock
            infl += -0.005
            debt += -0.02 + 0.01 * shock

        elif action == 2:  # stimulus
            growth += 0.01 - 0.005 * shock
            infl += 0.01 + 0.005 * shock
            debt += 0.03 + 0.02 * shock

        elif action == 3:  # adaptation investment
            # pay some debt now, but reduce shock impact over time
            growth += 0.002
            infl += 0.002
            debt += 0.015
            shock = float(np.clip(shock - 0.05, 0.0, 1.0))

        else:
            raise ValueError("Invalid action")

        # --- natural mean reversion / bounds ---
        growth = float(np.clip(growth, -0.10, 0.10))
        infl = float(np.clip(infl, 0.0, 0.20))
        debt = float(np.clip(debt, 0.0, 2.0))

        self.state = np.array([debt, growth, infl, shock], dtype=np.float32)

        # --- reward: encourage stability + lower debt + lower shock ---
        # (Toy reward; we'll refine later.)
        reward = (
            -2.0 * debt
            + 5.0 * growth
            - 2.0 * infl
            - 3.0 * shock
        )

        # termination conditions (toy)
        terminated = bool(debt >= 1.8)  # "default-like" threshold
        if terminated:
            reward -= 200.0

        self.t += 1
        truncated = bool(self.t >= self.max_steps)

        info = {"t": self.t}
        return self.state, float(reward), terminated, truncated, info

    def render(self):
        debt, growth, infl, shock = self.state
        print(f"t={self.t:02d} debt={debt:.2f} growth={growth:.3f} infl={infl:.3f} shock={shock:.2f}")
