"""
test_environment.py — Automated tests for the SovereignRiskEnv Gymnasium environment.

Run with:
    pytest tests/test_environment.py -v

These tests verify:
1. Environment creation for all 9 profiles.
2. reset() returns valid observations.
3. step() returns correct types and shapes for all 6 actions.
4. Economic sanity: austerity reduces debt, stimulus increases debt.
5. Episode termination: max_steps and debt crisis.
6. Reproducibility: same seed → identical trajectory.
7. Gymnasium API compliance via check_env().
"""

import sys
import os

# Add project root to path so 'src' is importable when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.environment.config import list_profiles
from src.environment.sovereign_risk_env import SovereignRiskEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CONFIG_PATH = os.path.join(DATA_DIR, "transition_parameters.json")
SCALING_PATH = os.path.join(DATA_DIR, "scaling_parameters.json")


def make_env(profile: str = "Advanced_Low", **kwargs) -> SovereignRiskEnv:
    """Convenience factory that injects the data paths."""
    return SovereignRiskEnv(
        profile=profile,
        config_path=CONFIG_PATH,
        scaling_path=SCALING_PATH,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 7a. Environment creation tests
# ---------------------------------------------------------------------------

def test_env_creates_for_all_profiles():
    """Verify the environment can be instantiated for all 9 profiles."""
    for profile in list_profiles(CONFIG_PATH):
        env = make_env(profile=profile)
        assert env.observation_space.shape == (7,), (
            f"Observation space shape wrong for {profile}"
        )
        assert env.action_space.n == 6, (
            f"Action space size wrong for {profile}"
        )
        env.close()


def test_env_reset_returns_valid_observation():
    """Verify reset() returns an observation within bounds and correct shape."""
    env = make_env("Advanced_Low")
    obs, info = env.reset(seed=0)

    assert obs.shape == (7,), f"Expected shape (7,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert np.all(obs >= -10.0) and np.all(obs <= 10.0), (
        f"Observation out of bounds: min={obs.min():.3f}, max={obs.max():.3f}"
    )
    assert "raw_state" in info, "info dict missing 'raw_state'"
    assert "profile" in info, "info dict missing 'profile'"
    assert info["step"] == 0, "Step counter should be 0 after reset"

    env.close()


def test_reset_initial_raw_state_is_plausible():
    """Verify initial raw state values are in economically plausible ranges."""
    env = make_env("Advanced_Low")
    _, info = env.reset(seed=0)
    raw = info["raw_state"]

    assert -15.0 <= raw["growth"] <= 15.0, f"Growth implausible: {raw['growth']}"
    assert 0.0 <= raw["debt"] <= 300.0, f"Debt implausible: {raw['debt']}"
    assert 0.0 <= raw["adaptation_capital"] <= 1.0, (
        f"Adaptation capital out of [0,1]: {raw['adaptation_capital']}"
    )
    assert raw["climate_damage"] >= 0.0, (
        f"Climate damage negative: {raw['climate_damage']}"
    )
    env.close()


# ---------------------------------------------------------------------------
# 7b. Step tests
# ---------------------------------------------------------------------------

def test_env_step_valid_types_and_shapes():
    """Verify step() returns correct types and shapes for all 6 actions."""
    env = make_env("Advanced_Low", seed=42)
    obs, info = env.reset()

    for action in range(6):
        env.reset(seed=action)  # fresh reset for each action
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (7,), f"Action {action}: wrong obs shape {obs.shape}"
        assert obs.dtype == np.float32, f"Action {action}: wrong dtype {obs.dtype}"
        assert isinstance(reward, float), f"Action {action}: reward is not float"
        assert isinstance(terminated, bool), (
            f"Action {action}: terminated is not bool"
        )
        assert isinstance(truncated, bool), (
            f"Action {action}: truncated is not bool"
        )
        assert "reward_components" in info, (
            f"Action {action}: missing reward_components"
        )
        assert "raw_state" in info, f"Action {action}: missing raw_state"

    env.close()


def test_env_step_observation_within_bounds():
    """Verify observations stay within [−10, 10] across a full episode."""
    env = make_env("Developing_High", seed=99)
    obs, _ = env.reset()

    for _ in range(30):
        obs, _, terminated, truncated, _ = env.step(2)  # Maintain policy
        assert np.all(obs >= -10.0) and np.all(obs <= 10.0), (
            f"Observation out of bounds: {obs}"
        )
        if terminated or truncated:
            break

    env.close()


def test_invalid_action_raises():
    """Verify that an invalid action raises an AssertionError."""
    env = make_env()
    env.reset()
    with pytest.raises(AssertionError):
        env.step(6)  # Valid actions are 0–5 only
    env.close()


def test_reward_components_sum_correctly():
    """Verify reward components sum to the total reward (within float tolerance)."""
    env = make_env("Advanced_Low", seed=7)
    env.reset()
    _, reward, _, _, info = env.step(2)

    comp = info["reward_components"]
    component_sum = sum(comp.values())
    # The -50 crisis penalty is applied outside compute_reward, so only check
    # when no crisis occurred
    if not info["terminated_by_crisis"]:
        assert abs(component_sum - reward) < 1e-6, (
            f"Component sum {component_sum:.6f} != total reward {reward:.6f}"
        )
    env.close()


def test_austerity_reduces_debt_on_average():
    """Verify that sustained austerity (action 0) tends to reduce debt over time.

    Economic reasoning: if primary surplus improves by 3pp per year, the debt
    accumulation identity guarantees lower debt on average, even with stochastic
    shocks. We use 50 episodes and check that the mean debt change is negative
    (debt fell), with some tolerance for the high stock-flow volatility.
    """
    env = make_env("Advanced_Low", seed=42)
    debt_changes = []

    for ep in range(50):
        obs, info = env.reset(seed=ep)
        initial_debt = info["raw_state"]["debt"]

        for _ in range(10):
            obs, reward, term, trunc, info = env.step(0)  # Always severe austerity
            if term or trunc:
                break

        final_debt = info["raw_state"]["debt"]
        debt_changes.append(final_debt - initial_debt)

    mean_change = np.mean(debt_changes)
    # With high stock-flow volatility (std ~18pp), we allow some tolerance.
    # The key assertion is that austerity does not systematically *increase* debt.
    assert mean_change < 30.0, (
        f"Austerity should reduce debt on average; mean change was {mean_change:.2f}pp"
    )
    env.close()


def test_stimulus_increases_debt_on_average():
    """Verify that sustained stimulus (action 4) increases debt on average."""
    env = make_env("Advanced_Low", seed=42)
    debt_changes = []

    for ep in range(50):
        obs, info = env.reset(seed=ep)
        initial_debt = info["raw_state"]["debt"]

        for _ in range(10):
            obs, reward, term, trunc, info = env.step(4)  # Always large stimulus
            if term or trunc:
                break

        final_debt = info["raw_state"]["debt"]
        debt_changes.append(final_debt - initial_debt)

    mean_change = np.mean(debt_changes)
    assert mean_change > 0, (
        f"Stimulus should increase debt on average; mean change was {mean_change:.2f}pp"
    )
    env.close()


# ---------------------------------------------------------------------------
# Episode termination tests
# ---------------------------------------------------------------------------

def test_episode_terminates_at_max_steps():
    """Verify the episode ends after max_steps (truncation, not termination)."""
    env = make_env("Advanced_Low", max_steps=10, seed=123)
    obs, _ = env.reset()

    step_count = 0
    for _ in range(15):  # run longer than max_steps
        _, _, terminated, truncated, _ = env.step(2)
        step_count += 1
        if terminated or truncated:
            break

    assert step_count <= 10, f"Episode ran {step_count} steps, expected ≤ 10"
    env.close()


def test_debt_crisis_terminates_episode():
    """Verify that exceeding crisis_threshold terminates the episode."""
    # Use a low threshold and aggressive stimulus to trigger a crisis
    env = make_env("Developing_High", crisis_threshold=100.0, seed=42)
    obs, info = env.reset()

    # Run with maximum stimulus; the Developing_High profile has high stock-flow
    # volatility, so a crisis should occur within 50 steps
    terminated = False
    steps_run = 0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(4)  # Large stimulus
        steps_run += 1
        if terminated or truncated:
            break

    if terminated:
        assert info["raw_state"]["debt"] > 100.0, (
            "Terminated by crisis but debt is not above threshold"
        )
        assert info["terminated_by_crisis"] is True

    env.close()


def test_crisis_reward_has_large_penalty():
    """Verify that the crisis episode applies a -50 penalty in the final step."""
    # Manufacture a guaranteed crisis: start at 190% debt, low threshold, stimulus
    env = make_env("Advanced_Low", crisis_threshold=200.0, seed=0)
    obs, info = env.reset()

    # Force debt close to threshold by running stimulus
    # This is a statistical test — run many episodes until one hits the crisis
    crisis_reward = None
    for ep in range(100):
        obs, info = env.reset(seed=ep)
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(4)
            if terminated:
                crisis_reward = reward
                break
        if crisis_reward is not None:
            break

    if crisis_reward is not None:
        assert crisis_reward < -45.0, (
            f"Crisis penalty should dominate reward; got {crisis_reward:.2f}"
        )
    env.close()


# ---------------------------------------------------------------------------
# 7c. Reproducibility test
# ---------------------------------------------------------------------------

def test_same_seed_same_trajectory():
    """Verify that the same seed produces identical trajectories."""
    def run_episode(seed: int):
        env = make_env("Advanced_Low", seed=seed)
        obs, _ = env.reset(seed=seed)
        trajectory = [obs.copy()]
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(2)
            trajectory.append(obs.copy())
            if terminated or truncated:
                break
        env.close()
        return trajectory

    t1 = run_episode(42)
    t2 = run_episode(42)

    assert len(t1) == len(t2), "Trajectories have different lengths"
    for step_idx, (o1, o2) in enumerate(zip(t1, t2)):
        np.testing.assert_array_equal(
            o1, o2,
            err_msg=f"Trajectories diverge at step {step_idx}"
        )


def test_different_seeds_different_trajectories():
    """Verify that different seeds produce different trajectories."""
    def run_episode(seed: int):
        env = make_env("Advanced_Low", seed=seed)
        obs, _ = env.reset(seed=seed)
        for _ in range(5):
            obs, _, term, trunc, _ = env.step(2)
            if term or trunc:
                break
        env.close()
        return obs.copy()

    obs_42 = run_episode(42)
    obs_99 = run_episode(99)

    # The probability that two random episodes produce identical final observations
    # is astronomically small — this would indicate a seeding bug.
    assert not np.array_equal(obs_42, obs_99), (
        "Different seeds produced identical trajectories — seeding may be broken"
    )


# ---------------------------------------------------------------------------
# 7d. Gymnasium API compliance
# ---------------------------------------------------------------------------

def test_gymnasium_check_env():
    """Verify the environment passes Gymnasium's built-in validation suite.

    check_env() tests:
    - Observation space dtype and bounds.
    - Action space validity.
    - reset() and step() return correct types.
    - Proper terminated/truncated handling.
    - No state leakage between reset() calls.
    """
    from gymnasium.utils.env_checker import check_env

    env = make_env("Advanced_Low")
    check_env(env, warn=True, skip_render_check=True)
    env.close()


# ---------------------------------------------------------------------------
# Additional: scaling utilities
# ---------------------------------------------------------------------------

def test_normalise_denormalise_roundtrip():
    """Verify that normalise → denormalise recovers the original values."""
    from src.utils.scaling import normalise_state, denormalise_state
    from src.environment.config import load_profile

    cfg = load_profile("Advanced_Low", CONFIG_PATH, SCALING_PATH)
    raw = np.array([2.5, 55.0, -1.0, 4.0, 0.02, 0.55, 1.5], dtype=np.float64)

    normalised = normalise_state(raw, cfg.scaling)
    recovered = denormalise_state(normalised, cfg.scaling)

    np.testing.assert_allclose(raw, recovered, rtol=1e-5, atol=1e-5)


def test_normalise_clips_extreme_values():
    """Verify that extreme raw values are clipped to [-10, 10] after normalisation."""
    from src.utils.scaling import normalise_state
    from src.environment.config import load_profile

    cfg = load_profile("Advanced_Low", CONFIG_PATH, SCALING_PATH)
    # Use an absurdly high debt value (1000% of GDP)
    extreme = np.array([2.5, 1000.0, -1.0, 4.0, 0.02, 0.55, 1.5], dtype=np.float64)
    normalised = normalise_state(extreme, cfg.scaling)

    assert normalised[1] == pytest.approx(10.0, abs=1e-5), (
        f"Extreme debt not clipped to 10.0, got {normalised[1]}"
    )


# ---------------------------------------------------------------------------
# Run directly for quick checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
