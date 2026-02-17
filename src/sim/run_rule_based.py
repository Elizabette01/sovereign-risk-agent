from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from src.envs.sovereign_env import SovereignRiskEnv


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool


def choose_action(obs) -> int:
    """
    Simple hand-coded policy (baseline):
    - If climate shock is high: invest in adaptation (3)
    - Else if debt is high: austerity (1)
    - Else: stimulus (2) to support growth
    """
    debt, growth, infl, shock = obs.tolist()

    if shock >= 0.6:
        return 3  # adaptation
    if debt >= 1.3:
        return 1  # austerity
    return 2      # stimulus


def run_one_episode(env: SovereignRiskEnv, max_steps: int = 36) -> tuple[EpisodeResult, pd.DataFrame]:
    obs, _ = env.reset()
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0

    for t in range(max_steps):
        action = choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        debt, growth, infl, shock = next_obs.tolist()

        rows.append({
            "t": t,
            "action": action,
            "reward": reward,
            "debt_ratio": debt,
            "gdp_growth": growth,
            "inflation": infl,
            "climate_shock": shock,
            "terminated": terminated,
            "truncated": truncated,
        })

        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    ep = EpisodeResult(
        total_reward=float(total_reward),
        steps=len(rows),
        terminated=bool(rows[-1]["terminated"]) if rows else False,
        truncated=bool(rows[-1]["truncated"]) if rows else False,
    )

    return ep, pd.DataFrame(rows)


def run_many(n_episodes: int = 50, seed: int = 42) -> None:
    env = SovereignRiskEnv(seed=seed)

    episode_summaries = []
    all_steps = []

    for i in range(n_episodes):
        ep, df_steps = run_one_episode(env)
        episode_summaries.append({
            "episode": i,
            "total_reward": ep.total_reward,
            "steps": ep.steps,
            "terminated": ep.terminated,
            "truncated": ep.truncated,
        })
        df_steps["episode"] = i
        all_steps.append(df_steps)

    df_eps = pd.DataFrame(episode_summaries)
    df_all = pd.concat(all_steps, ignore_index=True)

    print("\n=== Episode Summary (Rule-Based Policy) ===")
    print(df_eps.describe(include="all"))

    default_rate = df_eps["terminated"].mean()
    print(f"\nApprox 'default' rate (terminated=True): {default_rate:.2%}")

    df_eps.to_csv("episode_summary_rule_based.csv", index=False)
    df_all.to_csv("steps_rule_based.csv", index=False)
    print("\nSaved: episode_summary_rule_based.csv and steps_rule_based.csv")


if __name__ == "__main__":
    run_many(n_episodes=50, seed=42)
