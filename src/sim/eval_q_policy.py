import pandas as pd

from src.envs.sovereign_env import SovereignRiskEnv
from src.agents.discretizer import discretize_state
from src.agents.q_table import QTable
from src.sim.train_q_learning import train


def evaluate(qt: QTable, n_episodes: int = 50, n_bins: int = 5, seed: int = 123) -> None:
    env = SovereignRiskEnv(seed=seed)

    summaries = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        s = discretize_state(obs, n_bins=n_bins)

        total_reward = 0.0
        terminated = False
        truncated = False

        for t in range(env.max_steps):
            a = qt.best_action(s)  # epsilon = 0 (pure greedy)
            next_obs, r, terminated, truncated, _ = env.step(a)
            s = discretize_state(next_obs, n_bins=n_bins)

            total_reward += r
            if terminated or truncated:
                break

        summaries.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": t + 1,
            "terminated": terminated,
            "truncated": truncated,
        })

    df = pd.DataFrame(summaries)
    print("\n=== Q-Learning Policy (Evaluation, epsilon=0) ===")
    print(df.describe(include="all"))

    print(f"\nDefault rate: {df['terminated'].mean():.2%}")
    print(f"Mean total reward: {df['total_reward'].mean():.2f}")


if __name__ == "__main__":
    qt = train(n_episodes=2000, n_bins=5)  # trains then evaluates
    evaluate(qt, n_episodes=50, n_bins=5)
