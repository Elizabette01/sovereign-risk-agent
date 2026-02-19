import numpy as np
import pandas as pd

from src.envs.sovereign_env import SovereignRiskEnv
from src.agents.discretizer import discretize_state
from src.agents.q_table import QTable


def epsilon_greedy(qt: QTable, s, epsilon: float) -> int:
    # With probability epsilon: explore (random action)
    if np.random.rand() < epsilon:
        return np.random.randint(qt.n_actions)
    # Otherwise: exploit (best known action)
    return qt.best_action(s)


def train(
    n_episodes: int = 2000,
    n_bins: int = 5,
    alpha: float = 0.10,     # learning rate
    gamma: float = 0.95,     # discount factor
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 42,
):
    

    env = SovereignRiskEnv(seed=seed)
    qt = QTable.create(n_bins=n_bins, n_actions=env.action_space.n)

    print(f"Starting Q-learning: episodes={n_episodes}, bins={n_bins}, actions={env.action_space.n}")
    
    episode_log = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        s = discretize_state(obs, n_bins=n_bins)

        # linearly decay epsilon over training
        frac = ep / max(1, n_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        total_reward = 0.0
        terminated = False
        truncated = False

        for t in range(env.max_steps):
            a = epsilon_greedy(qt, s, epsilon)

            next_obs, r, terminated, truncated, _ = env.step(a)
            s2 = discretize_state(next_obs, n_bins=n_bins)

            # Q-learning update:
            # Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
            target = r + gamma * qt.best_value(s2) * (0.0 if terminated else 1.0)
            new_q = qt.get(s, a) + alpha * (target - qt.get(s, a))
            qt.set(s, a, new_q)

            total_reward += r
            s = s2

            if terminated or truncated:
                break

        episode_log.append({
            "episode": ep,
            "epsilon": epsilon,
            "total_reward": total_reward,
            "steps": t + 1,
            "terminated": terminated,
            "truncated": truncated,
        })

        # small progress print
        if (ep + 1) % 20 == 0:
            df_tmp = pd.DataFrame(episode_log[-200:])
            print(
                f"ep {ep+1}/{n_episodes} | "
                f"avg_reward(last20)={df_tmp['total_reward'].mean():.2f} | "
                f"default_rate(last20)={df_tmp['terminated'].mean():.2%} | "
                f"epsilon={epsilon:.2f}"
            )

    df = pd.DataFrame(episode_log)
    df.to_csv("q_learning_training_log.csv", index=False)
    print("\nSaved: q_learning_training_log.csv")

    return qt


if __name__ == "__main__":
    train()
