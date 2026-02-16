from envs.sovereign_env import SovereignRiskEnv


def main() -> None:
    env = SovereignRiskEnv(seed=42)
    obs, info = env.reset()

    print("✅ Environment reset")
    print("Initial state:", obs)

    for i in range(10):
        action = env.action_space.sample()  # random policy for now
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step={i} action={action} reward={reward:.2f} state={obs}")
        if terminated or truncated:
            print("Episode ended.", {"terminated": terminated, "truncated": truncated})
            break


if __name__ == "__main__":
    main()

