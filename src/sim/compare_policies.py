import pandas as pd


def load_and_summarize(path: str, name: str) -> dict:
    df = pd.read_csv(path)

    out = {
        "policy": name,
        "episodes": len(df),
        "default_rate": df["terminated"].mean(),
        "mean_total_reward": df["total_reward"].mean(),
        "median_total_reward": df["total_reward"].median(),
        "mean_steps": df["steps"].mean(),
    }
    return out


def main() -> None:
    random_summary = load_and_summarize("episode_summary_random.csv", "random")
    rule_summary = load_and_summarize("episode_summary_rule_based.csv", "rule_based")

    df = pd.DataFrame([random_summary, rule_summary])

    # nicer formatting
    df["default_rate"] = (df["default_rate"] * 100).round(2).astype(str) + "%"
    df["mean_total_reward"] = df["mean_total_reward"].round(2)
    df["median_total_reward"] = df["median_total_reward"].round(2)
    df["mean_steps"] = df["mean_steps"].round(2)

    print("\n=== Policy Scorecard ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
