import pandas as pd
import matplotlib.pyplot as plt


def plot_policy(path: str, label: str):
    df = pd.read_csv(path)

    # Plot first 10 episodes for visual clarity
    for ep in df["episode"].unique()[:10]:
        df_ep = df[df["episode"] == ep]
        plt.plot(df_ep["t"], df_ep["debt_ratio"], alpha=0.6)

    plt.title(f"Debt Trajectories ({label})")
    plt.xlabel("Time")
    plt.ylabel("Debt Ratio")
    plt.show()


def main():
    plot_policy("steps_random.csv", "Random Policy")
    plot_policy("steps_rule_based.csv", "Rule-Based Policy")


if __name__ == "__main__":
    main()
