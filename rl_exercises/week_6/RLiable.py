"""
RLiable Analysis for Actor-Critic Baseline Experiments

Adopted from GitHub Copilot and https://github.com/google-research/rliable
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils


def main():
    """
    Analyze and visualize results from Actor-Critic experiments using RLiable.
    """

    # Load runs
    results_dir = Path("results")
    baseline_data = {}
    for env_dir in results_dir.iterdir():
        if not env_dir.is_dir():
            continue

        env_name = env_dir.name

        for baseline_dir in env_dir.iterdir():
            if not baseline_dir.is_dir():
                continue
            baseline = baseline_dir.name

            # Load all evaluation rewards for this baseline
            all_eval_rewards = []
            for seed_file in baseline_dir.glob("seed_*_eval_rewards.npy"):
                eval_rewards = np.load(seed_file)
                all_eval_rewards.append(eval_rewards)

            if all_eval_rewards:
                eval_rewards = np.array(all_eval_rewards)
                # Reshape to (n_seeds, 1 , n_evals) for RLiable compatibility
                eval_rewards = eval_rewards.reshape(
                    eval_rewards.shape[0], 1, eval_rewards.shape[1]
                )
                baseline_data[baseline] = eval_rewards

    if not baseline_data:
        print("No valid data found in results directory.")
        return

    ### Plot Runs over Time ###

    def iqm(scores):
        """
        Compute the Inverse of the Interquartile Mean (IQM) for given scores.
        """
        return np.array(
            [
                metrics.aggregate_iqm(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )

    iqm_scores, iqm_cis = rly.get_interval_estimates(baseline_data, iqm, reps=10000)

    # Create x-axis
    eval_interval = 10000
    n_evals = next(iter(baseline_data.values())).shape[-1]
    training_steps = np.arange(1, n_evals + 1) * eval_interval

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use RLiable's plot function
    plot_utils.plot_sample_efficiency_curve(
        training_steps / 1000,  # Convert to thousands
        iqm_scores,
        iqm_cis,
        algorithms=list(baseline_data.keys()),
        xlabel="Training Steps (thousands)",
        ylabel="IQM Evaluation Return",
        ax=ax,
    )

    # Add title and legend
    ax.set_title(
        f"Actor-Critic Baseline Comparison with 10 Seeds per Baseline - {env_name}"
    )
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = Path("rliable_plots")
    plt.savefig(save_path / f"{env_name}_sample_efficiency_rliable.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
