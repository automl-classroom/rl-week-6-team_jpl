import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from rliable import metrics, plot_utils


def load_scores(env):
    mat = defaultdict(list)  # baseline → [run1, run2, …]
    steps = None
    for path in glob.glob(f"logs/{env}_*.npz"):
        data = np.load(path)
        baseline = path.split("_")[1]  # env_baseline_seed.npz
        mat[baseline].append(data["returns"])  # nur die Mittelwerte
        steps = data["steps"]  # identisch für alle
    # in 2-D-Arrays gießen
    scores = {b: np.vstack(runs) for b, runs in mat.items()}
    return scores, steps


def plot_env(env):
    scores, x = load_scores(env)
    plot_utils.plot_sample_efficiency(
        score_dict=scores,
        aggregate_fn=metrics.aggregate_mean,  # Ø-Return
        algorithms=list(scores.keys()),
        x_axis=x,
        xlabel="Steps",
        title=env,
    )
    plt.ylabel("Average return")
    plt.show()


plot_env("CartPole-v1")
plot_env("LunarLander-v3")
