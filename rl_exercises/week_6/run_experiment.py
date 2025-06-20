"""Adopted from GitHub Copilot"""

from typing import Dict, List, Tuple

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import numpy as np
from rl_exercises.week_6.actor_critic import ActorCriticAgent


def run_experiment(
    env_name: str, baseline: str, seed: int, config: Dict
) -> Tuple[str, str, int, List[float]]:
    """
    Run a single experiment and return evaluation history.

    Parameters
    ----------
    env_name : str
        Environment name
    baseline : str
        Baseline type ('none', 'avg', 'value', 'gae')
    seed : int
        Random seed
    config : Dict
        Training configuration

    Returns
    -------
    Tuple[str, str, int, List[float]]
        (env_name, baseline, seed, eval_rewards)
    """
    print(f"Running {env_name} with {baseline} baseline, seed {seed}")

    # Create environment and agent
    env = gym.make(env_name)
    agent = ActorCriticAgent(
        env=env, baseline_type=baseline, seed=seed, **config["agent"]
    )

    # Train and get evaluation history
    history = agent.train(
        total_steps=config["train"]["total_steps"],
        eval_interval=config["train"]["eval_interval"],
        eval_episodes=config["train"]["eval_episodes"],
    )

    env.close()

    # Return env_name, baseline, seed, and evaluation rewards
    return env_name, baseline, seed, history


def run_all_experiments(config: Dict, max_workers: int = None) -> None:
    """
    Run experiments for all environments, baselines, and seeds in parallel.

    Parameters
    ----------
    config : Dict
        Complete experiment configuration
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count.
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    results_dir = Path(config["experiment"]["results_dir"])

    # Create all experiment tasks
    tasks = []
    for env_name in config["environments"]:
        for baseline in config["baselines"]:
            for seed in config["experiment"]["seeds"]:
                tasks.append((env_name, baseline, seed, config))

    print(f"Running {len(tasks)} experiments with {max_workers} workers...")

    # Dictionary to collect results by (env_name, baseline)
    results_dict = {}

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_experiment, *task): task for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                env_name, baseline, seed, eval_rewards = future.result()

                # Group results by (env_name, baseline)
                key = (env_name, baseline)
                if key not in results_dict:
                    results_dict[key] = []

                results_dict[key].append((seed, eval_rewards))

                print(f"Completed: {env_name}/{baseline}/seed_{seed}")

            except Exception as e:
                print(f"Error in task {task}: {e}")

    # Save results for each (env_name, baseline) combination
    for (env_name, baseline), seed_results in results_dict.items():
        print(f"Saving results for {env_name}/{baseline}")

        # Create directory
        save_dir = results_dir / env_name / baseline
        save_dir.mkdir(parents=True, exist_ok=True)

        all_eval_rewards = []

        # Save individual seed results and collect for aggregation
        for seed, eval_rewards in seed_results:
            np.save(save_dir / f"seed_{seed}_eval_rewards.npy", eval_rewards)
            all_eval_rewards.append(eval_rewards)

        # Save aggregated results (not necessary, but generated by AI)
        if all_eval_rewards:
            # Convert to array: (n_seeds, n_evaluations)
            eval_array = np.array(all_eval_rewards)
            np.save(save_dir / "all_seeds_eval_rewards.npy", eval_array)

            # Save summary statistics
            final_rewards = eval_array[:, -1] if eval_array.size > 0 else []
            summary = {
                "mean_final_reward": float(np.mean(final_rewards)),
                "std_final_reward": float(np.std(final_rewards)),
                "num_seeds": len(final_rewards),
                "completed_seeds": [seed for seed, _ in seed_results],
            }

            with open(save_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)


# Example configuration
DEFAULT_CONFIG = {
    "experiment": {"results_dir": "results", "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
    "environments": ["LunarLander-v3"],
    "baselines": ["none", "avg", "value", "gae"],
    "agent": {
        "lr_actor": 5e-3,
        "lr_critic": 1e-2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "hidden_size": 128,
        "baseline_decay": 0.9,
    },
    "train": {"total_steps": 200000, "eval_interval": 10000, "eval_episodes": 3},
}


if __name__ == "__main__":
    run_all_experiments(DEFAULT_CONFIG)
