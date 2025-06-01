"""
Actor-Critic Hyperparameter Sweep Script

This script runs hyperparameter optimization for the ActorCriticAgent using HyperSweeper.
"""

from typing import Any, Dict

import gymnasium as gym
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def run_episodes(agent, env, training_steps, num_episodes: int = 10) -> float:
    """
    Run multiple episodes and return the average total reward.
    This function trains the agent for a short period and then evaluates it.

    Parameters
    ----------
    agent : ActorCriticAgent
        The agent to train and evaluate
    env : gym.Env
        The environment to run episodes in
    num_episodes : int
        Number of episodes to run for evaluation

    Returns
    -------
    float
        Average total reward across all episodes
    """

    # Train the agent
    agent.train(
        total_steps=training_steps,
        eval_interval=training_steps,  # Only evaluate at the end
        eval_episodes=3,  # Minimal evaluation during training
    )

    mean, std = agent.evaluate(env, num_episodes=num_episodes)
    print(f"Mean reward over {num_episodes} episodes: {mean:.2f} ± {std:.2f}")
    return mean


@hydra.main(
    config_path="../configs/agent/",
    config_name="actor_critic_sweep",
    version_base="1.1",
)
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main function for hyperparameter sweep.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing all parameters

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the performance metrics
    """
    # Hydra-instantiate the env
    env = gym.make(cfg.env.name)

    # Instantiate the agent using Hydra
    agent = instantiate(cfg.agent, env=env)

    # Run episodes and get performance
    mean_reward = run_episodes(agent, env, cfg.train.total_steps)

    env.close()

    # Return as dictionary for HyperSweeper compatibility
    return -mean_reward


if __name__ == "__main__":
    main()
