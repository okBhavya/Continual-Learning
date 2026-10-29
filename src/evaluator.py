"""
src/evaluator.py - Evaluation Routines for Continual RL.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import gymnasium as gym

from src.envs import make_env


def evaluate_policy(
    agent: Any,
    env_cfg: Dict[str, Any],
    episodes: int = 20,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    """
    Evaluates the agent's greedy policy (deterministic / greedy Q-argmax)
    on a target environment configuration over a specified number of episodes.

    Returns
    -------
    (mean_return, std_return, mean_length)
    """
    env = make_env(env_cfg)
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            a = agent.act(s, greedy=True)
            s, r, term, trunc, _ = env.step(a)
            ep_return += float(r)
            ep_len += 1
            done = term or trunc

        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def evaluate_all_tasks(
    agent: Any,
    task_configs: List[Dict[str, Any]],
    episodes: int = 20,
    seed: int = 12345,
) -> List[float]:
    """Evaluates the greedy policy across all tasks in a sequence."""
    mean_returns = []
    for cfg in task_configs:
        mean_ret, _, _ = evaluate_policy(agent, cfg, episodes=episodes, seed=seed)
        mean_returns.append(mean_ret)
    return mean_returns
