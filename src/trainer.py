"""
src/trainer.py - Sequential Continual Training Pipeline.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np

from src.envs import TASK_SEQUENCES, make_env
from src.evaluator import evaluate_all_tasks
from src.metrics import compute_crl_metrics
from src.utils import set_seed


def train_single_task(
    agent: Any,
    env_cfg: Dict[str, Any],
    task_idx: int,
    steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Trains the agent on a single sequential task stage for `steps` environment steps."""
    env = make_env(env_cfg, seed=seed)
    agent.start_task(task_idx)

    s, _ = env.reset(seed=seed)
    episode_returns = []
    current_ep_return = 0.0
    loss_logs = []

    start_time = time.time()
    for step in range(steps):
        a = agent.act(s)
        s2, r, term, trunc, _ = env.step(a)
        current_ep_return += float(r)

        # In CartPole, only pole falling (term) is true termination; time truncation is not
        info = agent.observe(s, a, float(r), s2, float(term))
        if info is not None:
            loss_logs.append(info)

        s = s2
        if term or trunc:
            episode_returns.append(current_ep_return)
            current_ep_return = 0.0
            s, _ = env.reset()

    env.close()
    agent.end_task(task_idx)
    elapsed = time.time() - start_time

    return {
        "task_name": env_cfg["name"],
        "steps": steps,
        "elapsed_seconds": elapsed,
        "mean_training_ep_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "num_episodes": len(episode_returns),
    }


def run_continual_experiment(
    agent: Any,
    sequence_name: str,
    steps_per_task: int = 10_000,
    eval_episodes: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Executes a complete Continual Reinforcement Learning trial:
    1. Untrained agent evaluated on all tasks -> R[0, :]
    2. Sequentially trains on each task k in sequence:
       - train on Task k
       - evaluate greedy policy on ALL tasks in sequence -> R[k + 1, :]
    3. Computes formal CRL metrics (AFR, Forgetting, Transfer, Plasticity).
    """
    set_seed(seed)
    if sequence_name not in TASK_SEQUENCES:
        raise ValueError(f"Unknown sequence: {sequence_name}. Options: {list(TASK_SEQUENCES.keys())}")

    tasks = TASK_SEQUENCES[sequence_name]
    num_tasks = len(tasks)

    # Initialize return matrix R: shape (num_tasks + 1, num_tasks)
    R = np.zeros((num_tasks + 1, num_tasks), dtype=np.float32)

    # Untrained baseline
    R[0] = evaluate_all_tasks(agent, tasks, episodes=eval_episodes, seed=seed)
    if verbose:
        print(f"[{agent.strategy.name.upper()} | {sequence_name} | Seed {seed}] "
              f"Untrained Baseline Returns: {np.round(R[0], 1)}")

    stage_timings = []
    # Sequential training stages
    for k, task_cfg in enumerate(tasks):
        stage_seed = seed + k * 1000
        stage_result = train_single_task(agent, task_cfg, task_idx=k, steps=steps_per_task, seed=stage_seed)
        stage_timings.append(stage_result["elapsed_seconds"])

        # Post-stage evaluation on all tasks
        R[k + 1] = evaluate_all_tasks(agent, tasks, episodes=eval_episodes, seed=seed + 999)
        if verbose:
            print(f"[{agent.strategy.name.upper()} | {sequence_name} | Seed {seed}] "
                  f"After Task {k + 1} ({task_cfg['name']}): "
                  f"Returns on all tasks: {np.round(R[k + 1], 1)} "
                  f"({stage_result['elapsed_seconds']:.1f}s)")

    metrics = compute_crl_metrics(R)
    history = {
        "sequence_name": sequence_name,
        "strategy": agent.strategy.name,
        "seed": seed,
        "steps_per_task": steps_per_task,
        "eval_episodes": eval_episodes,
        "stage_timings": stage_timings,
        "strategy_diagnostics": agent.strategy.get_diagnostics(),
    }

    return R, metrics, history
