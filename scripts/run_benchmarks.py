"""
scripts/run_benchmarks.py - Multi-Seed Benchmark Matrix Orchestration.

Executes systematic continual learning benchmark experiments across:
- 5 Core Strategies: Naive, Replay, EWC, Parseval, CAR-SR (Proposed)
- 2 Ablations: CAR-SR without Projection, CAR-SR without Parseval
- 3 Conflict Sequences: Similar, Partial, Dissimilar
- 3 Random Seeds: 42, 123, 456
"""

import argparse
import concurrent.futures
import itertools
import os
import sys
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import DQNAgent
from src.trainer import run_continual_experiment
from src.utils import save_json, set_seed
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_single_benchmark(
    strategy: str,
    sequence: str,
    seed: int,
    steps_per_task: int,
    eval_episodes: int,
    output_base_dir: str,
    strategy_kwargs: Dict[str, Any] | None = None,
):
    run_id = f"{strategy}_{sequence}_seed{seed}"
    run_dir = os.path.join(output_base_dir, run_id)
    result_path = os.path.join(run_dir, "results.json")

    if os.path.exists(result_path):
        print(f"Skipping already completed run: {run_id}")
        return

    print(f"\n=======================================================")
    print(f"Starting Run: Strategy={strategy} | Sequence={sequence} | Seed={seed}")
    print(f"=======================================================")

    strat_kwargs = strategy_kwargs or {}
    strat_type = strategy

    # Handle ablation strategy names
    if strategy == "car_sr_no_proj":
        strat_type = "car_sr"
        strat_kwargs = {**strat_kwargs, "enable_projection": False, "enable_parseval": True}
    elif strategy == "car_sr_no_parseval":
        strat_type = "car_sr"
        strat_kwargs = {**strat_kwargs, "enable_projection": True, "enable_parseval": False}

    set_seed(seed)
    agent = DQNAgent(
        strategy=strat_type,
        strategy_kwargs=strat_kwargs,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=20_000,
        target_sync=250,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=5_000,
        hidden_dim=128,
        device="cpu",
    )

    t0 = time.time()
    R, metrics, history = run_continual_experiment(
        agent=agent,
        sequence_name=sequence,
        steps_per_task=steps_per_task,
        eval_episodes=eval_episodes,
        seed=seed,
        verbose=True,
    )
    total_time = time.time() - t0

    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "return_matrix.npy"), R)

    history["strategy"] = strategy  # Preserve ablation label
    history["total_elapsed_seconds"] = total_time

    save_json({
        "metrics": metrics,
        "return_matrix": R.tolist(),
        "history": history,
    }, result_path)

    ckpt_path = os.path.join(run_dir, "agent.pt")
    agent.save_checkpoint(ckpt_path)
    print(f"Completed {run_id} in {total_time:.1f}s. Saved to {run_dir}")


def _worker_wrapper(kwargs):
    run_single_benchmark(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run Continual RL Benchmark Suite")
    parser.add_argument("--strategies", nargs="+", default=["naive", "replay", "ewc", "parseval", "car_sr"],
                        help="Strategies to evaluate")
    parser.add_argument("--sequences", nargs="+", default=["similar", "partial", "dissimilar"],
                        help="Task sequences to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds to evaluate")
    parser.add_argument("--steps", type=int, default=10_000, help="Steps per task")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Eval episodes")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks", help="Output directory")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel worker processes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = list(itertools.product(args.strategies, args.sequences, args.seeds))
    print(f"Total benchmark experiments to execute: {len(experiments)} (using {args.workers} workers)")

    tasks = [
        {
            "strategy": strat,
            "sequence": seq,
            "seed": seed,
            "steps_per_task": args.steps,
            "eval_episodes": args.eval_episodes,
            "output_base_dir": args.output_dir,
        }
        for strat, seq, seed in experiments
    ]

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(_worker_wrapper, tasks))
    else:
        for t in tasks:
            _worker_wrapper(t)


if __name__ == "__main__":
    main()
