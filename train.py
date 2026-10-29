"""
train.py - Unified Training CLI for Continual Reinforcement Learning.

Usage examples:
    python train.py --config configs/smoke.yaml
    python train.py --strategy car_sr --sequence similar --steps 10000 --seed 42
"""

import argparse
import os
import sys
from typing import Dict, Any
import numpy as np

from src.agent import DQNAgent
from src.trainer import run_continual_experiment
from src.utils import load_config, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Reinforcement Learning Training CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["naive", "replay", "ewc", "parseval", "car_sr"],
                        help="Continual learning strategy")
    parser.add_argument("--sequence", type=str, default=None,
                        choices=["similar", "partial", "dissimilar"],
                        help="Task conflict sequence")
    parser.add_argument("--steps", type=int, default=None, help="Environment steps per task")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes per task")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save logs and checkpoints")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save final agent checkpoint")
    return parser.parse_args()


def merge_config(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "strategy": "car_sr",
        "sequence": "similar",
        "steps_per_task": 10_000,
        "eval_episodes": 20,
        "seed": 42,
        "output_dir": "results/runs",
        "save_checkpoint": True,
        "agent": {
            "lr": 1e-3,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 20_000,
            "target_sync": 250,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "eps_decay_steps": 5_000,
            "hidden_dim": 128,
            "device": "cpu",
        },
        "strategy_kwargs": {},
    }

    if args.config is not None:
        file_cfg = load_config(args.config)
        for k, v in file_cfg.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    if args.strategy:
        cfg["strategy"] = args.strategy
    if args.sequence:
        cfg["sequence"] = args.sequence
    if args.steps:
        cfg["steps_per_task"] = args.steps
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.eval_episodes:
        cfg["eval_episodes"] = args.eval_episodes
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.save_checkpoint:
        cfg["save_checkpoint"] = True

    return cfg


def main():
    args = parse_args()
    cfg = merge_config(args)

    set_seed(cfg["seed"])
    print(f"=== Continual RL Training ===")
    print(f"Strategy:        {cfg['strategy']}")
    print(f"Task Sequence:   {cfg['sequence']}")
    print(f"Steps Per Task:  {cfg['steps_per_task']}")
    print(f"Seed:            {cfg['seed']}")
    print(f"Output Dir:      {cfg['output_dir']}")
    print("==============================")

    agent = DQNAgent(
        strategy=cfg["strategy"],
        strategy_kwargs=cfg.get("strategy_kwargs", {}),
        **cfg.get("agent", {})
    )

    R, metrics, history = run_continual_experiment(
        agent=agent,
        sequence_name=cfg["sequence"],
        steps_per_task=cfg["steps_per_task"],
        eval_episodes=cfg["eval_episodes"],
        seed=cfg["seed"],
        verbose=True,
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    run_id = f"{cfg['strategy']}_{cfg['sequence']}_seed{cfg['seed']}"
    run_dir = os.path.join(cfg["output_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save outputs
    np.save(os.path.join(run_dir, "return_matrix.npy"), R)
    save_json({
        "metrics": metrics,
        "return_matrix": R.tolist(),
        "history": history,
        "config": cfg,
    }, os.path.join(run_dir, "results.json"))

    if cfg.get("save_checkpoint", False):
        ckpt_path = os.path.join(run_dir, "agent.pt")
        agent.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("\n=== Final Evaluated Continual Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<20s}: {v:.2f}")
    print("=========================================\n")


if __name__ == "__main__":
    main()
