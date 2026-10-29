"""
evaluate.py - CLI to evaluate trained Continual RL checkpoints.

Usage examples:
    python evaluate.py --checkpoint results/runs/car_sr_similar_seed42/agent.pt --sequence similar
    python evaluate.py --checkpoint results/runs/car_sr_similar_seed42/agent.pt --sequence all
"""

import argparse
import sys
import numpy as np

from src.agent import DQNAgent
from src.envs import TASK_SEQUENCES
from src.evaluator import evaluate_all_tasks
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Continual RL Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent.pt checkpoint")
    parser.add_argument("--sequence", type=str, default="similar",
                        choices=["similar", "partial", "dissimilar", "all"],
                        help="Sequence to evaluate on")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task evaluation")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation random seed")
    return parser.parse_args()


def evaluate_on_sequence(agent: DQNAgent, seq_name: str, episodes: int, seed: int):
    tasks = TASK_SEQUENCES[seq_name]
    returns = evaluate_all_tasks(agent, tasks, episodes=episodes, seed=seed)
    print(f"\nEvaluation on Sequence: '{seq_name}' ({len(tasks)} tasks, {episodes} episodes each)")
    print("-" * 55)
    for cfg, ret in zip(tasks, returns):
        print(f"  {cfg['name']:<25s}: Return = {ret:.2f}")
    print("-" * 55)
    print(f"  Mean Sequence Return     : {np.mean(returns):.2f}")
    return returns


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading checkpoint from: {args.checkpoint}")
    agent = DQNAgent()
    agent.load_checkpoint(args.checkpoint)
    print(f"Loaded agent successfully. Internal step counter: {agent.steps}, Strategy: {agent.strategy.name}")

    if args.sequence == "all":
        for seq_name in ["similar", "partial", "dissimilar"]:
            evaluate_on_sequence(agent, seq_name, args.episodes, args.seed)
    else:
        evaluate_on_sequence(agent, args.sequence, args.episodes, args.seed)


if __name__ == "__main__":
    main()
