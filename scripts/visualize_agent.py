"""
scripts/visualize_agent.py - Real-Time Visualization and GIF Generation for Trained Agents.

Enables interactive GUI viewing of the agent or renders an animated GIF demonstrating
agent performance across tasks in a continual learning sequence.
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.agent import DQNAgent
from src.envs import TASK_SEQUENCES, make_env


def visualize(
    checkpoint_path: str,
    sequence_name: str = "partial",
    render_mode: str = "rgb_array",
    save_gif: str = "results/plots/agent_demo.gif",
    max_steps_per_task: int = 250,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Initialize a dummy agent and load checkpoint
    agent = DQNAgent()
    agent.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path} (Strategy={agent.strategy.name})")

    tasks = TASK_SEQUENCES[sequence_name]
    all_frames = []

    for task_idx, task_cfg in enumerate(tasks):
        task_name = task_cfg.get("name", f"Task-{task_idx+1}")
        print(f"\n--- Demonstrating Task {task_idx + 1}/{len(tasks)}: {task_name} ---")
        env = make_env(task_cfg, seed=task_idx + 42, render_mode=render_mode)
        obs, _ = env.reset(seed=task_idx + 42)
        total_return = 0.0
        step_count = 0

        for step in range(max_steps_per_task):
            step_count += 1
            action = agent.act(obs, greedy=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            obs = next_obs

            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    # Convert to PIL Image and annotate
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    header = f"Task {task_idx+1}: {task_name} | Step: {step_count} | Return: {int(total_return)}"
                    # Draw text banner
                    draw.rectangle([(10, 10), (480, 36)], fill=(20, 20, 20))
                    draw.text((15, 14), header, fill=(255, 255, 255))
                    all_frames.append(img)
            elif render_mode == "human":
                env.render()

            if terminated or truncated:
                print(f"Task {task_idx+1} finished after {step_count} steps. Total return: {total_return:.1f}")
                break

        env.close()

    if render_mode == "rgb_array" and save_gif and all_frames:
        os.makedirs(os.path.dirname(save_gif) or ".", exist_ok=True)
        # Sample every 2nd frame to keep GIF compact
        sampled_frames = all_frames[::2]
        sampled_frames[0].save(
            save_gif,
            save_all=True,
            append_images=sampled_frames[1:],
            duration=35,
            loop=0,
            optimize=True,
        )
        print(f"\nSaved animated demonstration GIF to: {save_gif} ({len(sampled_frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Trained Continual RL Agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/benchmarks/car_sr_partial_seed42/agent.pt",
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="partial",
        choices=["similar", "partial", "dissimilar"],
        help="Task sequence curriculum",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gif",
        choices=["human", "gif"],
        help="'human' for interactive window, 'gif' to save animated GIF",
    )
    parser.add_argument(
        "--output-gif",
        type=str,
        default="results/plots/agent_demo.gif",
        help="File path to save demonstration GIF",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Maximum steps per task episode",
    )
    args = parser.parse_args()

    render_mode = "human" if args.mode == "human" else "rgb_array"
    visualize(
        checkpoint_path=args.checkpoint,
        sequence_name=args.sequence,
        render_mode=render_mode,
        save_gif=args.output_gif,
        max_steps_per_task=args.max_steps,
    )


if __name__ == "__main__":
    main()
