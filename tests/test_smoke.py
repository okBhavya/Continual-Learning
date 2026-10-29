"""
tests/test_smoke.py - Integration Smoke Test for Continual RL Pipeline.
"""

import os
import tempfile
import numpy as np
import pytest

from src.agent import DQNAgent
from src.trainer import run_continual_experiment


def test_end_to_end_smoke_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        agent = DQNAgent(
            obs_dim=4,
            n_actions=2,
            hidden_dim=32,
            batch_size=16,
            buffer_size=500,
            target_sync=20,
            eps_decay_steps=50,
            strategy="car_sr",
            strategy_kwargs={"mem_per_task": 50, "mem_batch_size": 8, "parseval_beta": 0.01},
        )

        R, metrics, history = run_continual_experiment(
            agent=agent,
            sequence_name="similar",
            steps_per_task=100,
            eval_episodes=2,
            seed=42,
            verbose=False,
        )

        assert R.shape == (4, 3)
        assert "avg_final_return" in metrics
        assert "forgetting" in metrics
        assert "plasticity" in metrics

        # Test checkpoint save and reload
        ckpt_path = os.path.join(tmp_dir, "test_agent.pt")
        agent.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)

        new_agent = DQNAgent(obs_dim=4, n_actions=2, hidden_dim=32)
        new_agent.load_checkpoint(ckpt_path)
        assert new_agent.steps == agent.steps
