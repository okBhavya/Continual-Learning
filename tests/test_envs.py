"""
tests/test_envs.py - Unit Tests for Continual Task Sequences and Conflict Wrappers.
"""

import numpy as np
import pytest
from src.envs import TASK_SEQUENCES, make_env, FlipWhenRight, FlipActions


def test_task_sequences_definitions():
    assert "similar" in TASK_SEQUENCES
    assert "partial" in TASK_SEQUENCES
    assert "dissimilar" in TASK_SEQUENCES

    for seq_name, tasks in TASK_SEQUENCES.items():
        assert len(tasks) == 3, f"Sequence {seq_name} should have 3 tasks"
        for t in tasks:
            assert "gravity" in t
            assert "length" in t
            assert "force_mag" in t
            assert "flip" in t


def test_env_creation_and_step():
    cfg = TASK_SEQUENCES["similar"][0]
    env = make_env(cfg, seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (4,)
    assert isinstance(obs, np.ndarray)

    next_obs, reward, term, trunc, info = env.step(0)
    assert next_obs.shape == (4,)
    assert isinstance(reward, (float, int))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    env.close()


def test_flip_when_right_wrapper():
    cfg = {"gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": "half", "name": "test"}
    env = make_env(cfg, seed=42)
    obs, _ = env.reset(seed=42)

    # Force internal cart position x > 0
    wrapper = env
    while not isinstance(wrapper, FlipWhenRight) and hasattr(wrapper, "env"):
        wrapper = wrapper.env
    assert isinstance(wrapper, FlipWhenRight)

    wrapper._x = 0.5
    _, _, _, _, info = env.step(0)
    assert info.get("action_flipped") is True

    wrapper._x = -0.5
    _, _, _, _, info = env.step(0)
    assert info.get("action_flipped") is False
    env.close()


def test_flip_actions_wrapper():
    cfg = {"gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": True, "name": "test"}
    env = make_env(cfg, seed=42)
    wrapper = env
    while not isinstance(wrapper, FlipActions) and hasattr(wrapper, "env"):
        wrapper = wrapper.env
    assert isinstance(wrapper, FlipActions)
    assert wrapper.action(0) == 1
    assert wrapper.action(1) == 0
    env.close()
