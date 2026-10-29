"""
tests/test_methods.py - Unit Tests for CL Strategies (Naive, Replay, EWC, Parseval, CAR-SR).
"""

import numpy as np
import pytest
import torch

from src.agent import DQNAgent
from src.methods.naive import NaiveStrategy
from src.methods.replay import ReplayStrategy
from src.methods.ewc import EWCStrategy
from src.methods.parseval import ParsevalStrategy
from src.methods.car_sr import CARSRStrategy


def dummy_batch(batch_size=8):
    batch = []
    for _ in range(batch_size):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randint(0, 2)
        r = 1.0
        s2 = np.random.randn(4).astype(np.float32)
        d = 0.0
        batch.append((s, a, r, s2, d))
    return batch


@pytest.mark.parametrize("strat_name", ["naive", "replay", "ewc", "parseval", "car_sr"])
def test_strategy_instantiation_and_step(strat_name):
    agent = DQNAgent(hidden_dim=32, strategy=strat_name, buffer_size=100)
    batch = dummy_batch(16)
    agent.buffer.buf.extend(batch)

    info = agent.strategy.update_step(agent, batch)
    assert isinstance(info, dict)
    assert len(info) > 0


def test_ewc_fisher_computation():
    agent = DQNAgent(hidden_dim=32, strategy="ewc", buffer_size=100)
    batch = dummy_batch(20)
    agent.buffer.buf.extend(batch)

    assert isinstance(agent.strategy, EWCStrategy)
    fisher = agent.strategy.compute_fisher(agent)
    for name, f in fisher.items():
        assert f.shape == dict(agent.q.named_parameters())[name].shape
        assert (f >= 0.0).all()


def test_car_sr_gradient_projection():
    agent = DQNAgent(hidden_dim=32, strategy="car_sr", buffer_size=100)
    assert isinstance(agent.strategy, CARSRStrategy)

    # Store some mock transitions in memory
    for t in dummy_batch(20):
        agent.buffer.push(*t)
    agent.strategy.end_task(agent, 0)
    assert len(agent.strategy.memory) > 0

    batch = dummy_batch(16)
    info = agent.strategy.update_step(agent, batch)
    assert "conflict_cosine" in info
    assert -1.0 <= info["conflict_cosine"] <= 1.0
