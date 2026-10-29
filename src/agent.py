"""
src/agent.py - Deep Q-Network Agent with Modular Continual Learning Strategy.
"""

import random
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.optim as optim

from src.models import QNet
from src.buffer import ReplayBuffer
from src.methods.base import BaseCLStrategy
from src.methods import build_strategy


class DQNAgent:
    """
    DQN Agent supporting plug-and-play Continual Learning strategies.
    Operates without task IDs on a unified observation-action space.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 20_000,
        target_sync: int = 250,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 5_000,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
        strategy: str | BaseCLStrategy = "naive",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Networks
        self.q = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Optimizer
        self.lr = lr
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

        # Current-task active buffer
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)

        # State counters
        self.steps = 0
        self.epsilon = eps_start
        self.current_task_idx = 0

        # Continual Learning Strategy
        if isinstance(strategy, str):
            self.strategy = build_strategy(strategy, **(strategy_kwargs or {}))
        else:
            self.strategy = strategy

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Selects an action using epsilon-greedy exploration."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(s_t)
            return int(q_vals.argmax(dim=1).item())

    def _update_epsilon(self):
        frac = min(1.0, self.steps / float(self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def observe(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: float) -> Optional[Dict[str, float]]:
        """Stores transition in buffer and performs learning step if batch is ready."""
        self.buffer.push(s, a, r, s2, done)
        self.steps += 1
        self._update_epsilon()

        step_info = None
        if len(self.buffer) >= self.batch_size:
            curr_batch = self.buffer.sample(self.batch_size)
            step_info = self.strategy.update_step(self, curr_batch)

        if self.steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return step_info

    def start_task(self, task_idx: int):
        """Prepares agent state for a new sequential task."""
        self.current_task_idx = task_idx
        self.buffer.clear()
        self.steps = 0
        self.epsilon = self.eps_start
        self.q_target.load_state_dict(self.q.state_dict())
        self.strategy.start_task(self, task_idx)

    def end_task(self, task_idx: int):
        """Finalizes agent state and stores memory after completing a task."""
        self.strategy.end_task(self, task_idx)

    def save_checkpoint(self, path: str):
        """Saves weights and agent configuration to disk."""
        torch.save({
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "q_state_dict": self.q.state_dict(),
            "target_state_dict": self.q_target.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
            "current_task_idx": self.current_task_idx,
            "strategy_name": self.strategy.name,
        }, path)

    def load_checkpoint(self, path: str):
        """Loads weights and agent configuration from disk."""
        ckpt = torch.load(path, map_location=self.device)
        if "hidden_dim" in ckpt and ckpt["hidden_dim"] != self.hidden_dim:
            self.hidden_dim = ckpt["hidden_dim"]
            self.obs_dim = ckpt.get("obs_dim", self.obs_dim)
            self.n_actions = ckpt.get("n_actions", self.n_actions)
            self.q = QNet(obs_dim=self.obs_dim, n_actions=self.n_actions, hidden_dim=self.hidden_dim).to(self.device)
            self.q_target = QNet(obs_dim=self.obs_dim, n_actions=self.n_actions, hidden_dim=self.hidden_dim).to(self.device)
            self.opt = optim.Adam(self.q.parameters(), lr=self.lr)

        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_target.load_state_dict(ckpt["target_state_dict"])
        if "opt_state_dict" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt_state_dict"])
            except Exception:
                pass
        self.steps = ckpt.get("steps", 0)
        self.epsilon = ckpt.get("epsilon", self.eps_end)
        self.current_task_idx = ckpt.get("current_task_idx", 0)
