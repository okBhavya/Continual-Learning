"""
src/methods/replay.py - Experience Replay Baseline with Multi-Task Exemplar Memory.
"""

from typing import Dict, List, Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseCLStrategy
from src.buffer import Transition, to_tensors, MultiTaskMemory


class ReplayStrategy(BaseCLStrategy):
    """
    Standard Rehearsal Baseline:
    - Retains up to `mem_per_task` transitions per finished task in memory.
    - Each mini-batch is composed of (1 - replay_frac) current task transitions
      and (replay_frac) uniformly sampled past-task transitions.
    """

    def __init__(self, mem_per_task: int = 3000, replay_frac: float = 0.5):
        super().__init__(name="replay")
        self.mem_per_task = mem_per_task
        self.replay_frac = replay_frac
        self.memory = MultiTaskMemory(capacity_per_task=mem_per_task)

    def start_task(self, agent: Any, task_idx: int):
        pass

    def end_task(self, agent: Any, task_idx: int):
        self.memory.store_task_data(agent.buffer)

    def update_step(self, agent: Any, curr_batch: List[Transition]) -> Dict[str, float]:
        # Form composite batch
        if len(self.memory) > 0:
            n_old = int(agent.batch_size * self.replay_frac)
            n_new = agent.batch_size - n_old
            effective_batch = curr_batch[:n_new] + self.memory.sample(n_old)
        else:
            effective_batch = curr_batch

        s, a, r, s2, d = to_tensors(effective_batch, agent.device)
        q_sa = agent.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = agent.q_target(s2).max(dim=1).values
            target = r + agent.gamma * (1.0 - d) * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        agent.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.q.parameters(), agent.max_grad_norm)
        agent.opt.step()

        return {
            "td_loss": float(loss.item()),
            "total_loss": float(loss.item()),
            "stored_transitions": float(len(self.memory)),
        }
