"""
src/methods/naive.py - Sequential Fine-Tuning Baseline (Naive Continual Learning).
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseCLStrategy
from src.buffer import Transition, to_tensors


class NaiveStrategy(BaseCLStrategy):
    """
    Standard sequential fine-tuning:
    - Buffers are cleared at task boundaries.
    - No mechanism to protect previous task representations.
    - Serves as the primary baseline for evaluating catastrophic forgetting.
    """

    def __init__(self):
        super().__init__(name="naive")

    def start_task(self, agent: Any, task_idx: int):
        pass

    def end_task(self, agent: Any, task_idx: int):
        pass

    def update_step(self, agent: Any, curr_batch: List[Transition]) -> Dict[str, float]:
        s, a, r, s2, d = to_tensors(curr_batch, agent.device)
        q_sa = agent.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = agent.q_target(s2).max(dim=1).values
            target = r + agent.gamma * (1.0 - d) * next_q

        loss = F.smooth_l1_loss(q_sa, target)

        agent.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.q.parameters(), agent.max_grad_norm)
        agent.opt.step()

        return {"td_loss": float(loss.item()), "total_loss": float(loss.item())}
