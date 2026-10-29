"""
src/methods/parseval.py - Parseval Regularization for Continual RL.

Implementation of Parseval Spectral Regularization (Chung et al., NeurIPS 2024).
Regularizes hidden layer weight matrices towards isometry to preserve network
plasticity and prevent feature rank collapse without parameter freezing.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseCLStrategy
from src.buffer import Transition, to_tensors


class ParsevalStrategy(BaseCLStrategy):
    """
    Parseval Frame Regularization:
    - Buffers cleared at task transitions (zero-memory approach).
    - Objective: L_total = L_TD + beta * L_Parseval, where
      L_Parseval = 0.5 * sum_l || W_l^T W_l - I ||_F^2.
    """

    def __init__(self, parseval_beta: float = 0.01):
        super().__init__(name="parseval")
        self.parseval_beta = parseval_beta

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

        td_loss = F.smooth_l1_loss(q_sa, target)
        parseval_loss = agent.q.compute_parseval_loss()
        total_loss = td_loss + self.parseval_beta * parseval_loss

        agent.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.q.parameters(), agent.max_grad_norm)
        agent.opt.step()

        return {
            "td_loss": float(td_loss.item()),
            "parseval_loss": float(parseval_loss.item()),
            "total_loss": float(total_loss.item()),
        }
