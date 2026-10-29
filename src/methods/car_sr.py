"""
src/methods/car_sr.py - Conflict-Aware Replay with Spectral Regularization (CAR-SR).

Proposed Contribution:
1. Dynamic Gradient Conflict Diagnosis: Measures instantaneous cosine alignment
   between active-task and past-memory Bellman gradients: c_t = <g_curr, g_mem> / (||g_curr|| * ||g_mem||).
2. Gradient Orthogonal Projection: When c_t < 0 (destructive task conflict),
   projects g_curr onto the normal plane of g_mem, ensuring <g_task, g_mem> >= 0.
3. Parseval Spectral Frame Regularization: Enforces layer-wise weight matrix
   isometry (||W^T W - I||_F^2) on hidden representations to protect plasticity
   without freezing parameters.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseCLStrategy
from src.buffer import Transition, to_tensors, MultiTaskMemory


class CARSRStrategy(BaseCLStrategy):
    """
    Conflict-Aware Replay with Spectral Regularization (CAR-SR).
    """

    def __init__(
        self,
        mem_per_task: int = 3000,
        mem_batch_size: int = 32,
        mem_weight: float = 1.0,
        parseval_beta: float = 0.01,
        enable_projection: bool = True,
        enable_parseval: bool = True,
    ):
        super().__init__(name="car_sr")
        self.mem_per_task = mem_per_task
        self.mem_batch_size = mem_batch_size
        self.mem_weight = mem_weight
        self.parseval_beta = parseval_beta
        self.enable_projection = enable_projection
        self.enable_parseval = enable_parseval

        self.memory = MultiTaskMemory(capacity_per_task=mem_per_task)

        # Diagnostics counters
        self.total_updates = 0
        self.conflict_count = 0
        self.last_cosine = 0.0

    def start_task(self, agent: Any, task_idx: int):
        pass

    def end_task(self, agent: Any, task_idx: int):
        self.memory.store_task_data(agent.buffer)

    def _compute_td_loss(self, agent: Any, batch: List[Transition]) -> torch.Tensor:
        s, a, r, s2, d = to_tensors(batch, agent.device)
        q_sa = agent.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = agent.q_target(s2).max(dim=1).values
            target = r + agent.gamma * (1.0 - d) * next_q
        return F.smooth_l1_loss(q_sa, target)

    def update_step(self, agent: Any, curr_batch: List[Transition]) -> Dict[str, float]:
        self.total_updates += 1

        # 1. Active task loss & gradient
        loss_curr = self._compute_td_loss(agent, curr_batch)
        agent.opt.zero_grad()
        loss_curr.backward()
        g_curr = [p.grad.detach().clone() if p.grad is not None else None for p in agent.q.parameters()]

        has_memory = (len(self.memory) > 0)
        c_t = 1.0
        is_conflict = False

        if has_memory:
            # 2. Replay memory loss & gradient
            mem_batch = self.memory.sample(self.mem_batch_size)
            loss_mem = self._compute_td_loss(agent, mem_batch)

            agent.opt.zero_grad()
            loss_mem.backward()
            g_mem = [p.grad.detach().clone() if p.grad is not None else None for p in agent.q.parameters()]

            # 3. Inner product and cosine similarity
            dot_prod = 0.0
            norm_curr_sq = 0.0
            norm_mem_sq = 0.0

            for gc, gm in zip(g_curr, g_mem):
                if gc is not None and gm is not None:
                    dot_prod += torch.sum(gc * gm).item()
                    norm_curr_sq += torch.sum(gc * gc).item()
                    norm_mem_sq += torch.sum(gm * gm).item()

            norm_curr = (norm_curr_sq ** 0.5) + 1e-8
            norm_mem = (norm_mem_sq ** 0.5) + 1e-8
            c_t = dot_prod / (norm_curr * norm_mem)
            self.last_cosine = c_t

            # 4. Conflict Resolution via Orthogonal Projection
            if c_t < 0.0 and self.enable_projection:
                is_conflict = True
                self.conflict_count += 1
                proj_coeff = dot_prod / (norm_mem_sq + 1e-8)
                # g_curr_proj = g_curr - proj_coeff * g_mem
                # g_task = g_curr_proj + mem_weight * g_mem
                g_task = [
                    (gc - proj_coeff * gm) + self.mem_weight * gm
                    if gc is not None and gm is not None else gc
                    for gc, gm in zip(g_curr, g_mem)
                ]
            else:
                # Concordant / forward transfer: direct cooperative combination
                g_task = [
                    gc + self.mem_weight * gm
                    if gc is not None and gm is not None else gc
                    for gc, gm in zip(g_curr, g_mem)
                ]
        else:
            loss_mem = torch.tensor(0.0, device=agent.device)
            g_task = g_curr

        # 5. Parseval Regularization Gradient
        loss_parseval = torch.tensor(0.0, device=agent.device)
        if self.enable_parseval:
            loss_parseval = agent.q.compute_parseval_loss()
            agent.opt.zero_grad()
            (self.parseval_beta * loss_parseval).backward()
            g_parseval = [p.grad.detach().clone() if p.grad is not None else None for p in agent.q.parameters()]
        else:
            g_parseval = [None] * len(g_task)

        # 6. Synthesize combined gradient and apply
        agent.opt.zero_grad()
        for p, gt, gp in zip(agent.q.parameters(), g_task, g_parseval):
            if p.requires_grad:
                final_g = torch.zeros_like(p.data)
                if gt is not None:
                    final_g += gt
                if gp is not None:
                    final_g += gp
                p.grad = final_g

        nn.utils.clip_grad_norm_(agent.q.parameters(), agent.max_grad_norm)
        agent.opt.step()

        return {
            "td_curr_loss": float(loss_curr.item()),
            "td_mem_loss": float(loss_mem.item()) if has_memory else 0.0,
            "parseval_loss": float(loss_parseval.item()) if self.enable_parseval else 0.0,
            "conflict_cosine": float(c_t),
            "is_conflict": 1.0 if is_conflict else 0.0,
            "conflict_ratio": float(self.conflict_count / max(1, self.total_updates)),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "conflict_count": self.conflict_count,
            "conflict_ratio": float(self.conflict_count / max(1, self.total_updates)),
            "last_cosine": float(self.last_cosine),
        }
