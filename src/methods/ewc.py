"""
src/methods/ewc.py - Elastic Weight Consolidation (EWC) Baseline for DQN.

Kirkpatrick et al. (PNAS 2017) adapted to value-based reinforcement learning.
Estimates the diagonal Fisher Information Matrix (FIM) over the empirical policy
distribution pi(a|s) = softmax(Q(s, :)).
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import BaseCLStrategy
from src.buffer import Transition, to_tensors


class EWCStrategy(BaseCLStrategy):
    """
    Elastic Weight Consolidation:
    - At the conclusion of task k, evaluates diagonal Fisher Information.
    - Adds quadratic penalty: L_ewc = 0.5 * lambda * sum_k sum_p F_{k,p} * (theta_p - theta^*_{k,p})^2.
    """

    def __init__(self, ewc_lambda: float = 100.0, fisher_samples: int = 1000):
        super().__init__(name="ewc")
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.ewc_store: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []

    def start_task(self, agent: Any, task_idx: int):
        pass

    def compute_fisher(self, agent: Any) -> Dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p) for n, p in agent.q.named_parameters()}
        n_samples = min(self.fisher_samples, len(agent.buffer))
        if n_samples == 0:
            return fisher

        samples = agent.buffer.sample(n_samples)
        for (s, *_rest) in samples:
            agent.q.zero_grad()
            s_t = torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            logp = F.log_softmax(agent.q(s_t), dim=1)
            # Sample an action from the policy distribution
            action = torch.multinomial(logp.exp(), 1).item()
            logp[0, action].backward()

            for name, p in agent.q.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2

        for name in fisher:
            fisher[name] /= float(n_samples)

        agent.q.zero_grad()
        return fisher

    def end_task(self, agent: Any, task_idx: int):
        fisher = self.compute_fisher(agent)
        params = {n: p.detach().clone() for n, p in agent.q.named_parameters()}
        self.ewc_store.append((fisher, params))

    def compute_penalty(self, agent: Any) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=agent.device)
        for fisher, old_params in self.ewc_store:
            for name, p in agent.q.named_parameters():
                penalty = penalty + (fisher[name] * (p - old_params[name]) ** 2).sum()
        return 0.5 * self.ewc_lambda * penalty

    def update_step(self, agent: Any, curr_batch: List[Transition]) -> Dict[str, float]:
        s, a, r, s2, d = to_tensors(curr_batch, agent.device)
        q_sa = agent.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = agent.q_target(s2).max(dim=1).values
            target = r + agent.gamma * (1.0 - d) * next_q

        td_loss = F.smooth_l1_loss(q_sa, target)
        penalty = self.compute_penalty(agent) if self.ewc_store else torch.tensor(0.0, device=agent.device)
        total_loss = td_loss + penalty

        agent.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.q.parameters(), agent.max_grad_norm)
        agent.opt.step()

        return {
            "td_loss": float(td_loss.item()),
            "ewc_penalty": float(penalty.item()),
            "total_loss": float(total_loss.item()),
        }
