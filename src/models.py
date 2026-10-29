"""
src/models.py - Neural Network Architectures and Spectral Regularization.

Implements QNet (MLP value function) and Parseval frame orthogonality penalties
as introduced in Chung et al. (NeurIPS 2024).
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class QNet(nn.Module):
    """
    Standard Feed-Forward Deep Q-Network.
    Architecture: obs_dim -> hidden -> hidden -> n_actions
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        q_vals = self.out(h2)
        return q_vals

    def get_hidden_layers(self) -> List[nn.Linear]:
        """Returns the internal linear representation layers excluding the output head."""
        return [self.fc1, self.fc2]

    def compute_parseval_loss(self) -> torch.Tensor:
        """
        Computes the Parseval frame regularization loss across hidden linear layers:
        L_parseval = (1/2) * sum_l || W_l^T W_l - I ||_F^2

        Enforces near-orthogonality (isometry), preserving singular values and gradient
        flow through hidden layers to prevent plasticity loss (Chung et al., NeurIPS 2024).
        """
        loss = torch.tensor(0.0, device=self.fc1.weight.device)
        for layer in self.get_hidden_layers():
            w = layer.weight  # shape: (out_features, in_features)
            # If out >= in, w^T @ w -> (in, in)
            # If in > out, w @ w^T -> (out, out)
            if w.shape[0] >= w.shape[1]:
                gram = torch.matmul(w.t(), w)
                identity = torch.eye(w.shape[1], device=w.device)
            else:
                gram = torch.matmul(w, w.t())
                identity = torch.eye(w.shape[0], device=w.device)

            diff = gram - identity
            loss = loss + 0.5 * torch.sum(diff ** 2)
        return loss

    def compute_spectral_diagnostics(self) -> Tuple[float, float]:
        """
        Calculates mean singular value and condition number across hidden layers.
        Useful for tracking plasticity preservation and representation rank.
        """
        cond_nums = []
        mean_svs = []
        with torch.no_grad():
            for layer in self.get_hidden_layers():
                w = layer.weight.detach()
                s = torch.linalg.svdvals(w)
                mean_svs.append(float(s.mean().item()))
                cond = float((s[0] / (s[-1] + 1e-8)).item())
                cond_nums.append(cond)
        return float(sum(mean_svs) / len(mean_svs)), float(sum(cond_nums) / len(cond_nums))
