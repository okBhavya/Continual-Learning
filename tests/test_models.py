"""
tests/test_models.py - Unit Tests for QNet and Parseval Spectral Regularization.
"""

import torch
import pytest
from src.models import QNet


def test_qnet_forward_shape():
    net = QNet(obs_dim=4, n_actions=2, hidden_dim=64)
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


def test_parseval_loss_computation():
    net = QNet(obs_dim=4, n_actions=2, hidden_dim=64)
    loss = net.compute_parseval_loss()
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Test backward differentiability
    loss.backward()
    for layer in net.get_hidden_layers():
        assert layer.weight.grad is not None


def test_spectral_diagnostics():
    net = QNet(obs_dim=4, n_actions=2, hidden_dim=64)
    mean_sv, cond_num = net.compute_spectral_diagnostics()
    assert mean_sv > 0.0
    assert cond_num >= 1.0
