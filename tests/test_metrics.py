"""
tests/test_metrics.py - Unit Tests for Formal Continual Learning Metrics.
"""

import numpy as np
import pytest
from src.metrics import compute_crl_metrics


def test_perfect_retention_metrics():
    # 3 tasks: row 0 untrained, rows 1..3 post-training on T1, T2, T3
    # Scenario: Agent learns all tasks to 500 and never forgets
    R = np.array([
        [10.0, 10.0, 10.0],  # Untrained
        [500.0, 10.0, 10.0], # After T1
        [500.0, 500.0, 10.0],# After T2
        [500.0, 500.0, 500.0]# After T3
    ])

    m = compute_crl_metrics(R)
    assert m["avg_final_return"] == 500.0
    assert m["plasticity"] == 500.0
    assert m["forgetting"] == 0.0
    assert m["peak_forgetting"] == 0.0
    assert m["backward_transfer"] == 0.0


def test_catastrophic_forgetting_calculation():
    # Scenario: Agent learns each task to 400, but previous tasks drop to 100
    R = np.array([
        [10.0, 10.0, 10.0],  # Untrained
        [400.0, 10.0, 10.0], # After T1
        [100.0, 400.0, 10.0],# After T2
        [100.0, 100.0, 400.0]# After T3
    ])

    m = compute_crl_metrics(R)
    # T1 drop: 400 - 100 = 300
    # T2 drop: 400 - 100 = 300
    # Mean forgetting: (300 + 300) / 2 = 300
    assert m["forgetting"] == 300.0
    assert m["backward_transfer"] == -300.0
    assert m["plasticity"] == 400.0
    assert m["avg_final_return"] == 200.0  # (100 + 100 + 400) / 3 = 200


def test_forward_transfer_calculation():
    # Scenario: Learning T1 boosts zero-shot T2 from 10 to 60, and T3 to 80
    R = np.array([
        [10.0, 10.0, 10.0],
        [300.0, 60.0, 20.0],  # Before T2, T2 is 60 (boost of 50)
        [200.0, 300.0, 80.0], # Before T3, T3 is 80 (boost of 70)
        [200.0, 250.0, 300.0]
    ])

    m = compute_crl_metrics(R)
    # FWT T2: 60 - 10 = 50
    # FWT T3: 80 - 10 = 70
    # Mean FWT: (50 + 70) / 2 = 60.0
    assert m["forward_transfer"] == 60.0
