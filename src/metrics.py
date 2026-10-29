"""
src/metrics.py - Continual Reinforcement Learning Evaluation Metrics.

Implements formal definitions of:
- Average Final Return (AFR)
- Catastrophic Forgetting (F)
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Plasticity (P)
"""

from typing import Dict, Any
import numpy as np


def compute_crl_metrics(R: np.ndarray) -> Dict[str, float]:
    """
    Computes standard continual learning metrics from evaluation matrix R.

    Parameters
    ----------
    R : np.ndarray, shape (N + 1, N)
        R[0, j]     = Return on task j with untrained network.
        R[i + 1, j] = Return on task j after completing training on task i.
        (N = number of tasks)

    Returns
    -------
    dict with:
        - avg_final_return: Mean return across all tasks after training on final task.
        - forgetting: Mean performance drop on past tasks from post-training to final stage.
        - peak_forgetting: Mean performance drop on past tasks from historical peak to final stage.
        - backward_transfer: Mean change in past task performance (R[N, j] - R[j+1, j]).
        - forward_transfer: Zero-shot head start on upcoming tasks relative to untrained baseline.
        - plasticity: Mean immediate return on each task right after learning it.
    """
    if R.ndim != 2:
        raise ValueError(f"Expected 2D return matrix, got ndim={R.ndim}")

    num_eval_rows, num_tasks = R.shape
    if num_eval_rows != num_tasks + 1:
        raise ValueError(
            f"Expected shape (N + 1, N) for {num_tasks} tasks, got {R.shape}"
        )

    # Final evaluation vector (after training on all tasks)
    final_returns = R[-1]  # shape (N,)
    avg_final_return = float(np.mean(final_returns))

    # Plasticity: return on task j right after training on task j (row j + 1)
    immediate_returns = np.array([R[j + 1, j] for j in range(num_tasks)])
    plasticity = float(np.mean(immediate_returns))

    if num_tasks > 1:
        # Forgetting: drop from immediate return on task j to final return R[N, j]
        forgetting_per_task = [R[j + 1, j] - R[-1, j] for j in range(num_tasks - 1)]
        forgetting = float(np.mean(forgetting_per_task))

        # Peak forgetting: drop from max historical return achieved on task j
        peak_drops = []
        for j in range(num_tasks - 1):
            historical_peak = np.max(R[j + 1 : -1, j]) if (num_tasks > 2) else R[j + 1, j]
            peak_drops.append(max(0.0, float(historical_peak - R[-1, j])))
        peak_forgetting = float(np.mean(peak_drops))

        # Backward Transfer: BWT = -forgetting
        backward_transfer = float(np.mean([R[-1, j] - R[j + 1, j] for j in range(num_tasks - 1)]))

        # Forward Transfer: performance on task j right before training on it minus untrained
        # task j is trained at stage j (1-indexed), so right before is row j
        fwt_per_task = [R[j, j] - R[0, j] for j in range(1, num_tasks)]
        forward_transfer = float(np.mean(fwt_per_task))
    else:
        forgetting = 0.0
        peak_forgetting = 0.0
        backward_transfer = 0.0
        forward_transfer = 0.0

    return {
        "avg_final_return": avg_final_return,
        "plasticity": plasticity,
        "forgetting": forgetting,
        "peak_forgetting": peak_forgetting,
        "backward_transfer": backward_transfer,
        "forward_transfer": forward_transfer,
    }
