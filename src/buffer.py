"""
src/buffer.py - Experience Replay and Reservoir Buffers for Continual RL.
"""

from collections import deque
import random
from typing import List, Tuple, Any
import numpy as np
import torch


Transition = Tuple[np.ndarray, int, float, np.ndarray, float]


def to_tensors(batch: List[Transition], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts a batch of transitions into PyTorch tensors on the target device."""
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.int64, device=device),
        torch.as_tensor(rewards, dtype=torch.float32, device=device),
        torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
        torch.as_tensor(dones, dtype=torch.float32, device=device),
    )


class ReplayBuffer:
    """Standard fixed-capacity FIFO replay buffer for active task online learning."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: float):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def clear(self):
        self.buf.clear()

    def __len__(self) -> int:
        return len(self.buf)


class MultiTaskMemory:
    """
    Structured memory repository storing exemplar transitions from past tasks.
    Supports balanced sampling across previous tasks or flat sampling.
    """

    def __init__(self, capacity_per_task: int = 3000):
        self.capacity_per_task = capacity_per_task
        self.task_memories: List[List[Transition]] = []

    def store_task_data(self, source_buffer: ReplayBuffer):
        """Samples up to capacity_per_task transitions from finished task buffer."""
        k = min(self.capacity_per_task, len(source_buffer))
        if k > 0:
            exemplars = random.sample(list(source_buffer.buf), k)
            self.task_memories.append(exemplars)

    def sample(self, n: int) -> List[Transition]:
        """Samples uniformly across all stored past-task exemplars."""
        if not self.task_memories:
            return []
        all_samples = [t for mem in self.task_memories for t in mem]
        return random.sample(all_samples, min(n, len(all_samples)))

    def sample_balanced(self, n: int) -> List[Transition]:
        """Samples equally from each previously completed task."""
        if not self.task_memories:
            return []
        num_tasks = len(self.task_memories)
        per_task = max(1, n // num_tasks)
        batch = []
        for mem in self.task_memories:
            k = min(per_task, len(mem))
            batch.extend(random.sample(mem, k))
        return batch[:n]

    @property
    def total_transitions(self) -> int:
        return sum(len(mem) for mem in self.task_memories)

    def __len__(self) -> int:
        return self.total_transitions
