"""
src/methods/base.py - Abstract Base Class for Continual Learning Strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
from src.buffer import Transition


class BaseCLStrategy(ABC):
    """
    Interface governing how an RL agent updates parameters and manages memory
    across sequential task transitions.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def start_task(self, agent: Any, task_idx: int):
        """Invoked when training commences on a new sequential task."""
        pass

    @abstractmethod
    def end_task(self, agent: Any, task_idx: int):
        """Invoked immediately after completing training on a task."""
        pass

    @abstractmethod
    def update_step(self, agent: Any, curr_batch: List[Transition]) -> Dict[str, float]:
        """
        Executes one gradient update step.
        Returns a dictionary of loss and diagnostic metrics for logging.
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns strategy-specific diagnostic metrics for logging/analysis."""
        return {}
