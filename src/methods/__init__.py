"""
src/methods/__init__.py - Continual Learning Strategy Factory.
"""

from typing import Dict, Any
from src.methods.base import BaseCLStrategy
from src.methods.naive import NaiveStrategy
from src.methods.replay import ReplayStrategy
from src.methods.ewc import EWCStrategy
from src.methods.parseval import ParsevalStrategy
from src.methods.car_sr import CARSRStrategy


STRATEGY_REGISTRY = {
    "naive": NaiveStrategy,
    "replay": ReplayStrategy,
    "ewc": EWCStrategy,
    "parseval": ParsevalStrategy,
    "car_sr": CARSRStrategy,
}


def build_strategy(name: str, **kwargs) -> BaseCLStrategy:
    """Builds and returns the requested continual learning strategy."""
    name_clean = name.lower().strip()
    if name_clean not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name_clean](**kwargs)
