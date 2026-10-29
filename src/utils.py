"""
src/utils.py - Configuration, Seeding, and Serialization Utilities.
"""

import json
import os
import random
from typing import Dict, Any
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """Sets deterministic seeds across Python, NumPy, and PyTorch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    """Loads and validates a YAML experiment configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dictionary in config file {path}, got {type(cfg)}")
    return cfg


def save_json(data: Any, path: str):
    """Saves arbitrary serializable data to JSON file, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Loads data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
