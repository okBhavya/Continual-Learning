"""
src/envs.py - CartPole Task Sequences and Conflict Environments for Continual RL.

Every task shares an identical observation space (4-dimensional: cart position,
cart velocity, pole angle, pole angular velocity) and action space (discrete: 0 or 1).
No task identity is provided to the agent.

Conflict is systematically varied across three task sequences:
1. "similar": Minor physics variations (gravity, pole length, push force).
2. "partial": Significant physics variation, followed by state-dependent action inversion
              (actions flipped only when cart position x > 0). Resolvable by a single network.
3. "dissimilar": Significant physics variation, followed by global action inversion
                 (push left <-> push right everywhere). Direct global conflict.
"""

from typing import Any, Dict, List
import gymnasium as gym


TASK_SEQUENCES: Dict[str, List[Dict[str, Any]]] = {
    "similar": [
        {"name": "T1-default", "gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": False},
        {"name": "T2-slightly-longer", "gravity": 11.0, "length": 0.7, "force_mag": 10.0, "flip": False},
        {"name": "T3-slightly-heavier-g", "gravity": 12.0, "length": 0.9, "force_mag": 9.0, "flip": False},
    ],
    "partial": [
        {"name": "T1-default", "gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": False},
        {"name": "T2-big-physics-change", "gravity": 15.0, "length": 1.0, "force_mag": 8.0, "flip": False},
        {"name": "T3-flip-when-x>0", "gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": "half"},
    ],
    "dissimilar": [
        {"name": "T1-default", "gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": False},
        {"name": "T2-long-weak-push", "gravity": 9.8, "length": 1.5, "force_mag": 6.0, "flip": False},
        {"name": "T3-flipped-actions", "gravity": 9.8, "length": 0.5, "force_mag": 10.0, "flip": True},
    ],
}


class FlipWhenRight(gym.Wrapper):
    """
    Swaps actions ONLY when the cart is on the right half of the track (x > 0).
    Conflicts with default policy on x > 0, but is separable because cart position x
    is present in the observation vector. A single fixed-capacity network can theoretically
    solve both without task IDs.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._x: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._x = float(obs[0])
        return obs, info

    def step(self, action: int):
        flipped = bool(self._x > 0)
        effective_action = (1 - action) if flipped else action
        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        self._x = float(obs[0])
        info["action_flipped"] = flipped
        return obs, reward, terminated, truncated, info


class FlipActions(gym.ActionWrapper):
    """
    Swaps action 0 <-> 1 globally across the entire state space.
    Forces a direct contradiction with earlier tasks (optimal policy inverted).
    """

    def action(self, action: int) -> int:
        return 1 - action


def make_env(cfg: Dict[str, Any], seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """
    Constructs a Gymnasium CartPole-v1 environment with custom physics parameters
    and conflict wrappers specified by `cfg`.
    """
    if render_mode is not None:
        env = gym.make("CartPole-v1", render_mode=render_mode)
    else:
        env = gym.make("CartPole-v1")
    u = env.unwrapped

    u.gravity = float(cfg.get("gravity", 9.8))
    u.length = float(cfg.get("length", 0.5))
    u.force_mag = float(cfg.get("force_mag", 10.0))

    # Recompute derived physics properties in CartPole
    u.polemass_length = u.masspole * u.length
    u.total_mass = u.masspole + u.masscart

    flip_mode = cfg.get("flip", False)
    if flip_mode == "half":
        env = FlipWhenRight(env)
    elif flip_mode is True:
        env = FlipActions(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
