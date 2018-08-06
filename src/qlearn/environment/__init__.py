"""
Defines `Environment` class that wraps reward/transition/reward functions
into a `gym.core.Env` compatible object.

All `Environment` classes have a `step(action)` method and a `reset()` method.

All `Environment` classes have a `action_space` and `observation_space` attribute
of type `gym.core.Space`.
"""

from .environment import Environment
from . import dummy
