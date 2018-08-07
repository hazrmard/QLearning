"""
Defines `Environment` class that wraps reward/transition/reward functions
into a `gym.core.Env` compatible object.

All `Environment` classes have the following API:

* Methods:
  * `step(action)` which returns next state, reward, episode over, and any object
  containing diagnostic info. Action should conform to the type of the action_space.
  * `reset()` which sets the environment to some random initial state and
  returns that state vector. The return type conforms to observation_space.

* Attributes:
  * `action_space`: A `gym.core.Spaces` subclass which defines the actions possible.
  * `observation_space`: A `gym.core.Spaces` subclass which defines the range
  of observable states.
"""

from .environment import Environment
from . import dummy
