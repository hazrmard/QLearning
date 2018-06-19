"""
Defines the `Agent` class and sub-classes. Agents explore environment using
some action-selection policy, and derive control behavior.

Defines utility functions for converting `gym.Space` instances into `Tuples`
to be used consistently everywhere. See `spaces` module.

Defines utility functions to maximize a function over continuous, discrete, and
hybrid spaces. See `spaces` module.
"""

from .agent import Agent
from .agent import UNIFORM, GREEDY, SOFTMAX
from . import spaces
from .parameters import Schedule, LinearSchedule, LogarithmicSchedule, ExponentialSchedule
from .memory import Memory
from .onpolicy import NSarsaAgent
from .offpolicy import QAgent, TDLambdaAgent
