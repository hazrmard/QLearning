"""
Defines `Agent` classes for Generalized Policy Iteration (GPI) approaches. GPI
relies on interleaving steps of policy evaluation and policy improvement. GPI
methods are based on valuation of states/actions and therefore have a value
function to calculate.

All `Agent`s have the following API:

* Methods
  * Initialization with an 'Environment` and `Approximator` object to interact
  with and store action values in the agent's environment.
  * `learn(episodes, **kwargs)` calls an algorithm to take exploratory actions
  and store rewards for each episode.
  * `recommend(state)` returns the most valuable action to take. The state and
  returned action are tuples of int/float.
  * `next_action(state)` returns the next action to take based on an exploratory
  policy. Arguments and returns are tuples of int/float.
"""

from .agent import Agent
from .offpolicy import QAgent, NStepTDAgent
from .onpolicy import NStepSarsaAgent
from .agent import UNIFORM, SOFTMAX, GREEDY
