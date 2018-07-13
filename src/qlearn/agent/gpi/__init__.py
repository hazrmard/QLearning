"""
Defines `Agent` classes for Generalized Policy Iteration (GPI) approaches. GPI
relies on interleaving steps of policy evaluation and policy improvement. GPI
methods are based on valuation of states/actions and therefore have a value
function to calculate.
"""

from .agent import Agent
from .offpolicy import QAgent, NStepTDAgent
from .onpolicy import NStepSarsaAgent
from .agent import UNIFORM, SOFTMAX, GREEDY
