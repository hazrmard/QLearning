import unittest
import random
from typing import Tuple, Union

from ..agent import Agent
from ..environment import dummy
from ..approximator import Tabular
from ..algorithm import q, nsteptd, nstepsarsa



class DummyEnv(dummy.DummyIdentity):
    """
    An identity environment with only 2 states.
    """

    NSTATES = 2
    GOAL_STATES = (1,)

    def reset(self):
        self.state = 0
        return 0



class DummyAgent(Agent):
    """
    An agent with pre-defined policy. Picks actions randomly from a tuple
    of action choices provided at instantiation.
    """

    def __init__(self, actions: Tuple[Tuple[Union[int, float]]], env, value_function):
        self.action_pool = actions
        super().__init__(env=env, value_function=value_function)



    def set_action_selection_policy(self, policy):
        self.next_action = lambda x: random.choice(self.action_pool)



class TestGPIAlgorithms(unittest.TestCase):

    def setUp(self):
        self.env = DummyEnv()
        self.env.maxsteps = 3
        self.value = Tabular(dims=(2,2), lrate=1e-1)
        self.value.table *= 0
        self.agent = DummyAgent(actions=((0,),), env=self.env, value_function=self.value)        



    def test_q(self):
        r = self.agent.learn(algorithm=q, episodes=1, discount=1)
        expected = -0.1 + (-0.09) + (-0.081)
        actual = self.agent.value.table[0,0]
        self.assertAlmostEqual(expected, actual)



if __name__ == '__main__':
    unittest.main(verbosity=0)
