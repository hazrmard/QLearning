import numpy as np
from gym.spaces import Discrete, Box

from ..environment import Environment



class DummySwitch(Environment):
    """
    A discrete environment represented by a switch taking a finite number of
    states. A state is represented by a single number. An action moves the state
    value up, down, or remains static.
    """

    NSTATES = 5
    GOAL_STATES = (2,)


    @classmethod
    def goal_func(cls, state: int) -> bool:
        return state in cls.GOAL_STATES


    @classmethod
    def transition_func(cls, state: int, action: int) -> int:
        nstate = state + action - 1
        if nstate < 0:
            return 0
        if nstate >= cls.NSTATES:
            return cls.NSTATES
        return nstate


    @classmethod
    def reward_func(cls, state: int, action: int, nstate: int) -> float:
        if nstate in cls.GOAL_STATES:
            return 0
        return -1


    def __init__(self, random_state=None):
        observation_space = Discrete(self.NSTATES)
        action_space = Discrete(3)
        maxsteps = 2 * self.NSTATES
        super().__init__(reward=self.reward_func,
                         transition=self.transition_func,
                         observation_space=observation_space,
                         action_space=action_space,
                         goal=self.goal_func,
                         maxsteps=maxsteps,
                         random_state=random_state)
