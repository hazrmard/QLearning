"""
A discrete or bounded continuous environment specified with specified rewards
and transitions.
"""

from collections import OrderedDict
from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.random import RandomState
from gym import spaces
from gym.core import Env, Space



class Environment(Env):
    """
    Environment is a convenience wrapper for the openai gym's `Env` class. It
    encapsulates transition, reward, and goal functions into a cohesive object.
    The environment state is persistent - it remembers its last state from the
    previous call to `Environment.step()`.

    Args:
    * reward: A function that takes starting state, action, next state and returns
    a float representing the reward.
    * transition: A function that takes the current state and action and returns
    the next state tuple.
    * state_space: A `Space` object representing the range of values state variables
    can take.
    * action_space: A `Space` object representing the range of values actions
    can take.
    * random_state: An `int` or `np.random.RandomState` instance that is used
    to randomly sample actions/states. Defaults to `None`.

    Note: All spaces/actions are used in the same type as their spaces. For e.g
    a MultiDiscrete space will sample a tuple whereas a Box space will sample
    a numpy array.
    """


    def __init__(self, reward: Callable[[Any, Any, Any], float],
                transition: Callable[[Any, Any], Any],
                observation_space: Space, action_space: Space,
                goal: Callable[[Any], bool], maxsteps: int=np.inf,
                random_state: Union[int, RandomState]=None):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward = reward
        self.transition = transition
        self.goal = goal
        self.state = self.reset()
        self.maxsteps = maxsteps
        self.t = 0
        self.random = random_state if isinstance(random_state,\
                      np.random.RandomState) else np.random.RandomState(random_state)
        spaces.np_random = self.random


    def step(self, action, state=None) -> Tuple[Tuple, float, bool, object]:
        """
        Given an action, compute the next state and reward of the environment.

        Args:
        * action (Tuple/int): A tuple specifying the action to take.
        * state (Tuple): The state to take the action from - optional.

        Returns a tuple of:
        * new state (Tuple), reward (float), terminal state (bool), misc info
        """
        self.t += 1
        state = self.state if state is None else state
        nstate = self.transition(state, action)
        reward = self.reward(self.state, action, nstate)
        self.state = nstate
        done = (self.t == self.maxsteps) or self.goal(self.state)
        return self.state, reward, done, None


    def reset(self) -> Union[Tuple, np.ndarray, int, float, OrderedDict]:
        """
        Return the environment to a random initial state.

        Returns:
        * The new initial state of the environment. Same type as
        `self.observation_space.sample()`.
        """
        self.t = 0
        self.state = self.observation_space.sample()
        return self.state


    def render(self):
        pass
