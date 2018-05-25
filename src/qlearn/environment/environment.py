"""
A discrete or bounded continuous environment specified with specified rewards
and transitions.
"""

from typing import Union, Tuple, Set, Dict, Callable
import numpy as np
from gym.core import Env, Space


class Environment(Env):
    """
    Environment is a convenience wrapper for the openai gym's `Env` class. It
    encapsulates transition, reward, and goal functions into a cohesive object.

    Args:
    * reward: A function that takes starting state, action, next state and returns
    a float representing the reward.
    * transition: A function that takes the current state and action and returns
    the next state.
    * state_space: A `Space` object representing the range of values state variables
    can take.
    * action_space: A `Space` object representign the range of values actions
    can take.

    Note: All spaces/actions are used as tuples.
    """


    def __init__(self, reward: Callable[[Tuple, Tuple, Tuple], float],\
                transition: Callable[[Tuple, Tuple], Tuple],\
                state_space: Space, action_space: Space, goal: Callable[[Tuple], bool]):
        super().__init__()
        self.action_space = action_space
        self.observation_space = state_space
        self.reward = reward
        self.transition = transition
        self.goal = goal
        self.state = self.reset()


    def step(self, action) -> Tuple[Tuple, float, bool, object]:
        """
        Given an action, compute the next state and reward of the environment.

        Args:
        * action (Tuple): A tuple specifying the action to take.

        Returns a tuple of:
        * new state (Tuple), reward (float), terminal state (bool), misc info
        """
        action = tuple(action)
        nstate = tuple(self.transition(self.state, action))
        reward = self.reward(self.state, action, nstate)
        self.state = nstate
        done = self.goal(self.state)
        return self.state, reward, done, None


    def reset(self) -> Tuple:
        """
        Return the environment to its initial state.

        Returns:
        * The new initial state (Tuple) of the environment.
        """
        self.state = tuple(self.observation_space.sample())
        return self.state


    def render(self):
        pass