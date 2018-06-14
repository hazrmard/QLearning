from itertools import zip_longest
from typing import Generator, Tuple, Union, Callable, List

import numpy as np
from gym.core import Env, Space

from . import spaces
from .parameters import Schedule, evaluate_schedule_kwargs
from ..approximation import Approximator

UNIFORM = 'uniform'
GREEDY = 'greedy'
SOFTMAX = 'softmax'



class Agent:
    """
    A single-threaded agent that operates on continuous, discrete, and hybrid
    state and action spaces. Learning is episodic.

    Args:
    * env: The environment to operate on. Must be compatible with `gym.Env`.
    * value_function: An `Approximator` instance that learns to map `state, action`
    tuples to their values.
    * seed (int): A seed for all random number generation in instance. Default
    is None.

    Attributes:
    * next_action (Callable): A function that takes a state tuple and returns
    the action tuple for the next action.
    * eps_current (float): The current value of exploration rate (epsilon) during
    learning for each episode.
    * maximum (Callable): A function that takes the state tuple and returns the
    maximum value and corresponding action tuple from the value approximation.
    """

    def __init__(self, env: Env, value_function: Approximator, seed=None):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)

        self.env = env
        # a list of discrete actions. Continuous variables shown as None
        self.actions = list(spaces.enumerate_discrete_space(env.action_space))
        # partially-greedy exploration rate
        self.eps_curr = 0.
        # self.greedy_prob = greedy_prob
        # self.set_action_selection_policy(policy, greedy_prob=greedy_prob)

        self.value = value_function
        self.set_value_optimizer(self.env.action_space)


    def __str__(self):
        return self.__class__.__name__


    def set_value_optimizer(self, space: Space):
        """
        Sets the appropriate value maximization function depending on a discrete,
        continuous, hybrid action space.
        """
        cont = spaces.is_continuous(space)
        action_bounds = spaces.bounds(space)
        if all(cont):
            self.maximum = lambda s: spaces.max_continuous(self.value.predict,\
                                    action_bounds, s)
        
        elif any(cont):
            self.maximum = lambda s: spaces.max_hybrid(self.value.predict,\
                                    action_bounds, cont, s, self.actions)
        
        else:
            self.maximum = lambda s: spaces.max_discrete(self.value.predict,\
                                    self.actions, s)


    def set_action_selection_policy(self, policy: str, greedy_prob: float=None):
        """
        Sets a policy for selecting subsequent actions while in a learning
        episode.

        Args:
        * policy (str): One of QLearner.[UNIFORM | GREEDY | SOFTMAX].
        * greedy_prob (float): Probability of choosing action with highest utility [0, 1].
        """
        self.policy = policy
        
        if policy == UNIFORM:
            self.next_action = self._uniform_policy

        elif policy == GREEDY:
            self.next_action = self._greedy_policy

        elif policy == SOFTMAX:
            self.next_action = self._softmax_policy

        else:
            raise ValueError('Policy does not exist.')


    def episodes(self) -> Generator:
        """
        Provides a sequence of states for learning episodes to start from.

        Returns:
        * A generator of of state tuples.
        """
        while True:
            yield spaces.to_tuple(self.env.observation_space, self.env.reset())



    def learn(self, algorithm: Callable, episodes: int=100, policy: str=GREEDY,\
        epsilon: Schedule=Schedule(0,), **kwargs) -> List[List[float]]:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * algorithm: The learning function. Must have similar signature to functions
        in `qlearn.algorithms` package.
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * episolon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * kwargs: Any learning parameters required by the learning function.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * A List of:
            * Lists of rewards for each episode.
        """
        self.set_action_selection_policy(policy)
        histories = []
        for i in range(episodes):
            self.eps_curr = epsilon(i)  # greedy-rate for GREEDY policy
            kw = evaluate_schedule_kwargs(i, **kwargs)
            r = algorithm(self, **kw)
            histories.append(r)
        return histories


    def recommend(self, state: Tuple[Union[int, float]]):
        """
        Recommends an action based on the learned value function.
        Must be called after learn().

        Args:
        * state (int): Index of current state in [r|q]matrix.

        Returns:
        * Action to take (tuple).
        """
        value, action = self.maximum(state)
        return action


    def _uniform_policy(self, state: Tuple[Union[int, float]]):
        """
        Selects an action based on a uniform probability distribution.

        Args:
        * state (Tuple): State variable tuple.

        Returns:
        * The action tuple.
        """
        return spaces.to_tuple(self.env.action_space, self.env.action_space.sample())


    def _greedy_policy(self, state: Tuple[Union[int, float]]):
        """
        Select highest utility action with higher probability. Others are
        uniformly selected.

        Args:
        * state (tuple): The state variable.

        Returns:
        * The action tuple.
        """
        if self.random.uniform() < self.eps_curr:
            return self.recommend(state)
        else:
            return self._uniform_policy(state)


    def _softmax_policy(self, state: Tuple[Union[int, float]]):
    #TODO: implement for continuous action spaces
        """
        Selects actions with probability proportional to their utility in
        qmatrix[state,:]

        Args:
        * state (int): Index of current state.

        Returns:
        * Index of action in [r|q]matrix.
        """
        vals = np.array([self.value((*state, *a)) for a in self.actions])
        cumulative_utils = np.cumsum(vals - np.min(vals))
        random_num = self.random.rand() * cumulative_utils[-1]
        return self.actions[np.searchsorted(cumulative_utils, random_num)]


    def a_probs(self, state: Tuple[Union[int, float]]):
        """
        Calculates probability of taking all actions from a given state under an
        action selection policy.

        Args:
        * state (int/vector): Index of state in [r|q] matrix. Or if internal
        state representation is as vector, then list/tuple/array.

        Returns:
        * A numpy array of action probabilities.
        """
        # TODO: finish for continuous action spaces.
        if self.policy == UNIFORM:
            return np.ones(len(self.actions)) / len(self.actions)
        
        elif self.policy == GREEDY:
            over = zip_longest((state,), self.actions, fillvalue=state)
            value, action = self.maximum(state)
            probs = np.ones(len(self.actions)) * (1 - self.eps_curr) \
                   / (len(self.actions) - 1)
            probs[self.actions.index(action)] = self.eps_curr
            return probs
        
        elif self.policy == SOFTMAX:
            vals = np.array([self.value((*state, *a)) for a in self.actions])
            recentered = vals - np.min(vals)
            exp = np.exp(recentered)
            return exp / np.sum(exp)
