"""
The base `Agent` class for Generalized Policy Iteration. Relies on a value
function to derive policy and take actions. Implements various action selection
policies.
"""
from itertools import zip_longest
from typing import Generator, Tuple, Union, Callable

import numpy as np
from numpy.random import RandomState
from gym.core import Env, Space

from ...helpers import spaces, maximum
from ...helpers.parameters import Schedule, evaluate_schedule_kwargs
from ..memory import Memory

UNIFORM = 'uniform'
GREEDY = 'greedy'
SOFTMAX = 'softmax'



class Agent:
    """
    A single-threaded agent that operates on continuous, discrete, and hybrid
    state and action spaces. Learning is episodic, based on generalized policy
    iteration (GPI), and uses experience-replay.

    Args:
    * env: The environment to operate on. Must be compatible with `gym.Env`.
    * value_function: An `Approximator` instance that learns to map `state, action`
    tuples to their values.
    * random_state: Integer seed or `np.random.RandomState` instance.

    Attributes:
    * next_action (Callable): A function that takes a state tuple and returns
    the action tuple for the next action.
    * eps_current (float): The current value of exploration rate (epsilon) during
    learning for each episode.
    * maximum (Callable): A function that takes the state tuple and returns the
    maximum value and corresponding action tuple from the value approximation.
    """

    def __init__(self, env: Env, value_function: 'Approximator',\
        random_state: Union[int, RandomState] = None):
        self.random = random_state if isinstance(random_state, RandomState)\
                      else RandomState(random_state)
        self.env = env
        # a list of discrete actions. Continuous variables shown as None
        self.actions = list(spaces.enumerate_discrete_space(env.action_space))
        # partially-greedy exploration rate
        self.eps_curr = 0.

        self.value = value_function
        self.set_value_optimizer(self.env.action_space)


    def __str__(self):
        return self.__class__.__name__


    def set_value_optimizer(self, space: Space):
        """
        Sets the appropriate value maximization function depending on a discrete,
        continuous, hybrid action space.
        """
        # TODO: Allow for discrete maximization over continuous spaces in case
        # the learned value function is discrete/ does not have a defined minimum.
        cont = spaces.is_continuous(space)
        action_bounds = spaces.bounds(space)
        if all(cont):
            self.maximum = lambda s: maximum.max_continuous(self.value.predict,\
                                    action_bounds, s)
        
        elif any(cont):
            self.maximum = lambda s: maximum.max_hybrid(self.value.predict,\
                                    action_bounds, cont, s, self.actions)
        
        else:
            self.maximum = lambda s: maximum.max_discrete(self.value.predict,\
                                    self.actions, s)


    def set_action_selection_policy(self, policy: str):
        """
        Sets a policy for selecting subsequent actions while in a learning
        episode.

        Args:
        * policy (str): One of Agent.[UNIFORM | GREEDY | SOFTMAX].
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
        epsilon: Schedule=Schedule(0,), memsize: int=1, batchsize: int=1,\
        **kwargs) -> np.ndarray:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * algorithm: The learning function. Must have similar signature to functions
        in `qlearn.algorithms` package.
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * episilon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * memsize: Size of experience memory. Default 1 most recent observation.
        * batchsize: Number of past experiences to replay. Default 1.
        * kwargs: Any learning parameters required by the learning function.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * A List of average rewards for each episode.
        """
        memory = Memory(memsize=memsize, batchsize=batchsize) # experience-replay
        self.set_action_selection_policy(policy)
        histories = np.zeros(episodes)  # history of rewards for each episode
        epsilon = Schedule(epsilon) if not isinstance(epsilon, Schedule) else epsilon

        for i in range(episodes):
            self.eps_curr = epsilon(i)  # greedy-rate for GREEDY policy
            kw = evaluate_schedule_kwargs(i, **kwargs)
            r = algorithm(self, memory=memory, **kw)
            histories[i] = r

        return histories


    def recommend(self, state: Tuple[Union[int, float]]) -> Tuple[Union[int, float]]:
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
        if self.random.uniform() >= self.eps_curr:
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
    # TODO: implement for continuous action spaces
        """
        Calculates probability of taking all actions from a given state under an
        action selection policy.

        Args:
        * state (int/vector): Index of state in [r|q] matrix. Or if internal
        state representation is as vector, then list/tuple/array.

        Returns:
        * A numpy array of action probabilities.
        """
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
