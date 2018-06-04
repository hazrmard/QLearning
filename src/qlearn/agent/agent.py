from itertools import zip_longest
from typing import Generator, Tuple

import numpy as np
from gym.core import Env

from . import helpers
from ..approximation import Approximator
from ..algorithms import td0

UNIFORM = 'uniform'
GREEDY = 'greedy'
SOFTMAX = 'softmax'


class Agent:
    """
    A single-thread q learner with a reward matrix and goals that terminate the
    learning process. Works in environments with discrete action spaces.

    Args:
    * discount (float): Discount factor for q-learning.
    * policy (str): The action selection policy. Used durung learning/
    exploration to randomly select actions from a state. One of
    QLearner.[UNIFORM | GREEDY | SOFTMAX]. Default UNIFORM.
    * seed (int): A seed for all random number generation in instance. Default
    is None.
    """

    def __init__(self, env: Env, value_function: Approximator, discount=1,
                 policy=UNIFORM, seed=None, greedy_prob=1.0):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)

        self.env = env
        # a list of discrete actions. Continuous variables shown as None
        self.actions = list(helpers.enumerate_discrete_space(env.action_space))
        # [(low, high)] bounds for each variable in action space
        self.action_bounds = helpers.bounds(self.env.action_space)
        # True for each continuous action variable
        self.actions_cont = helpers.is_continuous(self.env.action_space)
        self.value = value_function
        # action selection policy for learning/exploration
        self._policy = None
        # probability with which to select most valuable action for GREEDY policy
        self.greedy_prob = 0.

        self.set_action_selection_policy(policy, greedy_prob=greedy_prob)


    def __str__(self):
        return self.__class__.__name__\
               + ', Policy: ' + self.policy


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
            self._policy = self._uniform_policy

        elif policy == GREEDY:
            nactions = helpers.size_space(self.env.action_space)
            if greedy_prob is not None and nactions > 0:
                self.greedy_prob = greedy_prob - (1 - greedy_prob) / nactions 
                self._policy = self._greedy_policy
            elif greedy_prob is None:
                raise KeyError('"greedy_prob" keyword argument needed for GREEDY policy.')
            elif len(self.actions) == 0:
                raise NotImplementedError('Greedy policy not implemented for continuous actions.')

        elif policy == SOFTMAX:
            self._policy = self._softmax_policy

        else:
            raise ValueError('Policy does not exist.')


    def episodes(self) -> Generator:
        """
        Provides a sequence of states for learning episodes to start from.

        Returns:
        * A generator of of state tuples.
        """
        while True:
            yield helpers.to_tuple(self.env.observation_space, self.env.reset())


    def next_action(self, state: Tuple):
        """
        Provides a sequence of actions based on the action selection policy.

        Args:
        * state: The state tuple to take next action from.

        Returns:
        * The action tuple.
        """
        return self._policy(state)


    def learn(self, episodes=None, actions=(), **kwargs):
        """
        Begins learning procedure over all (state, action) pairs. Populates the
        Q matrix with utility for each (state, action).
        Implements the n-step Tree Backup algorithm by default. See algorithms
        module for more implementations.
        See Reinforcement Learning - an Introduction by Sutton/Barto (Ch. 7)

        Args:
            episodes (list/generator): States to begin learning episodes from.
                Defaults to self.episodes().
            OR
            actions (list/tuple): A list of actions to take for each starting state
                provided in episodes. Optional.
            
            **kwargs: Any learning parameters (lrate, depth, stepsize, mode, steps,
                discount, exploration) which are stored.
        Returns:
            A list of lists of states traversed for each episode.
        """        
        episodes = episodes if episodes is not None else self.episodes()

        histories = []
        actions = []
        for i, ep in enumerate(episodes):
            s, a = td0(self, ep, discount=kwargs.get('discount'))
            histories.append(s)
            actions.append(a)
        return histories, actions


    def recommend(self, state):
        """
        Recommends an action based on the learned q matrix and current state.
        Must be called after learn().

        Args:
        * state (int): Index of current state in [r|q]matrix.

        Returns:
        * Action to take (tuple).
        """
        value, action = helpers.maximum(self.value.predict, self.action_bounds, \
                                        self.actions_cont, state, self.actions)
        return action


    def _uniform_policy(self, state: Tuple):
        """
        Selects an action based on a uniform probability distribution.

        Args:
        * state (Tuple): State variable tuple.

        Returns:
        * The action tuple.
        """
        return helpers.to_tuple(self.env.action_space, self.env.action_space.sample())


    def _greedy_policy(self, state: Tuple):
        """
        Select highest utility action with higher probability. Others are
        uniformly selected.

        Args:
        * state (tuple): The state variable.

        Returns:
        * The action tuple.
        """
        if self.random.uniform() < self.greedy_prob:
            return self.recommend(state)
        else:
            return self._uniform_policy(state)


    def _softmax_policy(self, state: Tuple):
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


    def a_probs(self, state: Tuple):
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
            value, action = helpers.max_discrete(self.value, over)
            probs = np.ones(len(self.actions)) * (1 - self.greedy_prob) \
                   / (len(self.actions) - 1)
            probs[self.actions.index(action)] = self.greedy_prob
            return probs
        
        elif self.policy == SOFTMAX:
            vals = np.array([self.value((*state, *a)) for a in self.actions])
            recentered = vals - np.min(vals)
            exp = np.exp(recentered)
            return exp / np.sum(exp)
