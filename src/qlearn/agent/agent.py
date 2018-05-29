from typing import Generator
from itertools import zip_longest
import numpy as np
from gym.core import Env
from ..algorithms import variablenstep
from .helpers import enumerate_discrete_space, max_discrete

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
    * depth (int): Max number of iterations in each learning episode. Defaults
    to number of states.
    * steps (int): Number of steps (state transitions) to look ahead to
    calculate next estimate of value of state, action pair. Default=1.
    * stepsize (func): A function that accepts a state and returns a
    number representing the step size of the next action. The stepsize is
    forwarded as a 'stepsize' keyword argument to self.next_state. Used
    by SLearner for variable simulation times. Optional. Can be used
    to convey other information to an overridden next_state function.
    * seed (int): A seed for all random number generation in instance. Default
    is None.
    """

    def __init__(self, env: Env, value_function, discount=1,
                 policy=UNIFORM, depth=None,
                 steps=1, seed=None, stepsize=lambda x:1, **kwargs):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)

        self.env = env
        # a list of discrete actions. Empty if actions continuous.
        self.actions = enumerate_discrete_space(env.action_space)
        self.value = value_function
        # action selection policy for learning/exploration
        self._policy = None

        self.depth = depth
        self.steps = steps
        self.discount = discount
        self.stepsize = stepsize

        self._greedy_prob = 0.

        self.set_action_selection_policy(policy, max_prob=kwargs.get('max_prob'))


    def __str__(self):
        return self.__class__.__name__\
               + ', Policy: ' + self.policy


    def set_action_selection_policy(self, policy, max_prob=None):
        """
        Sets a policy for selecting subsequent actions while in a learning
        episode.

        Args:
            policy (str): One of QLearner.[UNIFORM | GREEDY | SOFTMAX].
            max_prob (float): Probability of choosing action with highest utility [0, 1).
        """
        self.policy = policy
        
        if policy == UNIFORM:
            self._policy = self._uniform_policy

        elif policy == GREEDY:
            if max_prob is not None and len(self.actions) > 0:
                self._greedy_prob = max_prob \
                                     - (1 - max_prob) / len(self.actions)
                self._policy = self._greedy_policy
            elif max_prob is None:
                raise KeyError('"max_prob" keyword argument needed for GREEDY policy.')
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
            yield tuple(self.env.reset())


    def next_action(self, state):
        """
        Provides a sequence of actions based on the action selection policy.

        Args:
            state (int): Index of current state in [r|q]matrix.

        Returns:
            An index for the [q|r]matrix (column).
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
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        
        episodes = episodes if episodes is not None else self.episodes()

        histories = []
        actions = []
        for i, pair in enumerate(zip_longest(episodes, actions)):
            states, actions = variablenstep(self, state=pair[0], action=pair[1])
            histories.append(states)
            actions.append(actions)
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
        over = zip_longest((state,), self.actions, fillvalue=state)
        value, action = max_discrete(self.value, over)
        return action


    def _uniform_policy(self, state):
        """
        Selects an action based on a uniform probability distribution.

        Args:
        * state (Tuple): State variable tuple.

        Returns:
        * The action tuple.
        """
        return tuple(self.env.action_space.sample())


    def _greedy_policy(self, state):
        """
        Select highest utility action with higher probability. Others are
        uniformly selected.

        Args:
        * state (tuple): The state variable.

        Returns:
        * The action tuple.
        """
        if self.random.uniform() < self._greedy_prob:
            return self.recommend(state)
        else:
            return tuple(self.env.action_space.sample())


    def _softmax_policy(self, state):
        """
        Selects actions with probability proportional to their utility in
        qmatrix[state,:]

        Args:
            state (int): Index of current state.

        Returns:
            Index of action in [r|q]matrix.
        """
        vals = np.array([self.value((*state, *a)) for a in self.actions])
        cumulative_utils = np.cumsum(vals - np.min(vals))
        random_num = self.random.rand() * cumulative_utils[-1]
        return self.actions[np.searchsorted(cumulative_utils, random_num)]


    def a_probs(self, state):
        """
        Calculates probability of taking all actions from a given state under an
        action selection policy.

        Args:
            state (int/vector): Index of state in [r|q] matrix. Or if internal
                state representation is as vector, then list/tuple/array.

        Returns:
            A numpy array of action probabilities.
        """
        if self.policy == UNIFORM:
            return np.ones(len(self.actions)) / len(self.actions)
        
        elif self.policy == GREEDY:
            over = zip_longest((state,), self.actions, fillvalue=state)
            value, action = max_discrete(self.value, over)
            probs = np.ones(len(self.actions)) * (1 - self._greedy_prob) \
                   / (len(self.actions) - 1)
            probs[self.actions.index(action)] = self._greedy_prob
            return probs
        
        elif self.policy == SOFTMAX:
            vals = np.array([self.value((*state, *a)) for a in self.actions])
            recentered = vals - np.min(vals)
            exp = np.exp(recentered)
            return exp / np.sum(exp)

