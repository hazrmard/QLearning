"""
This module defines the QLearner class that learns behaviour from a Reward
matrix.
"""

import numpy as np
import utils


class QLearner:
    """
    A single-thread q learner with a reward matrix and goals that terminate the
    learning process.

    Args:
        rmatrix (ndarray/str): The reward matrix of [n states x m actions]. OR
            filepath to space delimited rmatrix file.
        goal (list/tuple/set/array/function): Indices of goal states in rmatrix
            OR a function that accepts a state index and returns true if goal.
        tmatrix (ndarray/str): A transition matrix of [n states x m actions], OR
            filepath to space delimited tmatrix. Where tmatrix[state, action]
            contains index of next state. If None, then rmatrix must be square
            of [n states x n states] i.e. no actions but direct state transitions.
        lrate (float): Learning rate for q-learning.
        discount (float): Discount factor for q-learning.
        policy (int): One of QLearner.[UNIFORM | GREEDY | SOFTMAX]. Default
            UNIFORM.
        mode (int): One of QLearner.[OFFLINE | ONLINE]. Default OFFLINE.
    """

    UNIFORM = 0
    GREEDY = 1
    SOFTMAX = 2

    OFFLINE = 0
    ONLINE = 1

    def __init__(self, rmatrix, goal, tmatrix=None, lrate=1, discount=1, policy=0,
                 mode=0, **kwargs):
        """
        Attributes:
            _policy (func): a function reference to self.[_unifotm | _greedy |
                _softmax]_policy
            mode/policy/lrate/discount/rmatrix/tmatrix/goal: Same as args.
            _action_param: A dict of helper values for GREEDY | SOFTMAX calcs.
            next_state (func): Returns next state given current state, action.
        """
        self.set_reward_matrix(rmatrix)
        self.set_transition_matrix(tmatrix)
        self.qmatrix = np.zeros_like(self.rmatrix)
        self.set_goal(goal)
        self.lrate = lrate
        self.discount = discount
        self._action_param = {}   # helper parameter for GREEDY/SOFTMAX policies
        self.set_action_selection_policy(policy, mode, **kwargs)


    def set_action_selection_policy(self, policy, mode=0, **kwargs):
        """
        Sets a policy for selecting subsequent actions while in a learning
        episode.

        Args:
            policy (int): One of QLearner.[UNIFORM | GREEDY | SOFTMAX].
            mode (int): One of QLearner.[OFFLINE | ONLINE]. Default OFFLINE.
            max_prob (float): Probability of choosing action with highest utility [0, 1).
        """
        self._action_param = {}
        self.mode = mode
        self.policy = policy
        if policy == QLearner.UNIFORM:
            self._policy = self._uniform_policy

        elif policy == QLearner.GREEDY:
            if 'max_prob' in kwargs:
                self._action_param['max_prob'] = kwargs['max_prob'] \
                                     - (1 - kwargs['max_prob']) / self.rmatrix.shape[1]
                self._policy = self._greedy_policy
            else:
                raise KeyError('"max_prob" keyword argument needed for GREEDY policy.')

        elif policy == QLearner.SOFTMAX:
            self._policy = self._softmax_policy

        else:
            raise ValueError('Policy does not exist.')


    def set_reward_matrix(self, rmatrix):
        """
        Sets the reward matrix for the QLearner instance.

        Args:
            rmatrix (ndarray): The reward matrix of [n states x m actions]. OR
                square matrix of [n states x n states] where each element is
                the reward for n->m transition.
        """
        if isinstance(rmatrix, np.ndarray):
            self.rmatrix = rmatrix
        elif isinstance(rmatrix, str):
            self.rmatrix = utils.read_matrix(rmatrix)
        else:
            raise TypeError('Either provide filename or ndarray for R matrix.')


    def set_goal(self, goal):
        """
        Sets a function that checks if a state is a goal state or not.

        Args:
            goal (list/tuple/set/array/function): Indices of goal states in
            rmatrix OR a function that accepts a state index and returns true if
            goal.
        """
        if isinstance(goal, (np.ndarray, list, tuple, set)):
            self.goal = lambda x: x in goal
        elif callable(goal):
            self.goal = goal
        else:
            raise TypeError('Provide goal as list/set/array/tuple/function.')


    def set_transition_matrix(self, tmatrix):
        """
        Sets the transition matrix in case of state-action-state transitions
        as opposed to state-state transitions (which only require a nxn matrix).

        Args:
            tmatrix (ndarray/str): A transition matrix of [n states x m actions]
                OR a filepath to the whitespace delimited tmatrix file.
            where tmatrix[state, action] contains index of next state.
        """
        if tmatrix is not None:
            if isinstance(tmatrix, str):    # if filepath, read file to matrix
                tmatrix = utils.read_matrix(tmatrix)
            elif not isinstance(tmatrix, np.ndarray): # if not filepath, must be array
                raise TypeError('tmatrix should be ndarray or filepath string.')
            if tmatrix.shape != self.rmatrix.shape:
                raise ValueError('Transition and R matrix must have same shape.')
            if tmatrix.dtype != int:
                raise TypeError('Transition matrix must have integer contents.')
            self.tmatrix = tmatrix
            self.next_state = lambda s, a: self.tmatrix[s, a]
        else:
            rows, cols = self.rmatrix.shape
            if rows != cols:
                raise ValueError('R matrix must be square if no transition matrix.')
            self.next_state = lambda s, a: a


    def episodes(self):
        """
        Provides a sequence of states for learning episodes to start from.

        Returns:
            A generator of of state indices.
        """
        for i in range(len(self.rmatrix)):
            yield i


    def next_action(self, state):
        """
        Provides a sequence of actions based on the action selection policy.

        Args:
            state (int): Index of current state in [r|q]matrix.

        Returns:
            An index for the [q|r]matrix (column).
        """
        return self._policy(state)


    def utility(self, state, action):
        """
        Computes the utility of a proposed action from a state based on
        temporal difference learning.

        Args:
            state (int): Index of current state in [r|q]matrix (row index).
            action (int): Index of action to be taken from state (column index).

        Returns:
            A tuple of a float containing the updated utility for
            qmatrix(state, action) and the next state.
        """
        next_state = self.next_state(state, action)
        return (self.qmatrix[state, action] \
                + self.lrate*(self.rmatrix[state, action]
                              + self.discount*np.max(self.qmatrix[next_state])
                              - self.qmatrix[state, action]),
                next_state)


    def learn(self):
        """
        Begins learning procedure over all (state, action) pairs. Populates the
        Q matrix with utility for each (state, action).
        """
        for state in self.episodes():
            if self.mode == QLearner.OFFLINE:
                self._update_policy()
            while not self.goal(state):
                action = self.next_action(state)
                self.qmatrix[state, action], state = self.utility(state, action)


    def recommend(self, state):
        """
        Recommends an action based on the learned q matrix and current state.
        Must be called after learn().

        Args:
            state (int): Index of current state in [r|q]matrix.

        Returns:
            Index of action to take (column) in [r|q]matrix.
        """
        return np.argmax(self.qmatrix[state])


    def _uniform_policy(self, state):
        """
        Selects an action based on a uniform probability distribution.

        Args:
            state (int): Index of current state.

        Returns:
            Index of action in [r|q]matrix.
        """
        return np.random.randint(self.qmatrix.shape[1])


    def _greedy_policy(self, state):
        """
        Select highest utility action with higher probability. Others are
        uniformly selected.

        Args:
            state (int): Index of current state.

        Returns:
            Index of action in [r|q]matrix.
        """
        if self.mode == QLearner.ONLINE:
            if np.random.uniform() < self._action_param['max_prob']:
                return np.argmax(self.qmatrix[state])
            else:
                return np.random.randint(self.qmatrix.shape[1])
        elif self.mode == QLearner.OFFLINE:
            if np.random.uniform() < self._action_param['max_prob']:
                return self._action_param['max_util_indices'][state]
            else:
                return np.random.randint(self.qmatrix.shape[1])


    def _softmax_policy(self, state):
        """
        Selects actions with probability proportional to their utility in
        qmatrix[state,:]

        Args:
            state (int): Index of current state.

        Returns:
            Index of action in [r|q]matrix.
        """
        if self.mode == QLearner.ONLINE:
            cumulative_utils = np.cumsum(self.qmatrix[state])
            random_num = np.random.rand() * cumulative_utils[-1]
            return np.searchsorted(cumulative_utils, random_num)
        elif self.mode == QLearner.OFFLINE:
            random_num = np.random.rand()  \
                        * self._action_param['cumulative_utils'][state][-1]
            return np.searchsorted(self._action_param['cumulative_utils'][state],\
                                random_num)


    def _update_policy(self):
        """
        Updates OFFLINE [SOFTMAX | GREEDY] policy every episode by updating
        how new actions are suggested based on current utility.
        """
        if self.policy == QLearner.GREEDY:
            # max_util_indices is a list of action indices (column #s) with the
            # highest q value for each state. Used to generate random numbers
            # based on the greedy policy.
            self._action_param['max_util_indices'] = np.argmax(self.qmatrix, axis=1)
        elif self.policy == QLearner.SOFTMAX:
            # cumulative_utils is the cumulative sum of the action q values for
            # each state. This is used to generate a random number based on the
            # relative utility of each action in a state.
            self._action_param['cumulative_utils'] = np.cumsum(self.qmatrix, axis=1)
