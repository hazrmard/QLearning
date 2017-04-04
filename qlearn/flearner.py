"""
This module implements the FLearner class. It uses temporal difference learning
on a descrete state/action space to generate a function approximation value of
each state in terms of state variables.

    Value(state, action) = f(state, action)

Where 'f' is of the form:

    f = weights . (a vector of state/action variable combinations)

For example, if there are three state/action variables: x, y, z, the approximation
could be:

    f = weights . (x, y, z, x*y, x^2)

Where FLearner learns weights. Here, the state+action space is 3D, but 'f' is 5D.
The weights are learned using gradient descent, therefore learning process is very
sensitive to the learning rate.

The learned weights are then used to generate a policy:

    Policy(s' | state) = max(Value(s' | state) | s' => all reachable states)

Having a function approximation instead of a matrix storing all of the state
space saves on space at the cost of accuracy.
"""


import numpy as np
try:
    from qlearner import QLearner
    from linsim import FlagGenerator
except ImportError:
    from .qlearner import QLearner
    from .linsim import FlagGenerator



class FLearner(QLearner):
    """
    A learner that uses a reward and transition matrix to approximate a function
    for the value of each state/action pair.

    Args:
        rmatrix (ndarray/str): The reward matrix of [n states x m actions]. OR
            filepath to space delimited rmatrix file. Element [n, m] in the
            matrix represents the reward for taking action m from state n.
        stateconverter (FlagGenerator): A FlagGenerator instance that can
            decode state number into state vectors and encode the reverse. For
            e.g if the state is defined by x,y coords it can encode (x, y) into
            a row index for the rmatrix, and decode the index(state number) into
            (x, y).
        actionconverter (FlagGenerator): Same as state converter but for actions.
        func (func): A linear function approximation for the value function.
            Returns the terms of the approximation as a numpy array. Signature:
                func(state_vec, action_vec)
            Where [state|action]_vec is a list of state variables. The returned array
            can be of any length, where each element is a combination of the
            state/action variables.
        goal (list/tuple/set/array/function): Indices of goal states in rmatrix
            OR a function that accepts a state index and returns true if goal.
        tmatrix (ndarray/str): A transition matrix of [n states x m actions], OR
            filepath to space delimited tmatrix. Where tmatrix[state, action]
            contains index of next state. If None, then rmatrix must be square
            of [n states x n states] i.e. no actions but direct state transitions.
        lrate (float): Learning rate for q-learning.
        discount (float): Discount factor for q-learning.
        exploration (float): Balance between exploring random states according
            to action selection policy, or exploiting already learned utilities
            to suggest max utility action. 1 means full exploration 0
            exploitation. 0.5 means half of each. Default is 0.
        policy (int): The action selection policy. Used durung learning/
            exploration to randomly select actions from a state. One of
            QLearner.[UNIFORM | GREEDY | SOFTMAX]. Default UNIFORM.
        mode (int): One of QLearner.[OFFLINE | ONLINE]. Offline updates action
            selection policy each learning episode. Online updates at every
            state/action inside the learning episode. Default OFFLINE (faster).
        seed (int): A seed for all random number generation in instance. Default
            is None.

    Instance Attributes:
        goal (func): Takes a state number (int) and returns bool whether it is
            a goal state or not.
        mode/policy/lrate/discount/exploration/rmatrix/tmatrix: Same as args.
        random (np.random.RandomState): A random number generator local to this
            instance.
    """

    def __init__(self, rmatrix, stateconverter, actionconverter, func, goal,
                 tmatrix=None, lrate=0.25, discount=1, exploration=0,
                 policy=0, mode=0, seed=None, **kwargs):
        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self._avecs = [actionconverter.decode(a) for a in range(actionconverter.states)]
        self.funcdim = len(func(np.ones(len(stateconverter.flags)),
                                np.ones(len(actionconverter.flags))))
        self.func = func
        self.weights = np.ones(self.funcdim)
        super().__init__(rmatrix, goal, tmatrix, lrate, discount, exploration,\
                         policy, mode, seed, **kwargs)


    def value(self, state):
        """
        The value of state i.e. the expected discounted rewards.

        Args:
            state (int/list/array): Index of current state in [r|q]matrix
                (row index).

        Returns:
            A tuple of a float representing value and the action index.
        """
        if isinstance(state, (list, tuple, np.ndarray)):
            state_ = self.stateconverter.encode(state)
            vals = [np.dot(self.func(state, a), self.weights)\
                    for a in self._avecs]
        else:
            state_ = self.stateconverter.decode(state)
            vals = [np.dot(self.func(state_, a), self.weights)\
                    for a in self._avecs]
        action = np.argmax(vals)
        return (vals[action], action)


    def error(self, state, action):
        """
        Returns the error in the last value estimate vs. the current value
        estimate using the function approximation.

        Args:
            state (int): Index of current state in [r|q]matrix (row index).
            action (int): Index of action to be taken from state (column index).

        Returns:
           A tuple of a float containing the error, the next state index,
           the current state vector, the current action vector.
        """
        next_state = self.next_state(state, action)
        avec = self.actionconverter.decode(action)
        svec = self.stateconverter.decode(state)
        nsvec = self.stateconverter.decode(next_state)
        return (self.lrate*(self.reward(state, action, next_state)
                            + self.discount*self.value(nsvec)[0]
                            - np.dot(self.func(svec, avec), self.weights)),
                next_state,
                svec,
                avec)


    def learn(self, coverage=1., ep_mode=None):
        """
        Updates weights for function approximation for state values over
        multiple episodes covering the state space.

        Args:
            coverage (float): Fraction of total states to start episodes from.
                Default=1 i.e. all state are covered by episodes().
            ep_mode (str): Order in which to iterate through states. See
                episodes() mode argument.
        """
        for state in self.episodes(coverage=coverage, mode=ep_mode):
            limit = 0
            if self.mode == QLearner.OFFLINE:
                self._update_policy()
            # The limit variable keeps the number of iterations in check. They
            # should not exceed the number of states in the system.
            while not self.goal(state) and limit < self.num_states:
                limit += 1
                action = self.next_action(state)
                err, state, curr_svec, curr_avec = self.error(state, action)
                self.weights += err * self.func(curr_svec, curr_avec)


    def recommend(self, state):
        """
        Recommends an action based on the learned q matrix and current state.
        Must be called after learn(). Either recommends an exploratory action
        or an action with the highest utility according to self.exploration
        setting (1 to 0).

        Args:
            state (int): Index of current state in [r|q]matrix.

        Returns:
            Index of action to take (column) in [r|q]matrix.
        """
        if self.random.rand() < self.exploration:
            # explore
            action = self.next_action(state)
            err, _, curr_svec, curr_avec = self.error(state, action)
            self.weights += err * self.func(curr_svec, curr_avec)
            return action
        else:
            # exploit
            svec = self.stateconverter.decode(state)
            vals = [np.dot(self.func(svec, a), self.weights) for a in self._avecs]
            return np.argmax(vals)


    def reset(self):
        """
        Resets weights to initial values.
        """
        self.weights = np.zeros(self.funcdim)
