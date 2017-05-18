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

Rewards and state transitions are stored descretely in matrices. For continuous
systems (with descrete actions), see SLearner.

All learners expose the following interface:

* Instantiation with relevant parameters any any number of positional and
    keyword arguments.
* reward(state, action, next_state) which returns the reward for taking an
    action from some state.
* next_state(state, action) which returns the next state based on the current
    state and action.
* value(state) which returns the utility of a state and the following action
    what leads to that utility.
* qvalue(state, action) which returns value of a state-action pair, or an array
    of values of all actions from a state if action is not specified.
* learn() which runs over multiple episodes to populate a utility function
    or matrix.
* recommend(state) which recommends an action based on the learned values
    depending on the exploration vs. exploitation setting of the learner.
* reset() which returns the value function/matrix to its initial state while
    keeping any learning parameters provided at instantiation.
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
        policy (str): The action selection policy. Used durung learning/
            exploration to randomly select actions from a state. One of
            QLearner.[UNIFORM | GREEDY | SOFTMAX]. Default UNIFORM.
        mode (str): One of QLearner.[OFFLINE | ONLINE]. Offline updates action
            selection policy each learning episode. Online updates at every
            state/action inside the learning episode. Default OFFLINE (faster).
        steps (int): Number of steps (state transitions) to look ahead to
            calculate next estimate of value of state, action pair. Default=1.
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
                 policy='uniform', mode='offline', steps=1,
                 seed=None, **kwargs):
        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self.funcdim = len(func(np.ones(len(stateconverter.flags)),
                                np.ones(len(actionconverter.flags))))
        self.func = func
        self.weights = np.ones(self.funcdim)
        super().__init__(rmatrix, goal, tmatrix, lrate, discount, exploration,\
                         policy, mode, steps, seed, **kwargs)
        self._avecs = [actionconverter.decode(a) for a in range(actionconverter.states)]


    def value(self, state):
        """
        The value of state i.e. the expected rewards by being greedy with
        the value function.

        Args:
            state (int/list/array): Index of current state in [r|q]matrix
                (row index).

        Returns:
            A tuple of a float representing value and the action index of the
            next most rewarding action.
        """
        if isinstance(state, (list, tuple, np.ndarray)):
            # state_ = self.stateconverter.encode(state)
            vals = [np.dot(self.func(state, a), self.weights)\
                    for a in self._avecs]
        else:
            state_ = self.stateconverter.decode(state)
            vals = [np.dot(self.func(state_, a), self.weights)\
                    for a in self._avecs]
        action = np.argmax(vals)
        return (vals[action], action)


    def qvalue(self, state, action=None):
        """
        The q-value of state, action pair.

        Args:
            state (int): Index of current state in [r|q]matrix (row index).
            action (int): Index of action to be taken from state (column index).

        Returns:
            The qvalue of state,action if action is specified. Else returns the
            qvalues of all actions from a state (array).
        """
        svec = self.stateconverter.decode(state)
        if action is not None:
            avec = self.actionconverter.decode(action)
            return np.dot(self.weights, self.func(svec, avec))
        else:
            return np.array([np.dot(self.func(svec, a), self.weights)\
                    for a in self._avecs])


    def _update(self, state, action, error):
        """
        Updates weights given state, action, and error in current and next
        value estimate.

        Args:
            state (int): State number.
            action (int): Action number.
            error (float): Error term (current value - next estimate)
        """
        svec = self.stateconverter.decode(state)
        avec = self.actionconverter.decode(action)
        self.weights -= self.lrate * error * self.func(svec, avec)


    def __error(self, state, action):
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
        vals = [np.dot(self.func(nsvec, a), self.weights) for a in self._avecs]
        return (-(self.reward(state, action, next_state)
                  + self.discount*np.max(vals)
                  - np.dot(self.func(svec, avec), self.weights)),
                next_state,
                svec,
                avec)


    def __learn(self, coverage=1., ep_mode=None, state_action=()):
        """
        Updates weights for function approximation for state values over
        multiple episodes covering the state space.

        Args:
            coverage (float): Fraction of total states to start episodes from.
                Default=1 i.e. all state are covered by episodes().
            ep_mode (str): Order in which to iterate through states. See
                episodes() mode argument.
            OR
            state_action (tuple): A tuple of state/action to learn from instead
                of multiple episodes that go all the way to a terminal state.
                Used by recommend() when learning from a single action.
        """
        episodes = [state_action[0]] if len(state_action) > 0 else\
                            self.episodes(coverage=coverage, mode=ep_mode)
        limit = 1 if len(state_action) > 0 else self.num_states

        for state in episodes:
            iterations = 0
            if self.mode == QLearner.OFFLINE:
                self._update_policy()
            # The limit variable keeps the number of iterations in check. They
            # should not exceed the number of states in the system.
            while not self.goal(state) and iterations < limit:
                limit += 1
                action = self.next_action(state)
                err, state, curr_svec, curr_avec = self.error(state, action)
                # self.weights += err * self.func(curr_svec, curr_avec)
                self._update(curr_svec, curr_avec, err)


    def reset(self):
        """
        Resets weights to initial values.
        """
        self.weights = np.ones(self.funcdim)
