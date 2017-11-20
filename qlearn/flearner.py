"""
This module implements the FLearner class. It uses temporal difference learning
on a descrete state/action space to generate a function approximation value of
each state in terms of state variables.

    Value(state, action) = f(state, action)

Where 'f' is of the form:

    f = function(state variables, action variables, weights)

The learner learns the values of weights that get the smallest errors during the
learning phase.

For example, if there are three state/action variables: x, y, z, the approximation
could be:

    f = (w1, w2, w3, w4) . (x, y, z, x*y, x^2) -- linear function approximation
Or:
    f = (sin(w1*x), y*z/w2)                    -- non-linear approximation

The weights are learned using gradient descent, therefore learning process is very
sensitive to the learning rate:

    weights[t+1] = -lrate * error * df/d weights[t]

The learned weights are then used to generate a policy:

    Policy(action | state) = max over a(Value(state, a) | a => all possible actions)

Having a function approximation instead of a matrix storing all of the state
space saves on space at the cost of accuracy.

Rewards and state transitions are stored descretely in matrices. For continuous
systems (with descrete actions), see SLearner.

All learners expose the following interface:

* Instantiation with relevant parameters any any number of positional and
    keyword arguments.
* reward(state, action, next_state, **kwargs) which returns the reward for taking an
    action from some state.
* next_state(state, action, **kwargs) which returns the next state based on the
    current state and action.
* neighbours (state) which returns states adjacent to provided state.
* value(state) which returns the utility of a state and the following action
    what leads to that utility.
* qvalue(state, action) which returns value of a state-action pair, or an array
    of values of all actions from a state if action is not specified.
* learn(episodes, actions, **kwargs) which runs over multiple episodes to populate
    a utility function or matrix.
* recommend(state, **kwargs) which recommends an action based on the learned values
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
        func (func): A function approximation for the value of a state/action.
            Returns the terms of the approximation as a numpy array. Signature:
                float = func(state_vec, action_vec, weights_vec)
            Where [state|action_weights]_vec are arrays. The returned array
            can be of any length, where each element is a combination of the
            state/action variables.
        dfunc (func): The derivative of func with respect to weights. Same
            input signature as func. Returns 'funcdim` elements in returned array.
        funcdim (int): The dimension of the weights to learn. Defaults to
            dimension of func.
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
        depth (int): Max number of iterations in each learning episode. Defaults
            to number of states.
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

    def __init__(self, rmatrix, stateconverter, actionconverter, goal, func,
                 funcdim, dfunc, tmatrix=None, lrate=0.25, discount=1, 
                 policy='uniform', mode='offline', depth=None,
                 steps=1, seed=None, stepsize=lambda x: 1, **kwargs):
        super().__init__(rmatrix, goal, tmatrix, lrate, discount,
                         policy, mode, depth, steps, seed, **kwargs)
        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self.funcdim = funcdim
        self.func = func
        self.dfunc = dfunc
        self.weights = np.ones(self.funcdim)
        self._avecs = [avec for avec in self.actionconverter]


    def value(self, state):
        """
        The value of state i.e. the expected rewards by being greedy with
        the value function.

        Args:
            state (int/list/array): Index of current state in [r|q]matrix
                (row index). Or the state vector.

        Returns:
            A tuple of a float representing value and the action index of the
            next most rewarding action.
        """
        if isinstance(state, (list, tuple, np.ndarray)):
            vals = [self.func(state, a, self.weights)\
                    for a in self._avecs]
        else:
            state_ = self.stateconverter.decode(state)
            vals = [self.func(state_, a, self.weights)\
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
            return self.func(svec, avec, self.weights)
        else:
            return np.array([self.func(svec, a, self.weights)\
                    for a in self._avecs])


    def update(self, state, action, error):
        """
        Updates weights given state, action, and error in current and next
        value estimate.

        Args:
            state (int): State number.
            action (int): Action number.
            error (float): Error term (current value - next estimate)
        """
        svec = self.stateconverter.decode(state)
        avec = self._avecs[action]
        self.weights -= self.lrate * error * self.dfunc(svec, avec, self.weights)


    def reset(self):
        """
        Resets weights to initial values.
        """
        self.weights = np.ones(self.funcdim)
