"""
This module defines the SLearner class. SLearner learns a policy over a
continuous state space defined by a set of state variables. For the initial
learning, however, it discretely samples state space to learn weights for a
functional approximation.

Because state-space is continuous, OFFLINE learning is not possible since it
cannot cache maximum q-values for each state reached during learning episodes.

The system is defined by a Simulator object (see linsim/simulate.py).

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
    from flearner import FLearner
except ImportError:
    from .flearner import FLearner



class SLearner(FLearner):
    """
    An SLearner learns policy from a Simulator over a continuous state-space.
    The space is descretely sampled for the learning process. All state/action
    representations are as vectors.

    Args:
        reward (func): A function that takes state, action, next state and
            returns the reward (float).
        simulator (Simulator): A Simulator instance that represents the
            environment.
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
        lrate (float): Learning rate for q-learning.
        discount (float): Discount factor for q-learning.
        exploration (float): Balance between exploring random states according
            to action selection policy, or exploiting already learned utilities
            to suggest max utility action. 1 means full exploration 0
            exploitation. 0.5 means half of each. Default is 0.
        policy (str): The action selection policy. Used durung learning/
            exploration to randomly select actions from a state. One of
            QLearner.[UNIFORM | GREEDY | SOFTMAX]. Default UNIFORM.
        depth (int): Max number of iterations in each learning episode. Defaults
            to number of states in stateconverter.
        steps (int): Number of steps (state transitions) to look ahead to
            calculate next estimate of value of state, action pair. Default=1.
        seed (int): A seed for all random number generation in instance. Default
            is None.
        duration (int): The duration for which a state/action is simulated to
            reach the next state. Defaults to simulator's timestep.

    Instance Attributes:
        goal (func): Takes a state number (int) and returns bool whether it is
            a goal state or not.
        mode/policy/lrate/discount/exploration/simulator/duration: Same as args.
        random (np.random.RandomState): A random number generator local to this
            instance.
    """

    def __init__(self, reward, simulator, stateconverter, actionconverter, func,
                 goal, lrate=0.25, discount=1, exploration=0, policy='uniform',
                 depth=None, steps=1, seed=None, duration=-1, **kwargs):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)

        self.simulator = simulator
        self.duration = duration if duration > 0 else simulator.timestep

        self.lrate = lrate
        self.discount = discount
        self.exploration = exploration
        self.depth = stateconverter.num_states if depth is None else depth
        self.steps = steps

        self.func = func
        self._reward = reward
        self.set_goal(goal)
        self.set_action_selection_policy(policy, mode=SLearner.ONLINE)

        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self._avecs = [avec for avec in self.actionconverter]
        self.funcdim = len(func(np.ones(len(stateconverter.flags)),
                                np.ones(len(actionconverter.flags))))
        self.weights = np.ones(self.funcdim)

    @property
    def num_states(self):
        return self.stateconverter.num_states

    @property
    def num_actions(self):
        return self.actionconverter.num_states


    def set_goal(self, goal):
        """
        Sets a function that checks if a state is a goal state or not.

        Args:
            goal (list/tuple/set/array/function): Goal state vectors.
                OR a function that accepts a state vector and returns true if
                goal.
        """
        if isinstance(goal, (np.ndarray, list, tuple, set)):
            # self._goals = set(goal)
            self.goal = lambda x: x in goal
        elif callable(goal):
            # self._goals = set([g for g in self.stateconverter if goal(g)])
            self.goal = goal
        else:
            raise TypeError('Provide goal as list/set/array/tuple/function.')


    def episodes(self, coverage=1., **kwargs):
        """
        Provides a sequence of states for learning episodes to start from.

        Args:
            coverage (float): Fraction of states to generate for episodes.
                Default= 1. Range [0, 1].

        Returns:
            A generator of of state vectors.
        """
        num = int(self.num_states * coverage)
        for _ in range(num):
            yield self.stateconverter.decode(self.random.choice(self.num_states))


    def reward(self, svec, avec, next_svec):
        return self._reward(svec, avec, next_svec)


    def next_state(self, svec, avec):
        return self.simulator.run(duration=self.duration, state=svec, action=avec)


    def next_action(self, svec):
        return self._avecs[super().next_action(svec)]


    def neighbours(self, svec):
        """
        Returns a list of state vectors adjacent to provided state.

        Args:
            svec (ndarray/list/tuple): The state vector.

        Returns:
            A list of adjacent state vectors.
        """
        return [self.next_state(svec, avec) for avec in self._avecs]


    def value(self, state):
        """
        The value of state i.e. the expected rewards by being greedy with
        the value function.

        Args:
            state (int/list/array): Index of current state in [r|q]matrix
                (row index).

        Returns:
            A tuple of (float, action vector) representing value and the next
            most rewarding action vector.
        """
        ans = super().value(state)
        return (ans[0], self._avecs[ans[1]])


    def qvalue(self, svec, avec=None):
        """
        The q-value of state, action pair.

        Args:
            svec (ndarray/list/tuple): Vector of state variables.
            avec (ndarray/list/tuple): Vector of action variables.

        Returns:
            The qvalue of state,action if action is specified. Else returns the
            qvalues of all actions from a state (array).
        """
        if avec is not None:
            return np.dot(self.weights, self.func(svec, avec))
        else:
            return np.array([np.dot(self.func(svec, a), self.weights)\
                    for a in self._avecs])


    def _update(self, svec, avec, error):
        """
        Updates weights given state, action, and error in current and next
        value estimate.

        Args:
            svec (ndarray/list/tuple): Vector of state variables.
            avec (ndarray/list/tuple): Vector of action variables.
            error (float): Error term (current value - next estimate)
        """
        self.weights -= self.lrate * error * self.func(svec, avec)
