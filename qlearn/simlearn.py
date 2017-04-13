"""
This module defines the SimLearner class. SimLearner learns a policy over a
continuous state space defined by a set of state variables. For the initial
learning, however, it discretely samples state space to learn weights for a
functional approximation.

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
* learn() which runs over multiple episodes to populate a utility function
    or matrix.
* recommend(state) which recommends an action based on the learned values
    depending on the exploration vs. exploitation setting of the learner.
* reset() which returns the value function/matrix to its initial state while
    keeping any learning parameters provided at instantiation.
"""

import numpy as np



class SimLearner:

    def __init__(self, reward, simulator, stateconverter, actionconverter, func,
                 goal, lrate=0.25, discount=1, exploration=0, seed=None,
                 duration=-1, **kwargs):
        if seed is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(seed)
        self.simulator = simulator
        self.duration = duration if duration > 0 else simulator.timestep
        self.lrate = lrate
        self.discount = discount
        self.exploration = exploration
        self.func = func
        self._reward = reward
        self._goal = goal
        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self.funcdim = len(func(np.ones(len(stateconverter.flags)),
                                np.ones(len(actionconverter.flags))))
        self.weights = np.ones(self.funcdim)


    def reward(self, state, action, next_state):
        return self._reward(state, action, next_state)


    def next_state(self, state, action):
        self.simulator.set_state(state)
        return self.simulator.run(self.duration)


    def value(self, state):
        pass


    def learn(self):
        pass


    def recommend(self, state):
        pass


    def reset(self):
        self.weights = np.ones(self.funcdim)
