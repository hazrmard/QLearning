"""
This script implements control with faults on a fuel tank system
on a cargo plane represented by a custom simulator. There are 6 tanks. 4 of
the tanks are primary tanks and have outputs to engines. The remaining 2 are
auxiliary tanks which feed into the primary tanks. The system drains outer tanks
first before using inner tanks to feed engines. Faults in the system are leak(s)
in fuel tanks. Only a single fault occurs at a time.

Two kinds of control are implemented:
    - Reinforcement learning: The controller explores a sample of the state space
        to estimate values for different actions. Actions with the largest
        values are picked at each step. Function approximation is used for
        value estimation.
    - Model predictive control: The controller samples each of the possible
        reachable states within a timestep horizon and picks a state closest
        to the goal.

It is assumed that the controller has an accurate model of the system.

Fuel tanks are arranged physically (and indexed) as:

    1  2  LAux   |   RAux   3  4
   [1  2  3          4      5  6]


Usage:

> python tanks.py --help
> python .\tankscustomdemo.py -c 2e-4 -f 6 -r 0.2 -s 5 -m 10 -e 0.75
> python .\tankscustomdemo.py --usempc -m 1

Default model and learning parameters can be changed below. Some of them
can be tuned from the command-line.

Requires:
    flask,
    numpy,
    scipy
"""

import math
import random
import flask
import numpy as np
from scipy.integrate import trapz
from argparse import ArgumentParser
from qlearn import SLearner
from qlearn import FlagGenerator



# Default configuration parameters
COVERAGE = 1e-4      # Fraction of states to cover in learning initially
LRATE = 1e-1        # Learning rate (0, 1]
DISCOUNT = 0.75     # Discount factor (0, 1]
EXPLORATION = 0     # Exploration while recommending actions [0, 1]
POLICY = SLearner.SOFTMAX   # The action selection policy
DEPTH = 10          # Number of steps at most in each learning episode
STEPS = 1           # Number of steps to look ahead during learning
DENSITY = 1.0       # Fraction of neighbouring states sampled for episodes when exploring
SEED = None         # Random number seed
FAULT = list(range(7))         # Default set of faults
DELTA_T = 1
FUNCDIM = 7

# Set up command-line configuration
args = ArgumentParser()
args.add_argument('-i', '--initial', metavar=tuple(['L']*6 + ['A']*6), type=float, 
                  nargs=12, help="Initial tank levels and switch values.",
                  default=[100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0])
args.add_argument('-c', '--coverage', metavar='C', type=float,
                  help="Fraction of states to cover in learning", default=COVERAGE)
args.add_argument('-r', '--rate', metavar='R', type=float,
                  help="Learning rate (0, 1]", default=LRATE)
args.add_argument('-d', '--discount', metavar='D', type=float,
                  help="Discount factor (0, 1]", default=DISCOUNT)
args.add_argument('-e', '--explore', metavar='E', type=float,
                  help="Exploration while recommending actions [0, 1]", default=EXPLORATION)
args.add_argument('-s', '--steps', metavar='S', type=int,
                  help="Number of steps to look ahead during learning", default=STEPS)
args.add_argument('-m', '--maxdepth', metavar='M', type=int,
                  help="Number of steps at most in each learning episode", default=DEPTH)
args.add_argument('-f', '--fault', metavar='F', type=int, nargs='+',
                  help="List of faults. For server, first fault is chosen.", default=FAULT)
args.add_argument('-p', '--policy', metavar='P', choices=['uniform', 'softmax', 'greedy'],
                  help="The action selection policy", default=POLICY)
args.add_argument('--seed', metavar='SEED', type=int,
                  help="Random number seed", default=SEED)
args.add_argument('-x', '--disable', action='store_true',
                  help="Learning disabled if included", default=False)
args.add_argument('--usempc', action='store_true',
                  help="Use model predictive controller.", default=False)
args.add_argument('--hierarchical', action='store_true',
                      help="Hierarchical state space traversal.", default=False)
args.add_argument('--density', type=float,
                      help="State sampling density (0, 1]. 1 => all neighbours sampled.", default=DENSITY)
args.add_argument('--numtrials', type=int, metavar='N',
                  help="Run trials instead of interactive server.", default=None)
args.add_argument('--noise', type=float, metavar='N',
                  help="Amount of noise in model behaviour.", default=0.0)
args.add_argument('--verbose', action='store_true',
                  help="Print parameters used.", default=False)
ARGS = args.parse_args()



class SixTankModel:
    def __init__(self, fault=0, noise=0, seed=None):
        self.R = 4.00
        self.F = 8.00
        self.fault = fault
        self.noise = noise
        self.random = np.random.RandomState(seed)


    def set_state(self, S):
        self.tank_1 = S[0]
        self.tank_2 = S[1]
        self.tank_LA = S[2]
        self.tank_RA = S[3]
        self.tank_3 = S[4]
        self.tank_4 = S[5]
        self.DL = S[6]
        self.EL = S[7]
        self.FL = S[8]
        self.FR = S[9]
        self.ER = S[10]
        self.DR = S[11]
    

    def set_action(self, A):
        self.DL = A[0]
        self.EL = A[1]
        self.FL = A[2]
        self.FR = A[3]
        self.ER = A[4]
        self.DR = A[5]


    def run(self, state, action, stepsize=DELTA_T, **kwargs):
        self.set_state(state)
        self.set_action(action)
        if ((self.tank_1 + self.tank_2 + self.tank_LA) >= 10 and (self.tank_3 + self.tank_4 + self.tank_RA) >= 10):
            demand_left = 10
            demand_right = 10
        else:
            if ((self.tank_1 + self.tank_2 + self.tank_LA) >= 10):
                demand_right = self.tank_3 + self.tank_4 + self.tank_RA
                if ((self.tank_1 + self.tank_2 + self.tank_LA) >= (10 + 10 - demand_right)):
                    demand_left = 10 + 10 - demand_right
                else:
                    demand_left = self.tank_1 + self.tank_2 + self.tank_LA
            if ((self.tank_3 + self.tank_4 + self.tank_RA) >= 10):
                demand_left = self.tank_1 + self.tank_2 + self.tank_LA
                if ((self.tank_3 + self.tank_4 + self.tank_RA) >= (10 + 10 - demand_left)):
                    demand_right = 10 + 10 - demand_left
                else:
                    demand_right = self.tank_3 + self.tank_4 + self.tank_RA
            if ((self.tank_1 + self.tank_2 + self.tank_LA) < 10 and (self.tank_3 + self.tank_4 + self.tank_RA) < 10):
                demand_left = self.tank_1 + self.tank_2 + self.tank_LA
                demand_right = self.tank_3 + self.tank_4 + self.tank_RA

        if (self.tank_1 >= demand_left):
            pump_1 = demand_left
            pump_2 = 0
            pump_LA = 0
        else:
            pump_1 = self.tank_1
            if (self.tank_2 >= (demand_left - self.tank_1)):
                pump_2 = demand_left - self.tank_1
                pump_LA = 0
            else:
                pump_2 = self.tank_2
                pump_LA = demand_left - self.tank_1 - self.tank_2
        if (self.tank_4 >= demand_right):
            pump_4 = demand_right
            pump_3 = 0
            pump_RA = 0
        else:
            pump_4 = self.tank_4
            if (self.tank_3 >= (demand_right - self.tank_4)):
                pump_3 = demand_right - self.tank_4
                pump_RA = 0
            else:
                pump_3 = self.tank_3
                pump_RA = demand_right - self.tank_3 - self.tank_4
        
        self.tank_1 = self.tank_1 - pump_1
        self.tank_2 = self.tank_2 - pump_2
        self.tank_LA = self.tank_LA - pump_LA
        self.tank_RA = self.tank_RA - pump_RA
        self.tank_3 = self.tank_3 - pump_3
        self.tank_4 = self.tank_4 - pump_4
        
        if ((self.DL + self.EL + self.FL + self.FR + self.ER + self.DR) == 0):
            p = 0
        else:
            p = (self.tank_1 * self.DL + self.tank_2 * self.EL + self.tank_LA * self.FL
                 + self.tank_RA * self.FR + self.tank_3 * self.ER + self.tank_4 * self.DR) / float(self.DL + self.EL + self.FL + self.FR + self.ER + self.DR)
        
        self.tank_1 = self.tank_1 + self.DL * \
            (((p / self.R) - (self.tank_1 / self.R)) * (stepsize)) \
            - ((self.tank_1 / self.F) * (stepsize) if self.fault == 1 else 0)
        self.tank_2 = self.tank_2 + self.EL * \
            (((p / self.R) - (self.tank_2 / self.R)) * (stepsize)) \
            - ((self.tank_2 / self.F) * (stepsize) if self.fault == 2 else 0)
        self.tank_LA = self.tank_LA + self.FL * \
            (((p / self.R) - (self.tank_LA / self.R)) * (stepsize)) \
            - ((self.tank_LA / self.F) * (stepsize) if self.fault == 3 else 0)
        self.tank_RA = self.tank_RA + self.FR * \
            (((p / self.R) - (self.tank_RA / self.R)) * (stepsize)) \
            - ((self.tank_RA / self.F) * (stepsize) if self.fault == 4 else 0)
        self.tank_3 = self.tank_3 + self.ER * \
            (((p / self.R) - (self.tank_3 / self.R)) * (stepsize)) \
            - ((self.tank_3 / self.F) * (stepsize) if self.fault == 5 else 0)
        self.tank_4 = self.tank_4 + self.DR \
            * (((p / self.R) - (self.tank_4 / self.R)) * (stepsize)) \
            - ((self.tank_4 / self.F) * (stepsize) if self.fault == 6 else 0)
        
        noisy = self.random.normal(1, self.noise, 6) * \
                [self.tank_1, self.tank_2, self.tank_LA, self.tank_RA, self.tank_3, self.tank_4]
        return np.concatenate((noisy, action))



class ModelPredictiveController(SLearner):
    """
    Creates a subclass of SLearner that uses Model Predictive
    Control to recommend actions. MPC does not learn a value function but
    instead does a receding horizon look-ahead at each timestep while
    choosing the action optimizing some static utility function.

    Args:
        dmap (func): A function that takes the state vector and returns a number
            representing the "distance" from ideal state.
        simulator:  An object with a run() function that takes state and action
            vectors and an optional stepsize argument. Returns the next state
            vector.
        state/actionconverter (FlagGenerator): Encodes/Decodes vectors into
            integer representation (mostly for compatibility w/ SLearner)
        depth (int): Maximum horizon to look ahead.
        density (float): The fraction of neighbouring states to sample.
        seed (int): Random number generator seed. Otherwise random.
    """

    def __init__(self, dmap, simulator, stateconverter, actionconverter, depth=1,
                 density=1, seed=None):
        self.random = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        self.dmap = dmap                   # cost measure to minimize
        self.depth = depth
        self.density = density
        self.simulator = simulator
        self.stateconverter = stateconverter
        self.actionconverter = actionconverter
        self.funcdim = 1                    # for compatibility
        self._avecs = [avec for avec in self.actionconverter]

        self.weights = np.ones((1, 13)) # just for compatibility

    def learn(self, *args, **kwargs):
        """
        An MPC has no learning phase.
        """
        return [[]], [[]]
  
    def recommend(self, state, **kwargs):
        """
        Implements the receding horizon online supervision algorithm by
        Abdelwahed et al.
        """
        min_dist = np.inf
        optimal = None
        tree = [(None, state, 0, None)] # (parent ref, state, depth, action)
        while len(tree):
            cnode = tree.pop()
            if cnode[2] == self.depth+1:
                break
            # add eligible states to be explored to tree
            neighbours = self.neighbours(cnode[1])
            self.random.shuffle(neighbours)
            for action, nstate in enumerate(neighbours[:int(np.ceil(len(neighbours) * self.density))]):
                node = (cnode, nstate, cnode[2]+1, action)
                tree.insert(0, node)
                # check state eligibility
                if self.dmap(nstate) < min_dist:
                    min_dist = self.dmap(nstate)
                    optimal = node
        # Trace back to first action
        if optimal is None: # i.e. starting state is closest state, maintain action
            return state[6:]
        while optimal[0] is not None:
            action = optimal[3]
            optimal = optimal[0]
        return self.actionconverter.decode(action)



def moment(s):
        return abs(3 * (s[0] - s[5]) + \
        2 * (s[1] - s[4]) + \
        1 * (s[2] - s[3]))



def goal(state):
    return sum(state[:6]) <= 5



def hierarchy(state):
    return DELTA_T + int(np.log10(1 + moment(state)))



def reward(state, action, nstate, **kwargs):
    # return(1000.0 / ((abs(state[0] - state[5])) + (abs(state[1] - state[4])) + (abs(state[2] - state[3])) + 1))
    return (sum(nstate[:6]) / 600) + (1 / (1 + moment(nstate)))



def dfunc(state, action, weights):
    # return np.concatenate((state[:6], action, [1])) / np.array([100, 100, 100, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1])
    return np.array([state[i] * (action[i] + 1) / 200 for i in range(6)] + [1])



def func(state, action, weights):
    return np.dot(dfunc(state, action, weights), weights)


# The sampling grid over the state space. A total of 1,000,000 states.
STATES = FlagGenerator((20, 5, 100), (20, 5, 100), (20, 5, 100), (20, 5, 100),
                       (20, 5, 100), (20, 5, 100), 2, 2, 2, 2, 2, 2)
# The possible set of actions (64).
ACTIONS = FlagGenerator(2, 2, 2, 2, 2, 2)

# The system with a possible fault
SIM = SixTankModel(fault=ARGS.fault[0])


if not ARGS.usempc:
# Create the SLearner instance
    LEARNER = SLearner(reward=reward, simulator=SIM, stateconverter=STATES,
                    actionconverter=ACTIONS, goal=goal, func=func, funcdim=FUNCDIM,
                    dfunc=dfunc, lrate=ARGS.rate, discount=ARGS.discount,
                    policy=ARGS.policy, depth=ARGS.maxdepth,
                    steps=ARGS.steps, seed=ARGS.seed,
                    stepsize=hierarchy if ARGS.hierarchical else lambda x:DELTA_T)
else:
    LEARNER = ModelPredictiveController(dmap=moment, simulator=SIM,
                                        stateconverter=STATES, actionconverter=ACTIONS,
                                        depth=ARGS.maxdepth, seed=ARGS.seed,
                                        density=ARGS.density)


# Print paramters if verbose
if ARGS.verbose:
    for key, value in vars(ARGS).items():
        try:
            print('%12s: %-12s' % (key, value))
        except:
            pass


# Either run interactive server, or multiple trials
if ARGS.numtrials is None:
    # Set up a server
    APP = flask.Flask('Tanks', static_url_path='', static_folder='', template_folder='')
    svec = np.zeros(12, dtype=float)
    avec = np.zeros(6, dtype=int)

    # Initial learning for RL controller
    if not ARGS.disable and not ARGS.usempc:
        LEARNER.learn(coverage=ARGS.coverage)

    @APP.route('/')
    def demo():
        svec[:] = np.array(ARGS.initial)
        avec[:] = ARGS.initial[6:]
        return flask.render_template('demo.html', N=100, T=6,
                                    L=['1', '2', 'LA', 'RA', '3', '4'],
                                    O=[0, 1, 2, 3, 4, 5])

    @APP.route('/status/')
    def status():
        s = list(svec)                                  # cache last results
        a = list(avec)
        w = list(LEARNER.weights)

        if not ARGS.disable:
            if LEARNER.random.rand() <= ARGS.explore:   # re-learn
                episodes = LEARNER.neighbours(svec)
                LEARNER.random.shuffle(episodes)
                LEARNER.learn(episodes=episodes[:int(np.ceil(len(episodes) * ARGS.density))])
            avec[:] = LEARNER.recommend(svec)

        svec[:] = LEARNER.next_state(svec, avec)        # compute new results

        if goal(s):
            exit('Goal state reached.')

        imbalance = -moment(s)

        return flask.jsonify(levels=[str(i) for i in s],
                            action=' '.join(['{:2d}'.format(a) for a in avec]),
                            weights=[str(i) for i in w],
                            imbalance=imbalance)   # return cached results

    APP.run(debug=1, use_reloader=False, use_evalex=False)

else:
    # Run multiple trials
    imbalances = []     # average imbalance for each trial
    lengths = []        # length of each trial until goal
    areas = []          # average areas under imbalance curves
    for i in range(ARGS.numtrials):
        LEARNER.simulator.fault = LEARNER.random.choice(ARGS.fault)  # introduce new fault
        if not ARGS.disable:                                    # re-learn on new trial
            LEARNER.reset()
            LEARNER.learn(coverage=ARGS.coverage)

        svec = np.array(ARGS.initial)   # all trials start with specified initial state
        avec = svec[6:]
        imbalance = [moment(svec)]
        length = 1
        while True:
            if not ARGS.disable:
                if LEARNER.random.rand() <= ARGS.explore:   # explore
                    episodes = LEARNER.neighbours(svec)
                    LEARNER.random.shuffle(episodes)
                    LEARNER.learn(episodes=episodes[:int(np.ceil(len(episodes) * ARGS.density))])
                avec = LEARNER.recommend(svec)              # exploit

            svec = LEARNER.next_state(svec, avec)
            imbalance.append(moment(svec))
            length += 1

            if goal(svec):                                  # quit trial on goal
                imbalances.append(max(imbalance))
                lengths.append(length)
                areas.append(trapz(imbalance))
                break



    print('MaxImbalance: {0:6.2f}\tLength: {1:6d}\tTotalImbalance: {2:6.2f}'\
            .format(np.mean(imbalances), int(np.mean(lengths)), np.mean(areas)))
