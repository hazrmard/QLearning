"""
This script implements reinforcement learning with faults on a fuel tank system
represented by a netlist (models/fuel_tanks.netlist). There are 6 tanks. 4 of
the tanks are primary tanks and have a outputs to engines. The remaining 2 are
auxiliary tanks which feed into the primary tanks. Each tank is represented
by a capacitor. Resistors are used to simulate internal resistances and
switches. The system rewards fuel tanks balanced on each side and penalizes
imbalance. Faults in the system are leak(s) in fuel tanks.

Fuel tanks are arranged physically as:

    1  2  LAux   |   RAux   3  4

You can use LTSpice (or similar applications) to view the graphical circuit
representation of the system in: models/fuel_tanks.asc

Usage:

> python tanks.py --help
> python tanks.py -c 1e-4  --seed 1001 -l tankweights.dat -t 3 -s 3 -m 6 -i 4 4 4 4 4 4 0 -u 4
> python tanks.py -c 1e-4  --seed 1001 -f tankweights.dat -t 2 -s 3 -m 6 -i 4 4 4 4 4 4 0 -u 3

Default model and learning parameters can be changed below. Some of them
can be tuned from the command-line.

Requires:
    flask,
    numpy,
    ahkab
"""

import numpy as np
import flask
from argparse import ArgumentParser, RawTextHelpFormatter
from qlearn import Netlist
from qlearn import Resistor
from qlearn import FlagGenerator
from qlearn import Simulator
from qlearn import SLearner
from qlearn import utils

# Default model configuration parameters
NETLIST_FILE = 'models/fuel_tanks.netlist'
ON_RESISTANCE = 1e0     # Valve resistance when on
OFF_RESISTANCE = 1e6    # Valve resistance when off
INTERNAL_RESISTANCE = 1e3   # Resistance associated with tanks for normal drainage
CAPACITANCE = 1e-3          # Tank capacities (Total fuel mass / total potential)
MAX_SIM_TSTEP = 1e-2        # Simulation time resolution i.e. timestep
DELTA_T = 3e-2              # Time duration of each action i.e. step size
NUM_TANKS = 6           # Not variable. Hardcoded in netlist.
NUM_VALVES = 14         # Not variable. Hardcoded in netlist.
NUM_LEVELS = 1 + 4      # Possible potential values to consider when generating episodes [0, NUM_LEVELS)
FAULT = ['']            # Fault type. See create_fault()

# Default learning configuration parameters
GOAL_THRESH = 0.05  # Sensitivity to a state being considered goal. Smaller -> strict
COVERAGE = 0.2      # Fraction of states to cover in learning initially (or load weights from file)
LRATE = 1e-2        # Learning rate (0, 1]
DISCOUNT = 0.75     # Discount factor (0, 1]
EXPLORATION = 0.25  # Exploration while recommending actions [0, 1]
POLICY = SLearner.UNIFORM   # The action selection policy
DEPTH = 5           # Number of steps at most in each learning episode
STEPS = 1           # Number of steps to look ahead during learning
SEED = None         # Random number seed

# Set up command-line configuration
args = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
args.add_argument('-i', '--initial', metavar=tuple(['L']*NUM_TANKS + ['A']), type=float, 
                  nargs=NUM_TANKS+1, help="Initial tank levels and first action",
                  default=None)
args.add_argument('-n', '--num_levels', metavar='N', type=int,
                  help="Number of levels per tank", default=NUM_LEVELS)
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
args.add_argument('-u', '--fault', metavar='U', type=str, nargs='*',
                  help="Name of tank with leak", default=FAULT)
args.add_argument('-p', '--policy', metavar='P', choices=['uniform', 'softmax', 'greedy'],
                  help="The action selection policy", default=POLICY)
args.add_argument('-l', '--load', metavar='F', type=str,
                  help="File to load learned policy from", default='')
args.add_argument('-f', '--file', metavar='F', type=str,
                  help="File to save learned policy to", default='')
args.add_argument('--seed', metavar='SEED', type=int,
                  help="Random number seed", default=SEED)
args.add_argument('-x', '--disable', action='store_true',
                  help="Learning disabled if included", default=False)
ARGS = args.parse_args()

# Specify dimension and resolution of state and action vectors
# A state vector is a NUM_TANKS+1 vector where the last element is the open valve
# and the first NUM_TANKS elements are potentials in tanks
STATES = FlagGenerator(*[ARGS.num_levels] * NUM_TANKS, NUM_VALVES + 1)
# An action vector is a single element vector signalling which of the 14 valves
# is active. Same as the last element in state vector. A 0 value means all
# valves are off.
ACTIONS = FlagGenerator(NUM_VALVES + 1)


# Instantiate netlist representing the fuel tank system
NET = Netlist('Tanks', path=NETLIST_FILE)
INITIAL = NET.directives['ic'][0]


# Get list of resistors to be used as switches - ignoring internal resistances
RESISTORS = [r for r in NET.elements_like('r') if not r.name.startswith('ri')]
for res in RESISTORS:
    res.value = OFF_RESISTANCE
# Set internal resistances
INT_RESISTORS = NET.elements_like('ri') 
for rint in INT_RESISTORS:
    rint.value = INTERNAL_RESISTANCE
# Get list of capacitors representing fuel tanks and set values
CAPACITORS = NET.elements_like('c')     # [c1, c2, c3, c4, cl, cr]
for cap in CAPACITORS:
    cap.value = CAPACITANCE


# Define a state mux for the simulator which converts state and action vectors
# into changes in the netlist
def state_mux(svec, avec, netlist):
    for i in range(NUM_TANKS):
        INITIAL.param('v(' + str(CAPACITORS[i].nodes[0]) + ')', svec[i])
    for resistor in RESISTORS:
        resistor.value = OFF_RESISTANCE
    if avec[0] != 0:
        RESISTORS[int(avec[0]-1)].value = ON_RESISTANCE
    return NET


# Define state demux for the simulator which converts simulation results into
# a state vector
def state_demux(psvec, pavec, netlist, result):
    svec = np.zeros(NUM_TANKS+1)
    svec[-1] = pavec[0]
    for i in range(NUM_TANKS):
        svec[i] = result['v(' + str(CAPACITORS[i].nodes[0]) + ')']
    return svec


# The reward function returns a measure of the desirability of a state,
# in this case the moment about the central axis
def reward(svec, avec, nsvec):
    moment = 3 * (nsvec[0] - nsvec[3]) + \
             2 * (nsvec[1] - nsvec[2]) + \
             1 * (nsvec[4] - nsvec[5])
    return -abs(moment) # reward is always negative, max=0


# Get minimum possible reward, to use as a threshold for measuring goal state
MIN_REWARD = abs(reward(None, None, np.ones(NUM_TANKS) * NUM_LEVELS))


# Returns Trus if a state is considered a terminal/goal state
def goal(svec):
    return abs(reward(None, None, svec)) < GOAL_THRESH * MIN_REWARD


# Returns the gradient of the policy function w.r.t weights: a vector of length
# FUNCDIM (see below)
def dfunc(svec, avec, weights):
    valves = np.zeros(NUM_VALVES)
    if avec[0] != 0:
        valves[int(avec[0])-1] = 1
    return np.concatenate((svec[:-1] / ARGS.num_levels, valves, [1]))


# Returns the value of a state/action given weights. The policy function.
# Used to compute the optimal action from each state when exploiting a policy.
def func(svec, avec, weights):
    return np.dot(dfunc(svec, avec, weights), weights)


# Number of weights to learn in functional approximation, in this case:
# 1 weight for each tank, 1 weight for each valve, and a bias term
FUNCDIM = NUM_TANKS + NUM_VALVES + 1


# Define a fault function. Calling it with an argument introduces a fault in the
# system defined by NET. A fault halves the internal resistance associated with
# the tank.
# Args:
#   faults: A sequence of tank names where to introduce a fault. If '', all
#           internal resistances are restored to INTERNAL_RESISTANCE
def create_fault(*faults):
    for fault in faults:
        if str(fault) == '':
            for resistor in INT_RESISTORS:
                resistor.value = INTERNAL_RESISTANCE
            break
        else:
            NET.element('ri' + str(fault)).value = ON_RESISTANCE

create_fault(*ARGS.fault)


# Create a simulator to be used by SLearner
SIM = Simulator(env=NET, timestep=MAX_SIM_TSTEP, state_mux=state_mux,
                state_demux=state_demux)


# Create the SLearner instance
LEARNER = SLearner(reward=reward, simulator=SIM, stateconverter=STATES,
                   actionconverter=ACTIONS, goal=goal, func=func, funcdim=FUNCDIM,
                   dfunc=dfunc, lrate=ARGS.rate, discount=ARGS.discount,
                   policy=ARGS.policy, depth=ARGS.maxdepth,
                   steps=ARGS.steps, seed=ARGS.seed, stepsize=DELTA_T)


# Print paramters
for key, value in vars(ARGS).items():
    print('%12s: %-12s' % (key, value))
# Loading weights or learning new policy
if not ARGS.disable:
    if ARGS.load == '':
        input('\nPress Enter to begin learning.')
        print('Learning episodes: %5d out of %d states' %
            (int(ARGS.coverage * STATES.num_states), STATES.num_states))
        LEARNER.learn(coverage=ARGS.coverage)
        if ARGS.file != '':
            utils.save_matrix(LEARNER.weights, ARGS.file)
    else:
        LEARNER.weights = utils.read_matrix(ARGS.load)


# Set up a server
APP = flask.Flask('Tanks', static_url_path='', static_folder='', template_folder='')
svec = np.zeros(NUM_TANKS + 1, dtype=float)
avec = np.zeros(1, dtype=int)
COUNT = 0       # number of steps taken since start of server

@APP.route('/')
def demo():
    if ARGS.initial is None:
        svec[:-1] = LEARNER.random.rand(NUM_TANKS) * (ARGS.num_levels - 1)
        svec[-1] = LEARNER.random.randint(14)
        avec[:] = LEARNER.next_action(svec)
    else:
        svec[:] = np.array(ARGS.initial)
        avec[:] = ARGS.initial[-1]
    return flask.render_template('demo.html', N=ARGS.num_levels, T=NUM_TANKS,
                                 L=[c.name[1:] for c in CAPACITORS],
                                 O=[0, 1, 4, 5, 2, 3])

@APP.route('/status/')
def status():
    global COUNT
    s = list(svec)                                  # cache last results
    a = list(avec)
    w = list(LEARNER.weights)

    if goal(s):
        exit('Goal state reached.')

    if LEARNER.random.rand() <= ARGS.explore and not ARGS.disable: # re-learn at interval steps
        episodes = LEARNER.neighbours(svec)
        LEARNER.learn(episodes=episodes)

    COUNT += 1
    svec[:] = LEARNER.next_state(svec, avec)        # compute new results
    if not ARGS.disable:
        avec[:] = LEARNER.recommend(svec)
    if a[0] == 0:
        action = 'All off'
    else:
        action = RESISTORS[a[0]-1].name[1:].upper() + ' on'
    return flask.jsonify(levels=[str(i) for i in s],
                         action=action,
                         weights=[str(i) for i in w],
                         imbalance=reward(None, None, s))   # return cached results

APP.run()