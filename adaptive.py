"""
This script implements adaptive fault-tolerant control of a sample system
using reinforcement learning. The environment is a topology. The objective is
to reach the lowest altitude. Actions are movements to adjacent grid points.
Faults introduced mix the outcome of one action with another. At occurance of
every fault, the policy is updated by learning from the current state's
neighbourhood. The learner is aware of the faults and how they change the
environment. The path traversed is divided into segments in between faults.
Faults are descrete and time independent.

Usage:

    > python adaptive.py -h
    > python adaptive.py
    > python .\adaptive.py --seed 40000 -f 0.2 -m 20 -a 0 0 -e 0.2 -s 10 -p softmax -r 0.75 -d 1
    > python .\adaptive.py --seed 40000 -f 0 -m 20 -a 0 0 -e 0.2 -s 10 -p softmax -r 0.01 -d 1 --usefunc
    > python .\adaptive.py -c 0.5 --seed 40000 -f 0.2 -m 20 -a 0 0 -e 0.2 -s 10 -p softmax -r 0.01 -d 0.1 --usefunc --hierarchical

Requires:
    numpy,
    matplotlib
"""

from argparse import ArgumentParser
import numpy as np
from qlearn import TestBench
from qlearn import QLearner
from qlearn import FLearner



class ModelPredictiveController(QLearner):
    """
    Creates a subclass of QLearner that uses Model Predictive
    Control to recommend actions. MPC does not learn a value function but
    instead does a receding horizon look-ahead at each timestep while
    choosing the action optimizing some static utility function.
    """

    def __init__(self, dmap, **kwargs):
        self.dmap = dmap
        super().__init__(**kwargs)

    def learn(self, **kwargs):
        """
        An MPC has no learning phase.
        """
        pass
    
    def recommend(self, state, **kwargs):
        """
        Implements the receding horizon online supervision algorithm by
        Abdelwahed et al.
        """
        min_dist = np.inf
        optimal = None
        visited = set()         # set of all states visited
        tree = [(None, state, 0, None)] # (parent ref, state, depth, action)
        while len(tree):
            cnode = tree.pop()
            if cnode[2] == self.depth+1:
                break
            visited.add(cnode[1])
            # add eligible states to be explored to tree
            for action, nstate in enumerate(self.neighbours(cnode[1])):
                if nstate not in visited:
                    node = (cnode, nstate, cnode[2]+1, action)
                    tree.insert(0, node)
                    visited.add(nstate)
                    # check state eligibility
                    if self.dmap[nstate] < min_dist:
                        min_dist = self.dmap[nstate]
                        optimal = node
        # Trace back to first action
        while optimal[0] is not None:
            action = optimal[3]
            optimal = optimal[0]
        return action



def adaptive_path(tb, start, fault):
    """
    Calculates path from start to a goal state while adapting to intermittent
    faults in the environment. At occurance of a fault, relearns policy in the
    vicinity of the state.

    Args:
        tb (TestBench): A TestBench instance with learner set up.
        start (tuple): The [Y, X] coordinates to start from.
        fault (func): A function that modifies the tb.learner's environment
            and returns True/False to indicate if a fault has occurred.

    Returns:
        A dict of the form {'001': [coords], ...'SEGMENT_NUMBER': List of coords}.
        Each segment is the path traversed in between faults,
        The coordinate of the final state reached,
        Number of iterations (i.e. number of points traversed).
        True if goal reached, else False
    """
    segment = [start]
    i = 0   # segments completed
    n = 0   # iterations completed
    segments = {}
    state = tb.coord2state(start)
    prevstate = state
    while not tb.learner.goal(state) and n < tb.num_states:
        if fault() or n == 0:
            segment = [segment[-1]]
            segments['{:03d}'.format(i)] = segment
            episodes = tb.learner.neighbours(state)
            tb.learner.learn(episodes=episodes)
            i += 1
        action = tb.learner.recommend(state)
        # stop path if no recommendation
        if action is None:
            break
        state = tb.learner.next_state(state, action)
        # stop path if stuck and no exploration
        if state == prevstate and tb.learner.exploration == 0:
            break
        prevstate = state
        segment.append(tb.state2coord(state))
        n += 1
    return segments, tb.state2coord(state), n, tb.learner.goal(state)



def fault_func(learner, probability, random, faultall=False):
    """
    Defines a function that introduces faults in the environment. The function
    is called after every action in adaptive_path().

    Args:
        learner (QLearner): The learner instance with rmatrix and tmatrix
            populated.
        probability (float): The probability of a fault occuring.\
        faultall (bool): Whether to create faults in all states or a sample.
        random (np.random.RandomState): A custom random number generator.

    Returns:
        A function that modifies the learner's environment. That function returns
        True or False depending on whether a fault occurred.
    """
    def fault():
        """
        * Picks a random action to fault,
        * Picks a random number of states where that action will be faulty,
        * For each fault state, the fault action gets 'shorted' with the results
          of one of the other actions or does nothing.
        * i.e. Transition/ reward matrix changes

        Returns:
            True if a fault occurs, False otherwise.
        """
        if random.rand() <= probability:
            # Choose a particular action to fault
            action = random.randint(learner.num_actions)
            # Number of states to create the action fault in
            if faultall:
                num_faults = learner.num_states
                states = np.arange(0, num_faults, step=1, dtype=int)
            else:
                num_faults = random.randint(1, learner.num_states+1)
                # The state indices where the fault occurs
                states = random.choice(learner.num_states, num_faults, replace=False)
            # -1 means no transition to another state
            faulty_actions = random.randint(-1, learner.num_actions, num_faults)
            # Applying changes
            learner.tmatrix[states, action] = learner.tmatrix[states, faulty_actions]
            learner.tmatrix[states, action] =\
                     np.where(faulty_actions == -1, states, learner.tmatrix[states, action])
            learner.rmatrix[states, action] =\
                     np.where(faulty_actions == -1, 0, learner.rmatrix[states, action])
            return True
        else:
            return False

    return fault



def distance_map(goals, rows, cols, flatten=True):
    """
    Calculates a distance map based on goal states on a rows x cols state space.

    Args:
        goals (list): List of [row, col] coordinates of goal states.
        rows, cols (int): Dimensions of state space.
        flatten (bool): Whether to flatten the distance map into a 1D vector.

    Returns:
        If flatten=True, a 1D array [rows x cols] where each element is distance
        to of that state index to closest goal state. The index = r * cols + c
        where [r,c] are state coordinates.
        If flatten=False, a [rows x cols] 2D array where each element is the
        closest distance of the state at that position from goal states.
    """
    dmap = np.zeros((rows, cols))
    dmap[:,:] = np.inf
    ycoords, xcoords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    for g in goals:
        gx = g % cols
        gy = g // cols
        distance = (ycoords - gy)**2 + (xcoords - gx)**2
        update_locations = dmap > distance
        dmap[update_locations] = distance[update_locations]
    if flatten:
        return np.sqrt(dmap).flatten()
    return np.sqrt(dmap)



def print_results(segments, final, length, goal):
    print('\nsegments: ', len(segments))
    print('path length: ', length)
    print('final state: ', final, ' (Goal reached)' if goal else ' (Not goal)')
    print()



if __name__ == '__main__':

    np.seterr(under='ignore')

    args = ArgumentParser()
    args.add_argument('-t', '--topology', metavar='T', type=int,
                      help="Topology grid size", default=10)
    args.add_argument('-a', '--start', metavar=('Y', 'X'), type=int, nargs=2,
                      help="Starting coordinates", default=None)
    args.add_argument('-r', '--rate', metavar='R', type=float,
                      help="Learning rate (0, 1]", default=0.5)
    args.add_argument('-c', '--coverage', metavar='C', type=float,
                      help="State space coverage for initial learning phase (0, 1]", default=1)
    args.add_argument('-d', '--discount', metavar='D', type=float,
                      help="Discount factor (0, 1]", default=0.75)
    args.add_argument('-e', '--explore', metavar='E', type=float,
                      help="Exploration while recommending actions [0, 1]", default=0.)
    args.add_argument('-s', '--steps', metavar='S', type=int,
                      help="Number of steps to look ahead during learning", default=1)
    args.add_argument('-m', '--maxdepth', metavar='M', type=int,
                      help="Number of steps at most in each episode (==horizon for MPC)", default=None)
    args.add_argument('-p', '--policy', metavar='P', choices=['uniform', 'softmax', 'greedy'],
                      help="The action selection policy", default='uniform')
    args.add_argument('-o', '--online', action='store_true',
                      help="Online or Offline policy update", default=False)
    args.add_argument('-f', '--faultprob', metavar='F', type=float,
                      help="Probability of fault after each action", default=0.25)
    args.add_argument('--greedyprob', metavar='G', type=float,
                      help="Probability of using best action when policy=greedy", default=0.5)
    args.add_argument('--seed', metavar='SEED', type=int,
                      help="Random number seed", default=None)
    args.add_argument('--hierarchical', action='store_true',
                      help="Hierarchical state space traversal for functional learning (use with --usefunc).", default=False)
    args.add_argument('--usempc', action='store_true',
                      help="Use model predictive control instead of reinforcement learning.", default=False)
    args.add_argument('--usefunc', action='store_true',
                      help="Use functional instead of tabular approximation for value function.", default=False)
    args.add_argument('--numtrials', metavar='TRIALS', type=int,
                      help="Analyse a number of trials to run instead of showing a single result", default=1)
    args = args.parse_args()

    # Print paramters
    for key, value in vars(args).items():
        print(key, ': ', value)
    
    random = np.random.RandomState(args.seed)
    paths = []          # stores segments for each trial
    endpoints = []      # stores final coords for each trial
    lengths = []        # stores total path lengths for each trial
    successes = []      # stores goal reached/not for each trial
    
    for i in range(args.numtrials):
        # Get the starting point and initial seed
        start = random.randint(0, args.topology, 2) if args.start is None else args.start
        seed = args.seed if args.numtrials == 1 else i
        
        
        # Use tabular reinforcement learning
        if not args.usefunc and not args.usempc:
            # set up testbench
            mode = QLearner.ONLINE if args.online else QLearner.OFFLINE
            tb = TestBench(size=args.topology, seed=seed, learner=QLearner,
                        lrate=args.rate, discount=args.discount, exploration=args.explore,
                        depth=args.maxdepth, steps=args.steps, policy=args.policy,
                        max_prob=args.greedyprob, mode=mode)
            dmap = distance_map(tb.goals, args.topology, args.topology, flatten=True)
            fault = fault_func(tb.learner, args.faultprob,\
                            np.random.RandomState(seed), faultall=True)

            # Run simulation with intermittent faults and adaptive learning
            tb.learner.learn(coverage=args.coverage)
            segments, final, length, goal = adaptive_path(tb, start, fault)
            if args.numtrials == 1:
                print_results(segments, final, length, goal)
                # Show optimal and adaptive paths to goal state
                tb.show_topology(**segments, showfield=args.faultprob==0)
        
        
        # Use functional reinforcement learning
        elif args.usefunc:
            # define functional form for value function
            funcdim = 15
            nsa = args.topology * 2     # normalization factors
            nss = args.topology ** 2
            naa = 4
            ns = args.topology
            na = 2
            def dfunc(s, a, w):
                return np.array([s[0]*a[0]/nsa,
                                 s[0]*a[1]/nsa,
                                 s[1]*a[0]/nsa,
                                 s[1]*a[1]/nsa,
                                 s[0]**2/nss,
                                 s[1]**2/nss,
                                 s[0]*s[1]/nss,
                                 a[0]**2/naa,
                                 a[1]**2/naa,
                                 a[0]*a[1]/naa,
                                 a[0], a[1], s[0], s[1], 1])
            def func(s, a, w):
                return np.dot(w, dfunc(s, a, w))
            # set up testbench
            mode = FLearner.ONLINE if args.online else FLearner.OFFLINE
            tbfunc = TestBench(size=args.topology, seed=seed, learner=FLearner,
                        lrate=args.rate, discount=args.discount, exploration=args.explore,
                        depth=args.maxdepth, steps=args.steps, policy=args.policy,
                        max_prob=args.greedyprob, mode=mode, func=func, dfunc=dfunc,
                        funcdim=funcdim)
            # define stepsize for heirarchical learning
            if args.hierarchical:
                deltaheight = np.amax(tbfunc.topology) - np.amin(tbfunc.topology)
                def stepsize(state):
                    cr, cc = tbfunc.state2coord(state)
                    height = tbfunc.topology[cr, cc]
                    return int(np.sqrt(height * tbfunc.size / deltaheight)) + 1
                tbfunc.learner.stepsize = stepsize

            fault = fault_func(tbfunc.learner, args.faultprob,\
                            np.random.RandomState(seed), faultall=True)

            # Run simulation with intermittent faults and adaptive learning
            tbfunc.learner.learn(coverage=args.coverage)
            segments, final, length, goal = adaptive_path(tbfunc, start, fault)
            if args.numtrials == 1:
                print_results(segments, final, length, goal)
                # Show optimal and adaptive paths to goal state
                tbfunc.show_topology(**segments, showfield=args.faultprob==0)

        
        # Use model predictive control (mpc)
        elif args.usempc:
            # set up testbench
            tbmpc = TestBench(size=args.topology, seed=seed, learner=None)
            dmap = distance_map(tbmpc.goals, args.topology, args.topology, flatten=True)
            learner = ModelPredictiveController(dmap=dmap, seed=tbmpc.seed,
                                            depth=args.maxdepth, goal=tbmpc.goals,
                                            rmatrix=tbmpc.rmatrix, tmatrix=tbmpc.tmatrix)
            tbmpc.learner = learner
            faultmpc = fault_func(tbmpc.learner, args.faultprob,\
                                np.random.RandomState(seed), faultall=True)
        # Run simulation with intermittent faults and adaptive learning
            segments, final, length, goal = adaptive_path(tbmpc, start, faultmpc)
            if args.numtrials == 1:
                print_results(segments, final, length, goal)
                # Show optimal and adaptive paths to goal state:
                tbmpc.show_topology(**segments, showfield=False)
    
    # Analyse results from one of three approaches (tabular, functional, mpc)
