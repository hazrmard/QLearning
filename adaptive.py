"""
This script implements adaptive fault-tolerant control of a sample system
using reinforcement learning. The environment is a topology. The objective is
to reach the lowest altitude. Actions are movements to adjacent grid points.
Faults introduced mix the outcome of one action with another. A single trial
starts with a fault in the system. The controllers are unaware a fault has
occurred.

Usage:

    > python adaptive.py -h
    > python adaptive.py
    > python .\adaptive.py --seed 40000 -a 0 0 -m 30 -s 20 -r 0.75 -d 1  --showfield -p greedy --greedyprob 0.75
    > python .\adaptive.py --seed 40000 -a 0 0 -m 100 -s 10 -r 0.01 -d 0.9  --showfield -p greedy --greedyprob 0.75 -e 0.1 --usefunc

Requires:
    numpy,
    matplotlib
"""

from argparse import ArgumentParser
import numpy as np
from qlearn import TestBench
from qlearn import QLearner
from qlearn import FLearner
from qlearn import FlagGenerator



class ModelPredictiveController(QLearner):
    """
    Creates a subclass of QLearner that uses Model Predictive
    Control to recommend actions. MPC does not learn a value function but
    instead does a receding horizon look-ahead at each timestep while
    choosing the action optimizing some static utility function.
    """

    def __init__(self, dmap, tmatrix, goal, depth, seed, rmatrix):
        self.random = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        self.dmap = dmap                   # cost measure to minimize
        self.itmatrix = np.copy(tmatrix)   # initial model of environment
        self.tmatrix = tmatrix             # actual model of environment
        self.depth = depth if depth is not None else np.inf
        self.rmatrix = rmatrix             # not used - just for compatibility
        self.exploration = 0
        self.set_goal(goal)

    
    def model_neighbours(self, state):
        """
        Neighbouring states according to the initial model used by MPC. Does
        not reflect changes due to faults.
        """
        return self.itmatrix[state, :]

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
        visited = set()         # set of all states visited
        tree = [(None, state, 0, None)] # (parent ref, state, depth, action)
        while len(tree):
            cnode = tree.pop()
            if cnode[2] == self.depth+1:
                break
            visited.add(cnode[1])
            # add eligible states to be explored to tree
            for action, nstate in enumerate(self.model_neighbours(cnode[1])):
                if nstate not in visited:
                    node = (cnode, nstate, cnode[2]+1, action)
                    tree.insert(0, node)
                    visited.add(nstate)
                    # check state eligibility
                    if self.dmap[nstate] < min_dist:
                        min_dist = self.dmap[nstate]
                        optimal = node
        # Trace back to first action
        if optimal is None: # i.e. starting state is closest state
            return None
        while optimal[0] is not None:
            action = optimal[3]
            optimal = optimal[0]
        return action



def adaptive_path(tb, start, exploration, faultfunc, *args):
    """
    Calculates path from start to a goal state while adapting to intermittent
    faults in the environment. At occurance of a fault, relearns policy in the
    vicinity of the state.
    Proceeds as follows:

    Args:
        tb (TestBench): A TestBench instance with learner set up.
        start (tuple): The [Y, X] coordinates to start from.
        exploration (float): Probability of exploring at each step (==0 for MPC)
        faultfunc (func): A function that accepts the testbench and any other
            positional arguments to create a fault in the system.

    Returns:
        - A list of coordinates [y, x] traversed,
        - A list of state indices traversed equal to the path length limit. If
            goal reached earlier, the rest is padded with the last state.
        - The coordinate of the final state reached,
        - Length of path to goal
        - True if goal reached, else False
    """
    i = 0                       # segments completed
    n = 0                       # iterations completed
    state = tb.coord2state(start)
    coords = [start]           # list of coords
    traversed = [state]        # list of corresponding state numbers

    # initially learn w/o faults
    tb.learner.learn(coverage=1, ep_mode='bfs')
    faultfunc(tb, *args)
    
    while not tb.learner.goal(state) and n < tb.num_states:
        # exploring
        if tb.learner.random.rand() < exploration:
            history, _ = tb.learner.learn(episodes=[state])
            traversed.extend(history[0])
            coords.extend([tb.state2coord(s) for s in history[0]])
            state = traversed[-1]
        # exploiting
        else:
            action = tb.learner.recommend(state)
            # stop path if no recommendation
            if action is None:
                break
            state = tb.learner.next_state(state, action)
            traversed.append(state)
            coords.append(tb.state2coord(state))
        
        n = len(traversed)
    
    traversed.extend([traversed[-1]] * (tb.num_states-n))
    return coords, traversed, tb.state2coord(state), n, tb.learner.goal(state)



def fault(tb, ftype=None, faultall=False):
    """
    * Picks a random action to fault,
    * Picks a random number of states where that action will be faulty,
    * For each fault state, the fault action gets 'shorted' with the results
        of one of the other actions or does nothing.
    * i.e. Transition/ reward matrix changes

    Args:
        tb (TestBench): A testbench with a learner attached with r/t matrices
            populated.
        ftype (tuple/list): List of integers. [action1, action2]. Where action1's
            output is overwritten by action2's output. If -1, randomly chosen.
        faultall (bool): Whether to introduce fault in all states or a randomly
            chosen subset of states.
    """
    if ftype is None:
        return
    ftype = [tb.random.randint(learner.num_actions) if a == -1 else a for a in ftype]
    for i in range(0, len(ftype), 2):
        # Number of states to create the action fault in
        if faultall:
            num_faults = tb.learner.num_states
            states = np.arange(0, num_faults, step=1, dtype=int)
        else:
            num_faults = tb.random.randint(1, tb.learner.num_states+1)
            # The state indices where the fault occurs
            states = random.choice(tb.learner.num_states, num_faults, replace=False)
        # Applying changes
        tb.learner.tmatrix[states, ftype[i]] = learner.tmatrix[states, ftype[i+1]]
        tb.learner.rmatrix = tb.create_rmatrix(tb.goals, tb.topology, tb.tmatrix)



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

    # np.seterr(all='ignore')

    args = ArgumentParser()
    args.add_argument('-t', '--topology', metavar='T', type=int,
                      help="Topology grid size", default=10)
    args.add_argument('-a', '--start', metavar=('Y', 'X'), type=int, nargs=2,
                      help="Starting coordinates", default=None)
    args.add_argument('-r', '--rate', metavar='R', type=float,
                      help="Learning rate (0, 1]", default=0.1)
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
                      help="The action selection policy", default='greedy')
    args.add_argument('-o', '--online', action='store_true',
                      help="Online or Offline policy update", default=False)
    args.add_argument('-f', '--fault', metavar=('F1', 'F2'), type=int, nargs='+',
                      help="Pairs of actions to fault. F2's output replaces F1.", default=None)
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
    args.add_argument('--showfield', action='store_true',
                      help="Show optimal action field for RL methods.", default=False)
    args = args.parse_args()

    # Print paramters
    for key, value in vars(args).items():
        print(key, ': ', value)
    
    random = np.random.RandomState(args.seed)
    paths = []          # stores segments for each trial
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
            tb = TestBench(size=args.topology, seed=seed, learner=None)
            dmap = distance_map(tb.goals, args.topology, args.topology, flatten=True)
            learner = QLearner(lrate=args.rate, discount=args.discount,
                        depth=args.maxdepth, steps=args.steps, policy=args.policy,
                        max_prob=args.greedyprob, mode=mode, rmatrix=tb.rmatrix,
                        tmatrix=tb.tmatrix, goal=tb.goals, seed=seed)
            tb.learner = learner
            fault(tb, args.fault)
            

            # Run simulation with intermittent faults and adaptive learning
            # tb.learner.learn(coverage=args.coverage)
            coords, traversed, final, length, goal = adaptive_path(tb, start, args.explore, fault, args.fault)
            if args.numtrials == 1:
                print_results(coords, final, length, goal)
                # Show optimal and adaptive paths to goal state
                tb.show_topology(path=coords, showfield=args.showfield)
        
        
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
            tb = TestBench(size=args.topology, seed=seed, learner=None)
            learner = FLearner(lrate=args.rate, discount=args.discount,
                        depth=args.maxdepth, steps=args.steps, policy=args.policy,
                        max_prob=args.greedyprob, mode=mode, rmatrix=tb.rmatrix,
                        tmatrix=tb.tmatrix, goal=tb.goals, seed=seed,
                        func=func, dfunc=dfunc, funcdim=funcdim,
                        stateconverter=FlagGenerator(tb.size, tb.size),
                        actionconverter=FlagGenerator(2,2))
            tb.learner = learner
            fault(tb, args.fault)
            # define stepsize for heirarchical learning
            if args.hierarchical:
                deltaheight = np.amax(tb.topology) - np.amin(tb.topology)
                def stepsize(state):
                    cr, cc = tb.state2coord(state)
                    height = tb.topology[cr, cc]
                    return int(np.sqrt(height * tb.size / deltaheight)) + 1
                tb.learner.stepsize = stepsize

            dmap = distance_map(tb.goals, args.topology, args.topology, flatten=True)

            # Run simulation with intermittent faults and adaptive learning
            # tb.learner.learn(coverage=args.coverage)
            coords, traversed, final, length, goal = adaptive_path(tb, start, args.explore, fault, args.fault)
            if args.numtrials == 1:
                print_results(coords, final, length, goal)
                # Show optimal and adaptive paths to goal state
                tb.show_topology(path=coords, showfield=args.showfield)

        
        # Use model predictive control (mpc)
        elif args.usempc:
            # set up testbench
            tb = TestBench(size=args.topology, seed=seed, learner=None)
            dmap = distance_map(tb.goals, args.topology, args.topology, flatten=True)
            learner = ModelPredictiveController(dmap=dmap, seed=tb.seed,
                                            depth=args.maxdepth, goal=tb.goals,
                                            tmatrix=tb.tmatrix, rmatrix=tb.rmatrix)
            tb.learner = learner
            fault(tb, args.fault)
        # Run simulation with intermittent faults and adaptive learning
            coords, traversed, final, length, goal = adaptive_path(tb, start, 0, fault, args.fault)
            if args.numtrials == 1:
                print_results(coords, final, length, goal)
                # Show optimal and adaptive paths to goal state:
                tb.show_topology(path=coords, showfield=False)
        
        # append results to list
        paths.append(traversed)
        successes.append(goal)
    
    # Analyse results from one of three approaches (tabular, functional, mpc)
    hmap = tb.topology.reshape(dmap.shape)
    allstates = [s for states in paths for s in states]
    havg = np.mean(hmap[allstates])
    cavg = np.mean(dmap[[path[-1] for path in paths]])
    davg = np.mean(dmap[allstates])
    tavg = 1 if args.usempc else args.explore
    print('=====')
    print('havg: %5.2e\tcavg: %5.2f\tdavg: %5.2f\ttavg: %5.2f\t' % (havg, cavg, davg, tavg))
    print('=====')
