"""
This script implements adaptive fault-tolerant control of a sample system
using reinforcement learning. The environment is a topology. The objective is
to reach the lowest altitude. Actions are movements to adjacent grid points.
Faults introduced mix the outcome of one action with another. At occurance of
every fault, the policy is updated by learning from the current state's
neighbourhood. The path traversed is divided into segments in between faults.

Usage:

    > python adaptive.py -h
"""

from argparse import ArgumentParser
import numpy as np
from qlearn import TestBench
from qlearn import QLearner



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
    """
    segment = [start]
    i = 0   # segments completed
    n = 0   # iterations completed
    segments = {}
    state = tb.coord2state(start)
    while not tb.learner.goal(state) and n <= 2 * tb.num_states:
        if fault() or n == 0:
            segment = [segment[-1]]
            segments['{:03d}'.format(i)] = segment
            episodes = tb.learner.neighbours(state)
            tb.learner.learn(episodes=episodes)
            i += 1
        action = tb.learner.recommend(state)
        state = tb.learner.next_state(state, action)
        segment.append(tb.state2coord(state))
        n += 1
    return segments, tb.state2coord(state), n



def fault_func(learner, probability, random):
    """
    Defines a function that introduces faults in the environment. The function
    is called after every action in adaptive_path().

    Args:
        learner (QLearner): The learner instance with rmatrix and tmatrix
            populated.
        probability (float): The probability of a fault occuring.
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
            # Number of states to creat the action fault in
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




if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-t', '--topology', metavar='T', type=int,
                      help="Topology grid size", default=10)
    args.add_argument('-a', '--start', metavar=('Y', 'X'), type=int, nargs=2,
                      help="Starting coordinates", default=None)
    args.add_argument('-r', '--rate', metavar='R', type=float,
                      help="Learning rate (0, 1]", default=1e-1)
    args.add_argument('-d', '--discount', metavar='D', type=float,
                      help="Discount factor (0, 1]", default=0.5)
    args.add_argument('-e', '--explore', metavar='E', type=float,
                      help="Exploration while recommending actions [0, 1]", default=0.)
    args.add_argument('-s', '--steps', metavar='S', type=int,
                      help="Number of steps to look ahead during learning", default=1)
    args.add_argument('-m', '--maxlimit', metavar='M', type=int,
                      help="Number of steps at most in each episode", default=1)
    args.add_argument('-f', '--faultprob', metavar='F', type=float,
                      help="Probability of fault after each action", default=0.25)
    args.add_argument('--seed', metavar='SEED', type=int,
                      help="Random number seed", default=None)
    args = args.parse_args()


    # Initialize the QLearner with reward/transition matrices
    tb = TestBench(size=args.topology, seed=args.seed, learner=QLearner,
                   lrate=args.rate, discount=args.discount, exploration=args.explore,
                   depth=args.maxlimit, steps=args.steps, policy=QLearner.UNIFORM)
    # Create a fault function that alters the environment
    fault = fault_func(tb.learner, args.faultprob, tb.random)
    # Get the starting point
    args.start = tb.random.randint(0, tb.size, 2) if args.start is None else args.start
    # Print paramters
    for key, value in vars(args).items():
        print(key, ': ', value)
    # Calculate optimal path to goal based on initial conditions
    optimal = tb.shortest_path(point=args.start)
    # Run simulation with intermittent faults and adaptive learning
    segments, final, length = adaptive_path(tb, args.start, fault)
    print('\nsegments: ', len(segments))
    print('path length: ', length)
    print('final state: ', final)
    print()
    # Show optimal and adaptive paths to goal state
    tb.show_topology(**segments, Optimal=optimal)
