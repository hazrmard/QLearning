"""
Action Selection Policy Comparison

This script compares various action selection policies implemented by the
QLearner class.

An action selection policy is how the Q-Learning algorithm selects action from
a given state to calculate the utility of that (state -> action) pair. The
QLearner class implements three policies:

* QLearner.UNIFORM: Each action has the same chance of being chosen from any
    state. That is, the utility of taking an action from a given state does not
    affect the chances of that action being chosen later for exploration.

* QLearner.GREEDY: For each state, the action with the highest utility is
    chosen with a certain probability. All the other actions are uniformly
    chosen at random.

* QLearner.SOFTMAX: The probability for an action being chosen for a particular
    state is proportional to its utility for that state. At the start of the
    learning algorithm all actions are chosen uniformly. Later as the utilities
    of each action change, so do their probabilities of being chosen.

This script uses 2 metrics to compare performance:

* Relative Path Length: The number of states traversed from point to goal by
    each policy divided by the path length achieved by greedy search.

* Relative Average Path Height: The average height in the topology over the
    policy path divided by the average topology height over the greedy path.
"""

import numpy as np
import matplotlib.pyplot as plt
from qlearn import QLearner
from qlearn import TestBench

NUM_TRIALS = 5
EPISODES_PER_TRIAL = 200
TOPOLOGY_SIZE = 10
GOALS = TOPOLOGY_SIZE
TOPOLOGY_METHOD = 'fault'
WRAP = False
POLICIES = (QLearner.UNIFORM, QLearner.GREEDY, QLearner.SOFTMAX)
policy_str = ('Uniform', 'Greedy', 'Softmax')
LEARNING_RATE = 0.25
DISCOUNT_FACTOR = 1
GREEDY_MAX_PROB = 0.25
MODE = QLearner.OFFLINE

# RPATHS is a 3D arraf of path lengths from a point to a goal relative to the
# path length computed by a greedy algorithm.
RPATHS = np.zeros((NUM_TRIALS, EPISODES_PER_TRIAL, len(POLICIES)))
# RAVGHT is a 3D array of average height over the path from point to goal
# relative to the average height over the greedy path.
RAVGHT = np.zeros_like(RPATHS)

# Each trial generates its own terrian defined by the trial # which becomes
# the seed of the random number generator for the TestBench instance.
for trial in range(NUM_TRIALS):
    tb = []

    # For a trial/seed #, TestBench instances for each policy are created and
    # the Q-Matrix learned.
    for i, policy in enumerate(POLICIES):
        testbench = TestBench(size=TOPOLOGY_SIZE, seed=trial, method=TOPOLOGY_METHOD,\
                    goals=TOPOLOGY_SIZE, wrap=WRAP, lrate=LEARNING_RATE,\
                    discount=DISCOUNT_FACTOR, policy=policy, max_prob=GREEDY_MAX_PROB)
        testbench.qlearner.learn()
        tb.append(testbench)

    # Each episode generates a random point which is used by all policies to
    # find a path to a goal state. The Path is compared against one obtained
    # using a greedy algorithm.
    for episode in range(EPISODES_PER_TRIAL):
        point = (tb[0].random.randint(TOPOLOGY_SIZE), tb[0].random.randint(TOPOLOGY_SIZE))
        greedy_path = tb[0].shortest_path(point=point)
        gy, gx = zip(*greedy_path)
        greedy_len = len(greedy_path)
        greedy_ht = tb[0].topology[gy, gx]
        greedy_ht += np.min(greedy_ht)
        greedy_ht = np.sum(greedy_ht) / greedy_len

        for p in range(len(POLICIES)):
            path = tb[p].episode(start=point, interactive=False)
            # tb[p].show_topology(Qpath=path, Greedy=greedy_path)
            py, px = zip(*path)
            RPATHS[trial, episode, p] = len(path) / greedy_len
            path_ht = tb[p].topology[py, px]
            path_ht += np.min(path_ht)
            path_ht = np.sum(path_ht) / len(path)
            RAVGHT[trial, episode, p] = path_ht / greedy_ht

PRPATHS = np.zeros(len(POLICIES))    # Policy-wise relative path length
PRAVGHT = np.zeros_like(PRPATHS)      # Policy-wise relative average height
for p in range(len(POLICIES)):
    PRPATHS[p] = np.average(RPATHS[:, :, p])
    PRAVGHT[p] = np.average(RAVGHT[:, :, p])




ind = np.arange(len(POLICIES))  # the x locations for the groups
width = 0.35                    # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, PRPATHS, width, color='r')
rects2 = ax.bar(ind + width, PRAVGHT, width, color='y')

# add some text for labels, title and axes ticks
ax.set_title('Path lengths and average heights relative to greedy approach.')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(policy_str)
ax.legend((rects1[0], rects2[0]), ('Path Length', 'Path Height'))
plt.show()
