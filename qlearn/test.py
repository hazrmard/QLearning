"""
Tests for the qlearn package.
"""

import os
import numpy as np
try:
    from qlearner import QLearner
    from flearner import FLearner
    from slearner import SLearner
    from testbench import TestBench
    from linsim import FlagGenerator
except ImportError:
    from .qlearner import QLearner
    from .flearner import FLearner
    from .slearner import SLearner
    from .testbench import TestBench
    from .linsim import FlagGenerator

NUM_TESTS = 0
TESTS_PASSED = 0
QLEARNER = None

def test(func):
    """
    Decorator for test cases.

    Args:
        func (function): A test case function object.
    """
    global NUM_TESTS
    NUM_TESTS += 1
    def test_wrapper(*args, **kwargs):
        """
        Wrapper that calls test function.

        Args:
            desc (str): Description of test.
        """
        print(func.__doc__.strip(), end='\t')
        try:
            func(*args, **kwargs)
            global TESTS_PASSED
            TESTS_PASSED += 1
            print('PASSED')
        except Exception as ex:
            print('FAILED: ' + str(ex))

    return test_wrapper



@test
def test_instantiation():
    """
    Testing common QLearner initial arguments and support functions.
    """
    # Set-up:
    STATES = 10
    ACTIONS = 5
    rmatrix_sq = np.random.rand(STATES, STATES)
    rmatrix_rec = np.random.rand(STATES, ACTIONS)
    tmatrix = np.random.randint(0, STATES, size=(STATES, ACTIONS))
    # making sure tmatrix points to goal states:
    tmatrix[:, ACTIONS-1] = np.random.randint(0, 1, size=STATES)
    goal_l = (0, 1)
    goal_f = lambda x: x <= 1
    np.savetxt('test.dat', rmatrix_sq)
    global QLEARNER

    # Test 1: list goal
    temp = QLearner(rmatrix_sq, goal_l)
    assert np.array_equal(temp.rmatrix, rmatrix_sq), "R matrix not equal to arg."
    assert temp.goal(0) and temp.goal(1) and not temp.goal(2) and not temp.goal(3), \
            'List goal not working.'
    QLEARNER = temp

    # Test 2: function goal
    temp = QLearner(rmatrix_sq, goal_f)
    assert temp.goal(0) and temp.goal(1) and not temp.goal(2), 'Function goal not working.'
    QLEARNER = temp

    # Test 3: File I/O
    temp = QLearner('test.dat', goal_l)
    assert temp.qmatrix.shape == rmatrix_sq.shape, "Q & R matrix dimension mismatch."
    assert np.array_equal(temp.rmatrix, rmatrix_sq), "R matrix not equal to arg."
    QLEARNER = temp

    # Test 4: rectangular r matrix, no tmatrix
    try:
        QLearner(rmatrix_rec, goal_l)
    except ValueError:
        pass

    # Test 5: rectangular r matrix, t matrix of same dimension
    temp = QLearner(rmatrix_rec, goal_f, tmatrix)
    assert temp.next_state(1, 2) == tmatrix[1, 2], 'Next state prediction incorrect.'
    QLEARNER = temp

    # Test 6: episodes
    l = set(temp.episodes(coverage=1.0, mode='bfs'))
    assert l == set(range(temp.num_states)), 'Full episode coverage failed.'

    # Finalize
    os.remove('test.dat')


@test
def test_offline_learning():
    """
    Testing uniform, softmax, greedy offline learning.
    """
    global QLEARNER
    # Test 1: uniform
    QLEARNER.set_action_selection_policy(QLearner.UNIFORM)
    assert QLEARNER._policy == QLEARNER._uniform_policy, 'Incorrect policy set.'
    assert QLEARNER.policy == QLearner.UNIFORM, 'Incorrect policy mode.'
    QLEARNER.reset()
    QLEARNER.learn()

    # Test 2: greedy
    QLEARNER.set_action_selection_policy(QLearner.GREEDY, max_prob=0.5)
    assert QLEARNER._policy == QLEARNER._greedy_policy, 'Incorrect policy set.'
    assert QLEARNER.policy == QLearner.GREEDY, 'Incorrect policy mode.'
    QLEARNER.reset()
    QLEARNER.learn()

    # Test 3: softmax
    QLEARNER.set_action_selection_policy(QLearner.SOFTMAX)
    assert QLEARNER._policy == QLEARNER._softmax_policy, 'Incorrect policy set.'
    assert QLEARNER.policy == QLearner.SOFTMAX, 'Incorrect policy mode.'
    QLEARNER.reset()
    QLEARNER.learn()



@test
def test_online_learning():
    """
    Testing softmax, greedy online learning.
    """
    global QLEARNER
    # Test 1: greedy
    QLEARNER.set_action_selection_policy(QLearner.GREEDY, mode=QLearner.ONLINE, max_prob=0.5)
    assert QLEARNER._policy == QLEARNER._greedy_policy, 'Incorrect policy set.'
    assert QLEARNER.policy == QLearner.GREEDY, 'Incorrect policy mode.'
    QLEARNER.reset()
    QLEARNER.learn()

    # Test 2: softmax
    QLEARNER.set_action_selection_policy(QLearner.SOFTMAX, mode=QLearner.ONLINE)
    assert QLEARNER._policy == QLEARNER._softmax_policy, 'Incorrect policy set.'
    assert QLEARNER.policy == QLearner.SOFTMAX, 'Incorrect policy mode.'
    QLEARNER.reset()
    QLEARNER.learn()


@test
def qlearner_testbench():
    """
    Testing Qlearner testbench.
    """
    # set up
    size = 10
    mode = 'bfs'
    coverage = 0.5
    exploration = 0.25
    seed = 40000
    lrate = 0.25
    discount = 1
    steps = 3
    start = (0, 0)

    # Test 1: Instantiation
    t = TestBench(size=size, seed=seed, mode=QLearner.ONLINE, wrap=True)
    assert t.learner.mode == QLearner.ONLINE, 'Args not passed on to QLearner.'
    t = TestBench(size=size, goals=[(1, 1), (2, 2), (2*size, size)])
    assert t.num_goals == 2, 'Goal states incorrectly processed.'
    t = TestBench(size=size, seed=seed, mode=QLearner.OFFLINE, wrap=False,\
                    policy=QLearner.SOFTMAX, lrate=lrate, discount=discount, steps=steps)
    t2 = TestBench(size=size, seed=seed, mode=QLearner.OFFLINE, wrap=False,\
                    policy=QLearner.SOFTMAX, lrate=lrate, discount=discount, steps=steps)

    # Test 2: Qlearner compatibility
    t.learner.learn()
    t2.learner.learn()
    assert np.array_equal(t.topology, t2.topology), \
        'Identically seeded topologies not equal.'
    assert np.array_equal(t.learner.tmatrix, t2.learner.tmatrix), \
        'Identically seeded tmatrices not equal.'
    assert np.array_equal(t.learner.rmatrix, t2.learner.rmatrix), \
        'Identically seeded rmatrices not equal.'
    assert np.array_equal(t.learner.qmatrix, t2.learner.qmatrix), \
        'Identically seeded qmatrices not equal.'
    assert t.learner.qmatrix.size == size * size * len(t.actions), \
        'Qlearner matrix size mismatch.'

    # Test 3: Pathfinding
    t.learner.reset()
    t.learner.learn(coverage=coverage, ep_mode=mode)
    t.learner.exploration = exploration
    res = t.episode(start=start, interactive=False)
    assert res == t.path, 'Returned list not equal to stored path.'
    assert len(res) > 0, 'Episode path not computed.'
    res = t.shortest_path(point=start)
    assert len(res) > 0 and res[0] == start, 'Shortest path not computed.'

    # Test 4: Visualization
    t.show_topology(showfield=True, QPath=t.path)


@test
def flearner_testbench():
    """Testing FLearner testbench"""

    # Set up
    size = 10
    ep_mode = 'bfs'
    coverage = 0.25
    exploration = 0.25
    seed = 1000
    lrate = 1e-1
    discount = 1e-2
    steps = 3
    start = (3, 4)
    funcdim = 7
    def func(s, a, w):
        return np.dot(w, np.array([s[0]*a[0]/20, s[1]*a[1]/20, s[0]**2/100, s[1]**2/100,
                                  a[0]**2/4, a[1]**2/4, 1]))
    def dfunc(s, a, w):
        return np.array([s[0]*a[0]/20, s[1]*a[1]/20, s[0]**2/100, s[1]**2/100,
                                  a[0]**2/4, a[1]**2/4, 1])

    # Test 1: Instantiation
    t = TestBench(size=size, seed=seed, learner=FLearner, lrate=lrate,
                  discount=discount, exploration=exploration, func=func,
                  funcdim=funcdim, dfunc=dfunc, steps=steps)

    # Test 2: F learning
    assert t.learner.depth == t.num_states, 'Learning depth not set.'
    t.learner.learn(coverage=coverage, ep_mode=ep_mode)
    res = t.episode(start=start, interactive=False)
    assert res == t.path, 'Returned list not equal to stored path.'
    assert len(res) > 0, 'Episode path not computed.'
    res = t.shortest_path(point=start)
    assert len(res) > 0 and res[0] == start, 'Shortest path not computed.'

    # Test 3: Visualization
    t.show_topology(showfield=True, QPath=t.path, Dijkstra=res)

@test
def slearner_testbench():
    """Testing SLearner testbench"""

    # Set up
    size = 5
    policy = SLearner.UNIFORM
    coverage = 0.3
    exploration = 0.25
    seed = 1000
    lrate = 1e-1
    discount = 1e-2
    steps = 3
    start = (2, 2)
    funcdim = 7
    def dfunc(s, a, w):
        return np.array([s[0]*a[0]/20, s[1]*a[1]/20, s[0]**2/100, s[1]**2/100,
                         a[0]**2/4, a[1]**2/4, 1])
    def func(s, a, w):
        return np.dot(w, dfunc(s, a, w))

    # Test 1: Instantiation
    t = TestBench(size=size, seed=seed, learner=SLearner, lrate=lrate, policy=policy,
                  discount=discount, exploration=exploration, func=func, funcdim=7,
                  dfunc=None, steps=steps)

    # Test 2: S learning
    assert t.learner.depth == t.num_states, 'Learning depth not set.'
    t.learner.learn(coverage=coverage)
    res = t.episode(start=start, interactive=False)
    assert np.array_equal(res, t.path), 'Returned list not equal to stored path.'
    assert len(res) > 0, 'Episode path not computed.'
    res = t.shortest_path(point=start)
    assert len(res) > 0 and res[0] == start, 'Shortest path not computed.'

    # # Test 3: Visualization
    t.show_topology(showfield=True, QPath=t.path, Dijkstra=res)



if __name__ == '__main__':
    print()
    test_instantiation()
    test_offline_learning()
    test_online_learning()
    qlearner_testbench()
    flearner_testbench()
    slearner_testbench()

    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
