"""
Tests for the qlearn package.
"""

import os
import numpy as np
from qlearner import QLearner
from testbench import TestBench

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
    Testing QLearner with initial arguments and support functions.
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
    l = set(temp.episodes(coverage=1.0))
    assert l == set(range(len(temp.rmatrix))), 'Full episode coverage failed.'

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
def test_testbench():
    """
    Testing Qlearner testbench.
    """
    # set up
    size = 10
    mode = 'dfs'
    coverage = 0.5
    exploration = 0.5

    # Test 1: Instantiation
    t = TestBench(size=size, seed=400000, mode=QLearner.ONLINE, wrap=True)
    assert t.qlearner.mode == QLearner.ONLINE, 'Args not passed on to QLearner.'
    t = TestBench(size=size, goals=[(1, 1), (2, 2), (2*size, size)])
    assert t.num_goals == 2, 'Goal states incorrectly processed.'
    t = TestBench(size=size, seed=400000, mode=QLearner.OFFLINE, wrap=False,\
                    policy=QLearner.SOFTMAX)
    t2 = TestBench(size=size, seed=400000, mode=QLearner.OFFLINE, wrap=False,\
                    policy=QLearner.SOFTMAX)

    # Test 2: Qlearner compatibility
    t.qlearner.learn()
    t2.qlearner.learn()
    assert np.array_equal(t.topology, t2.topology), \
        'Identically seeded topologies not equal.'
    assert np.array_equal(t.qlearner.tmatrix, t2.qlearner.tmatrix), \
        'Identically seeded tmatrices not equal.'
    assert np.array_equal(t.qlearner.rmatrix, t2.qlearner.rmatrix), \
        'Identically seeded rmatrices not equal.'
    assert np.array_equal(t.qlearner.qmatrix, t2.qlearner.qmatrix), \
        'Identically seeded qmatrices not equal.'
    assert t.qlearner.qmatrix.size == size * size * len(t.actions), \
        'Qlearner matrix size mismatch.'

    # Test 3: Pathfinding
    t.qlearner.reset()
    t.qlearner.learn(coverage=coverage, ep_mode=mode)
    t.qlearner.exploration = exploration
    res = t.episode(start=(8, 8), interactive=False)
    assert res == t.path, 'Returned list not equal to stored path.'
    assert len(res) > 0, 'Episode path not computed.'
    res = t.shortest_path(point=(8, 8))
    assert len(res) > 0 and res[0] == (8, 8), 'Shortest path not computed.'

    # Test 4: Visualization
    t.show_topology(QPath=t.path, Greedy=res)
    t.episode(start=(8, 8), interactive=True)


if __name__ == '__main__':
    print()
    test_instantiation()
    test_offline_learning()
    test_online_learning()
    test_testbench()

    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
