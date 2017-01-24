"""
Tests for the qlearn package.
"""

import os
import numpy as np
from qlearner import QLearner

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
    Testing QLearner with initial arguments.
    """
    # Set-up:
    rmatrix_sq = np.random.rand(4, 4)
    rmatrix_rec = np.random.rand(4, 3)
    tmatrix = np.random.randint(0, 4, size=(4, 3))
    goal_l = (1, 2, 3)
    goal_f = lambda x: x < 1
    np.savetxt('test.dat', rmatrix_sq)

    # Test 1: list goal
    temp = QLearner(rmatrix_sq, goal_l)
    assert np.array_equal(temp.rmatrix, rmatrix_sq), "R matrix not equal to arg."

    # Test 2: function goal
    QLearner(rmatrix_sq, goal_f)

    # Test 3: File I/O
    temp = QLearner('test.dat', goal_l, 'test.dat')
    assert temp.qmatrix.shape == rmatrix_sq.shape, "Q & R matrix dimension mismatch."
    assert np.array_equal(temp.rmatrix, rmatrix_sq), "R matrix not equal to arg."

    # Test 4: rectangular r matrix, no tmatrix
    try:
        QLearner(rmatrix_rec, goal_l)
    except ValueError:
        pass

    # Test 5: rectangular r matrix, t matrix of same dimension
    QLearner(rmatrix_rec, goal_f, tmatrix)

    # Finalize
    os.remove('test.dat')
    global QLEARNER
    QLEARNER = test


@test
def test_offline_learning():
    """
    Testing uniform, softmax, greedy offline learning.
    """
    pass


@test
def test_online_learning():
    """
    Testing uniform, softmax, greedy online learning.
    """
    pass



if __name__ == '__main__':
    print()
    test_instantiation()

    print('\n==========\n')
    print('Tests passed:\t' + str(TESTS_PASSED))
    print('Total tests:\t' + str(NUM_TESTS))
    print()
