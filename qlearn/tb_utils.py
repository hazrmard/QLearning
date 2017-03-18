"""
This module defines functions that implement special algorithms used by the
TestBench class.
"""

import numpy as np



def fault_algorithm(iterations, size):
    """
    The Fault Algorithm generates a terrain by drawing a line of a random
    gradient through the grid, and be elevating points on one side, and
    lowering points on the other side for some number of iterations.

    Args:
        iterations (int): Number of times to generate faults.
        size (tuple): The size of the topology (rows, columns).

    Returns:
        A 2D ndarray of dimensions=size where array[y, x] is altitude at
        that point in the topology.
    """
    topology = np.zeros(size)
    angle = np.random.rand(iterations) * 2 * np.pi
    a = np.sin(angle)
    b = np.cos(angle)
    disp = (np.random.rand() / iterations) * (np.arange(iterations)[::-1] + 1)
    for i in range(iterations):
        cx = np.random.rand() * size[1]
        cy = np.random.rand() * size[0]
        for x in range(size[1]):
            for y in range(size[0]):
                topology[y, x] += disp[i] \
                                        if -a[i]*(x-cx) + b[i]*(y-cy) > 0 \
                                        else -disp[i]
    return topology


def abs_cartesian(topology, source, target):
    """
    Computes the edge weight/distance between adjacent states/points in the
    topology. If target is downwards from source then reward is the rectangular
    distance (on x-y plane) between source and target. Otherwise returns the
    Euclidean distance between source and target.
    In state space terms, in case of a positive reward only registers the
    number of state changes between source and target. In case of a negative
    reward, accounts for the number of state changes and the cost of the reward.
    Used by TestBench.shortest_path() as a metric.

    Args:
        topology (2D ndarray: A TestBench.topology array.
        source (tuple/list/ndarray): The (y, x)/(row, column) coordinates of
            the source point.
        target (tuple/list/ndarray): The (y, x)/(row, column) coordinates of
            the target point.

    Returns:
        The distance measure between source and target (float).
    """
    state_diff = abs(source[0]-target[0]) + abs(source[1]-target[1])
    height_diff = topology[target[0], target[1]] - topology[source[0], source[1]]
    if height_diff <= 0:
        return float(state_diff)
    else:
        return np.sqrt(state_diff**2 + height_diff**2)
