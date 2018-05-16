"""
This module defines functions that implement special algorithms used by the
TestBench class.
"""

import numpy as np
try:
    from linsim import Netlist
    from linsim import Simulator
    from linsim import Directive
except ImportError:
    from .linsim import Netlist
    from .linsim import Simulator


def create_sim_env(size, random):
    """
    Creates the environment for SLearner. The environment is 2 capacitors
    with an initial charge being fed by separate current sources. Each capacitor
    has a feeding source and a draining source. The voltage level of the
    capacitor current to ground is dictated by the connected current sources'
    states.

    The state vector is [c1 charge, c2 charge]
    The action vector is one of [(0, 0), (0, 1), (1, 0), (1, 1)] corresponding
    to the 4 action vectors in TestBench.actions.

    Args:
        size (int): A measure of the number of states. Corresponds to the number
            of levels in each capacitor.
        random (np.random.RandomState): A random number generator for consistent
            terrain generation given the seed for TestBench.

    Returns:
        A Simulator instance.
    """
    TS = 1e-1   # timestep/duration for simulator
    CAP = 1e-3  # capacitance
    net = ("*SLearner circuit",
           "C1 n1 0 " + str(CAP),
           "I1 0 n1 type=idc idc=1e-2",
           "I2 n1 0 type=idc idc=1e-2",
           "C2 n2 0 " + str(CAP),
           "I3 0 n1 type=idc idc=1e-2",
           "I4 n2 0 type=idc idc=1e-2",
           ".end")
    netinstance = Netlist(name='slearner', netlist=net)

    def state_mux(svec, avec, netlist):
        ic = netlist.directives.get('ic')   # ret. None or list of ic directives
        if ic is None:
            ic = Directive('.ic')
            netlist.add_directive(ic)
        else:
            ic = ic[0]                      # we'll only have 1 ic directive

        ic.param('v(n1)', svec[0])
        ic.param('v(n2)', svec[1])
        # Testbench.actions defines the 4 action vectors in relation to
        # movement on the topology. This mux emulates that so the first
        # action in this environment corresponds to the first action in
        # the topology. For e.g. if Testbench.actions[0] = [0, -1] i.e. move
        # back in x direction [(y, x) coords], then the first avec = [0, 0]
        # drains voltage from the first capacitor, while holding second
        # capacitor constant.
        if avec[0] == 0 and avec[1] == 0:
            netlist.element('i1').param('idc', 0)
            netlist.element('i2').param('idc', CAP / TS)
            netlist.element('i3').param('idc', 0)
            netlist.element('i4').param('idc', 0)
        elif avec[0] == 0 and avec[1] == 1:
            netlist.element('i1').param('idc', CAP / TS)
            netlist.element('i2').param('idc', 0)
            netlist.element('i3').param('idc', 0)
            netlist.element('i4').param('idc', 0)
        elif avec[0] == 1 and avec[1] == 0:
            netlist.element('i1').param('idc', 0)
            netlist.element('i2').param('idc', 0)
            netlist.element('i3').param('idc', 0)
            netlist.element('i4').param('idc', CAP / TS)
        elif avec[0] == 1 and avec[1] == 1:
            netlist.element('i1').param('idc', 0)
            netlist.element('i2').param('idc', 0)
            netlist.element('i3').param('idc', CAP / TS)
            netlist.element('i4').param('idc', 0)
        return netlist


    def state_demux(prevsvec, prevavec, netlist, result):
        return (np.clip(result['v(n1)'], 0, size-1),
                np.clip(result['v(n2)'], 0, size-1))

    sim = Simulator(env=netinstance, timestep=TS, state_mux=state_mux,
                    state_demux=state_demux, ic=None)
    return sim


def fault_algorithm(iterations, size, random):
    """
    The Fault Algorithm generates a terrain by drawing a line of a random
    gradient through the grid, and be elevating points on one side, and
    lowering points on the other side for some number of iterations.

    Args:
        iterations (int): Number of times to generate faults.
        size (tuple): The size of the topology (rows, columns).
        random (np.random.RandomState): A random number generator for consistent
            terrain generation given the seed for TestBench.

    Returns:
        A 2D ndarray of dimensions=size where array[y, x] is altitude at
        that point in the topology.
    """
    topology = np.zeros(size)
    angle = random.rand(iterations) * 2 * np.pi
    a = np.sin(angle)
    b = np.cos(angle)
    disp = (random.rand() / iterations) * (np.arange(iterations)[::-1] + 1)
    for i in range(iterations):
        cx = random.rand() * size[1]
        cy = random.rand() * size[0]
        for x in range(size[1]):
            for y in range(size[0]):
                topology[y, x] += disp[i] \
                                        if -a[i]*(x-cx) + b[i]*(y-cy) > 0 \
                                        else -disp[i]
    topology = topology - np.amin(topology)
    return topology / np.amax(topology)


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
