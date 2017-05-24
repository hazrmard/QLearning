"""
"""

import sys
import flask
import numpy as np
from qlearn import SLearner
from qlearn.linsim import Netlist
from qlearn.linsim import Block
from qlearn.linsim import Directive
from qlearn.linsim import Simulator
from qlearn.linsim import FlagGenerator
from qlearn.linsim import elements



def create_system(num_tanks=4, tank_levels=5):
    """
    Creates a Simulator that represents the fuel tank system. Each fuel tank
    is a capacitor, and the tank's level is the charge (proportional to voltage)
    on the capacitor. The level is increased on decreased by changing the values
    of attached current sources (fuel pumps) connecting different tanks.

    Args:
        num_tanks (int): An even number of tanks.
        tank_levels (int): Number of fuel levels per tank. One level=1V. During
            learning, the states are sampled in 1V intervals. Tank voltage range
            is then from 0 to tank_levels - 1.

    Returns:
        An SLearner instance.
    """
    # For a capacitor (assuming constant current):
    # Q = C * V
    # I * t = C * V
    # I = C * dV/dt
    # So for a fixed dV = 1V, I = C / dt
    # where dt is the duration/timestep of the simulation. This 'I' value is used
    # when fuel pumps are on to simulate a 1V change in tank levels.
    capacitance = 1e-3
    timestep = 1e-1
    idc = capacitance / timestep

    # Creating the circuit describing the system
    system = Netlist('Fuel Tanks')
    system.add_directive(Directive('.ic'))
    num_pumps = 0       # number of fuel pumps/current sources b/w tanks
    for i in range(num_tanks):
        system.add(elements.Capacitor('c'+str(i), 'n'+str(i), '0', capacitance))
        for j in range(i+1, num_tanks):
            num_pumps += 1
            system.add(elements.CurrentSource('i'+str(i)+str(j), 'n'+str(i), 'n'+str(j),
                                              type='idc', idc=idc))

    # Creating FlagGenerators describing the state/action vectors
    # A state is a num_tanks-length vector where the ith element is the level
    # of the ith tank.
    state_flags = [tank_levels] * num_tanks
    fstate = FlagGenerator(*state_flags)
    # Action is a single element vector where:
    #   0    ->  all pumps off
    #   1    ->  1st pump reverse
    #   2    ->  1st pump forward
    #   ︙   ->  ︙
    #   2n-1 ->  nth pump reverse
    #   2n   ->  nth pump forward
    # For a total of 2*num_pumps + 1 flags from 0-2*num_pumps inclusive.
    action_flags = [1 + num_pumps * 2]
    faction = FlagGenerator(*action_flags)

    # Defining state_mux function which converts state and action vectors into
    # changes in the circuit representing that state/action
    def state_mux(svec, avec, netlist):
        svec = np.clip(svec, 0, tank_levels-1)
        ic = netlist.directives['ic'][0]
        for i in range(num_tanks):
            ic.param('v(n'+str(i)+')', "{0:.2f}".format(svec[i]))
        pump = (avec[0] + 1) // 2 - 1       # pump index corresponding to action
        reverse = avec[0] % 2               # 1 - > reversed, 0 - > forward
        current = idc if reverse == 0 else -idc
        for i, source in enumerate(netlist.elements_like('i')):
            if i == pump:
                source.param('idc', current)
            else:
                source.param('idc', 0)
        return netlist

    # Defining state_demux function which converts simulation results into a
    # state vector
    def state_demux(svec, avec, netlist, result):
        for i in range(num_tanks):
            svec[i] = np.clip(result['v(n' + str(i) + ')'], 0, tank_levels-1)
        return svec

    # Creating the simulator to be used for learning
    sim = Simulator(env=system, timestep=timestep, state_mux=state_mux,
                    state_demux=state_demux)

    # Defining the reward function which penalizes imbalance in fuel tanks
    def reward(svec, avec, nsvec):
        left = nsvec[:num_tanks // 2]
        right = nsvec[num_tanks // 2 + num_tanks % 2:]
        diff = [(i+1) * (left[i] - right[-i - 1]) for i in range(num_tanks // 2)]
        return -abs(sum(diff))

    # Defining the function approximation vector and the goal function
    def func(svec, avec):
        return np.concatenate((svec, avec, [1]))

    def goal(svec):
        return abs(reward(None, None, svec)) <= 2

    # Creating the SLearner instance
    learner = SLearner(reward=reward, simulator=sim, stateconverter=fstate,
                       actionconverter=faction, func=func, goal=goal,
                       lrate=1e-2, discount=0.75)
    return learner







if __name__ == '__main__':
    learner = create_system(2)
    learner.learn(coverage=1, verbose=True)
