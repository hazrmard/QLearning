"""
This script implements reinforcement learning on a system of interconnected
fuel tanks. The objective is to balance the tanks by minimizing the moment
about the center. The state space is continuous and the action space is
descrete. The system is implemented as a circuit created programmatically.

For N tanks, there are N-choose-2 current sources that act as valves to transfer
charge/fuel between any 2 tanks. Only 1 valve can be active at a time. The system
is created during runtime (as opposed to loading a netlist file).

For another implementation of the tanks system based on a netlist, see 
tanksnetlistdemo.py

Usage:

    > python demo.py -h

Requires:
    flask,
    numpy,
    ahkab
"""

from argparse import ArgumentParser
import flask
import numpy as np
from ..qlearn import SLearner
from ..qlearn.linsim import Netlist
from ..qlearn.linsim import Directive
from ..qlearn.linsim import Simulator
from ..qlearn.linsim import FlagGenerator
from ..qlearn.linsim import elements
from ..qlearn import utils



def create_system(num_tanks=4, tank_levels=5, lrate=1e-2, discount=0.75, exploration=0, steps=1):
    """
    Creates a Simulator that represents the fuel tank system. Each fuel tank
    is a capacitor, and the tank's level is the charge (proportional to voltage)
    on the capacitor. The level is increased or decreased by changing the values
    of attached current sources (fuel pumps) connecting different tanks.

    The state of the system is defined by the vector:
        [tank 1 voltage, tank 2 voltage,..., tank n voltage]
    Actions to be taken are one-hot encodings of pumps/current sources. The
    action vector has only one element. That value determines which pump is
    turned on in forwards/reverse direction. Only one pump is active at one time.

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
    # where dt is the duration/stepsize of the simulation. This 'I' value is used
    # when fuel pumps are on to simulate a 1V change in tank levels.
    capacitance = 1e-3
    deltat = 1e-1
    idc = capacitance / deltat

    # Creating the circuit describing the system
    system = Netlist('Fuel Tanks')
    system.add_directive(Directive('.ic'))
    num_pumps = 0       # number of fuel pumps/current sources b/w tanks
    for i in range(num_tanks):
        system.add(elements.Capacitor('c'+str(i), 'n'+str(i), '0', capacitance))
        for j in range(i+1, num_tanks):
            num_pumps += 1
            # Current source direction is low node number -> high node number.
            # The first [num_tanks choose 2] sources are between tanks.
            system.add(elements.CurrentSource('i'+str(i)+'_'+str(j), 'n'+str(i), 'n'+str(j),
                                              type='idc', idc=0))

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
        ic = netlist.directives['ic'][0]
        for i in range(num_tanks):
            ic.param('v(n'+str(i)+')', svec[i])
        pump = (avec[0] + 1) // 2 - 1       # pump index corresponding to action
        reverse = avec[0] % 2               # odd - > reversed, even - > forward
        current = idc if reverse == 0 else -idc
        for i, source in enumerate(netlist.elements_like('i')):
            if i == pump:
                source.param('idc', current)
            else:
                source.param('idc', 0)
        return netlist

    # Defining state_demux function which converts simulation results into a
    # state vector to be used in the next iteration
    def state_demux(svec, avec, netlist, result):
        nsvec = np.zeros(num_tanks)
        pump = (int(avec[0]) + 1) // 2 - 1       # pump index corresponding to action
        for i in range(num_tanks):
            nsvec[i] = result['v(n' + str(i) + ')']
            # Correcting negative voltages by transferring voltage deficit.
            # -ive voltage means that more charge was transferred than existed
            # in the tank.
            if nsvec[i] < 0:
                pump_name = netlist.elements_like('i')[pump].name   #i<src>_<dst>
                src, dst = pump_name.split('_')
                src = int(src[1:])
                dst = int(dst)
                other = src if src != i else dst
                result['v(n' + str(other) + ')'] += nsvec[i]
                nsvec[other] += nsvec[i]
                result['v(n' + str(i) + ')'] = 0
                nsvec[i] = 0
            # Correcting excessive voltages by transferring voltage excess.
            # > tank_levels-1 voltage means that more charge was transferred than
            # allowed in the tank.
            elif nsvec[i] > tank_levels-1:
                pump_name = netlist.elements_like('i')[pump].name   #i<src>_<dst>
                src, dst = pump_name.split('_')
                src = int(src[1:])
                dst = int(dst)
                other = src if src != i else dst
                result['v(n' + str(other) + ')'] += nsvec[i] - (tank_levels-1)
                nsvec[other] += nsvec[i] - (tank_levels-1)
                result['v(n' + str(i) + ')'] = tank_levels-1
                nsvec[i] = tank_levels-1
        return nsvec

    # Creating the simulator to be used for learning
    sim = Simulator(env=system, timestep=deltat/10, state_mux=state_mux,
                    state_demux=state_demux)

    # Defining the reward function which penalizes imbalance in fuel tanks
    def reward(svec, avec, nsvec):
        left = nsvec[:num_tanks // 2]                   # left tank levels
        right = nsvec[num_tanks // 2 + num_tanks % 2:]  # right tank levels
        diff = [(num_tanks // 2 - i) * (left[i] - right[-i - 1]) for i in range(num_tanks // 2)]
        return -abs(sum(diff))                          # net moment due to tanks

    # Defining the function approximation vectors and the goal function
    def dfunc(svec, avec, weights):
        pumpvec = np.zeros(num_pumps)
        pump = (int(avec[0]) + 1) // 2 - 1       # pump index corresponding to action
        reverse = int(avec[0]) % 2               # odd - > reversed, even - > forward
        if avec[0] != 0:
            pumpvec[pump] = 1 if reverse == 0 else -1
        return np.concatenate((np.square(svec) / tank_levels**2,
                               np.array(svec) / tank_levels,
                               pumpvec,
                               [1]))

    def func(svec, avec, weights):
        return np.dot(weights, dfunc(svec, avec, weights))

    funcdim = 2*num_tanks + num_pumps + 1

    def goal(svec):
        return abs(reward(None, None, svec)) <= 1

    # Creating the SLearner instance
    learner = SLearner(reward=reward, simulator=sim, stateconverter=fstate,
                       actionconverter=faction, func=func, funcdim=funcdim,
                       dfunc=dfunc, goal=goal, steps=steps, lrate=lrate,
                       discount=discount, exploration=exploration, stepsize=deltat)
    return learner



def create_server(learner, T, N):
    """
    Sets up a Flask server to send system status in JSON format.

    Args:
        learner (SLearner): The learner instance to visualize
        T (int): Number of tanks.
        N (int): Levels per tank [0, N). N-1 is the max voltage on tank.

    Returns:
        A Flask instance
    """
    svec = np.zeros(T, dtype=float)
    avec = np.zeros(1, dtype=int)
    app = flask.Flask('Demo', static_url_path='', static_folder='', template_folder='')

    @app.route('/')
    def demo():
        svec[:] = np.random.random(T) * (N - 1)
        avec[:] = learner.next_action(svec)
        return flask.render_template('demo.html', N=N, T=T, L=list(range(T)),
                                     O=list(range(T)))


    @app.route('/status/')
    def status():
        s = list(svec)                              # cache last results
        a = list(avec)
        w = list(learner.weights)
        svec[:] = learner.next_state(svec, avec)    # compute new results
        avec[:] = learner.recommend(svec)
        if a[0] != 0:
            pump_name = learner.simulator.env.elements_like('i')[(a[0] - 1) // 2].name
            src, dst = pump_name.split('_')
            src = src[1:]
            dst = dst
            reverse = int(a[0]) % 2
            action = '%s to %s' % ((src, dst) if reverse == 0 else (dst, src))
        else:
            action = ''
        return flask.jsonify(levels=[str(i) for i in s],
                             action=action,
                             weights=[str(i) for i in w],
                             imbalance=learner.reward(None, None, s))    # return cached results

    return app





if __name__ == '__main__':
    # Set up command-line arguments
    args = ArgumentParser()
    args.add_argument('-t', '--tanks', metavar='T', type=int,
                      help="Number of tanks", default=2)
    args.add_argument('-n', '--num_levels', metavar='N', type=int,
                      help="Number of levels per tank", default=5)
    args.add_argument('-c', '--coverage', metavar='C', type=float,
                      help="Fraction of states to cover in learning", default=0.2)
    args.add_argument('-r', '--rate', metavar='R', type=float,
                      help="Learning rate (0, 1]", default=1e-2)
    args.add_argument('-d', '--discount', metavar='D', type=float,
                      help="Discount factor (0, 1]", default=0.5)
    args.add_argument('-e', '--explore', metavar='E', type=float,
                      help="Exploration while recommending actions [0, 1]", default=0.)
    args.add_argument('-s', '--steps', metavar='S', type=int,
                      help="Number of steps to look ahead during learning", default=1)
    args.add_argument('-m', '--maxdepth', metavar='M', type=int,
                      help="Number of steps at most in each episode", default=1)
    args.add_argument('-l', '--load', metavar='F', type=str,
                      help="File to load learned policy from", default='')
    args.add_argument('-f', '--file', metavar='F', type=str,
                      help="File to save learned policy to", default='')
    args.add_argument('-x', '--server', action='store_true',
                      help="Run server on localhost:5000 to visualize problem")
    args = args.parse_args()

    # Set up the learner environment
    learner = create_system(args.tanks, args.num_levels, args.rate, args.discount,
                            args.explore, args.steps)
    print('System Netlist:')
    print(learner.simulator.env)

    # Loading weights or learning new policy
    if args.load == '':
        input('\nPress Enter to begin learning.')
        learner.learn(coverage=args.coverage, depth=args.maxdepth)
        if args.file != '':
            utils.save_matrix(learner.weights, args.file)
    else:
        learner.weights = utils.read_matrix(args.load)

    if not args.server:
        # Setting up in-console session
        svec = np.random.random(args.tanks) * (args.num_levels - 1)
        avec = learner.next_action(svec)

        print('\nEnter "c" to exit loop...')
        while input() != 'c':
            if avec[0] != 0:
                pump = learner.simulator.env.elements_like('i')[(avec[0] - 1) // 2].name
                pump += ' ->' if avec[0] % 2 == 0 else ' <-'
            else:
                pump = 'All off'
            print('Tank Levels: ', *["%10.2f" % s for s in svec], '\tAction: ' + pump, end=' ')
            svec = learner.next_state(svec, avec)
            avec = learner.recommend(svec)
    else:
        # Setting up an interactive server
        create_server(learner, args.tanks, args.num_levels).run()
