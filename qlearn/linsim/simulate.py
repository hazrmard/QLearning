"""
This module defines the Simulator class which conducts transient analysis
of a circuit. The simulator uses Ahkab library for calculations.
"""

import os
os.environ['LANG'] = 'en_US.UTF-8'
import copy
import ahkab

# Fixed time-step too small error. Make larger if errors persist.
ahkab.options.transient_max_nr_iter = 30


class Simulator:
    """
    The Simulator calculates behaviour of a circuit described by a netlist over
    time. It remembers its state i.e. after every run() the latest
    potentials/currents are stored as the next run's initial values.

    Note: The netlist should not have inline initial conditions in element
    definitions. Instead they should be provided in special .ic directives or as
    a dict to this class. Inline initial conditions will be ignored.

    Args:
        netlist (Netlist): a Netlist instance.
        timestep (float): Interval between calculations during simulation.
        ic (dict): See 'ic' in Instance Attributes. Default None, in which case
            initial conditions are parsed from the netlist.

    Instance Attributes:
        netlist (Netlist): Same as netlist argument.
        circuit (ahkab.Circuit): An ahkab.Circuit instance derived from netlist.
            Returned by the preprocess() function.
        ic (dict): A dictionary containing initial node potentials and branch
            flows/currents. Keys/Values are of the form:
                v(<NODE_NAME>):POTENTIAL
                i(<ELEMENT_NAME>):CURRENT
            Populated from .ic directives in the netlist.
    """

    def __init__(self, netlist, timestep, ic=None):
        self.netlist = netlist
        self.circuit = self.preprocess(netlist)
        self.timestep = timestep
        self.ic = self.parse_ic(netlist) if ic is None else ic


    def preprocess(self, netlist):
        """
        preprocess() is called right after Simulator is instantiated. It performs
        relevant conversions on the netlist in case an external simulation
        library is being used.
        * Converts a Netlist instance into an ahkab.Circuit instance.
        * Extracts initial conditions for the first simulator run.

        Args:
            netlist (Netlist): a Netlist instance. Must have a node named '0'.
                Required by ahkab.Circuit.
        """
        temp_file = netlist.name + '.net.temp'
        with open(temp_file, 'w') as net:
            tnetlist = copy.deepcopy(netlist)
            tnetlist.flatten()
            net.write(tnetlist.definition)
        circuit, _, _ = ahkab.netlist_parser.parse_circuit(temp_file)
        os.remove(temp_file)
        return circuit


    def run(self, duration):
        """
        Runs a simulation for the specified time. Passes simulation results to
        postprocess().

        Args:
            duration (float): Time over which to run simulation.
        """
        # Setting initial conditions to either Operating Point or values
        # provided to the class.
        x0 = 'op' if len(self.ic) == 0 else ahkab.new_x0(self.circuit, self.ic)
        tran = ahkab.new_tran(tstart=0, tstop=duration, tstep=self.timestep,\
                              x0=x0, method=ahkab.transient.TRAP)
        res = ahkab.run(self.circuit, tran)['tran']
        return self.postprocess(res)


    def postprocess(self, result):
        """
        Runs any post-processing operations on the result of last simulation.
        Called by run().
        * Stores initial conditions from result for the next simulator run.
        """
        vals = result.items()
        latest = [(v[0].lower(), v[1][-1]) for v in vals if v[0] != 'T']
        res = {l[0][0] + '(' + l[0][1:] + ')': l[1] for l in latest}
        self.ic = res
        return res


    def update(self):
        """
        Synchronizes any changes made to self.netlist
        """
        for element in self.circuit:
            pass


    def parse_ic(self, netlist):
        """
        Parses initial conditions from the netlist instance.
        """
        icdict = {}
        if 'ic' in netlist.directives:
            for ic in netlist.directives['ic']:
                icdict.update(ic.kwargs)
        # Ahkab requires .ic directives to have a name=NAME pair, irrelevant here
        if 'name' in icdict:
            del icdict['name']
        return icdict
