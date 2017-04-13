"""
This module defines the Simulator class which conducts transient analysis
of a circuit. The simulator uses Ahkab library for calculations.

All simulators expose the following interface:

* Instantiation with the environment 'env'.
* set_state(variables): A list of state variable values that modifies the
    environment.
* run(duration): Which simulates the environment for 'duration' and returns
    the new environment variables.
"""

import os
os.environ['LANG'] = 'en_US.UTF-8'
import ahkab

# Fixed time-step too small error. Make larger if errors persist.
ahkab.options.transient_max_nr_iter = 100


class Simulator:
    """
    The Simulator calculates behaviour of a circuit described by a netlist over
    time. It remembers its state i.e. after every run() the latest
    potentials/currents are stored as the next run's initial values.

    Note: The netlist should not have inline initial conditions in element
    definitions. Instead they should be provided in special .ic directives or as
    a dict to this class. Inline initial conditions will be ignored.

    Args:
        env (Netlist): a Netlist instance defining the environment.
        timestep (float): Interval between calculations during simulation.
        state_mux (func): A function that gets a vector of state variables and
            modifies the netlist accordingly. Returns the modified netlist.
            Signature:
                modified Netlist = state_mux(Netlist, state_vector)
        state_demux (func): A function that gets simulation results and converts
            them into a state variable vector to be returned. Signature:
                state_vector = state_demux(Netlist, result)
            Where 'result' is a dict of the form 'ic' (see ic in Attributes).
        ic (dict): See 'ic' in Instance Attributes. Default None, in which case
            initial conditions are parsed from the netlist.

    Instance Attributes:
        netlist (Netlist): Same as netlist argument.
        env (Netlist): A common property that returns the environment (in this
            case the netlist instance).
        circuit (ahkab.Circuit): An ahkab.Circuit instance derived from netlist.
            Returned by the preprocess() function.
        ic (dict): A dictionary containing initial node potentials and branch
            flows/currents. Keys/Values are of the form:
                v(<NODE_NAME>):POTENTIAL
                i(<ELEMENT_NAME>):CURRENT
            Populated from .ic directives in the netlist.
    """

    def __init__(self, env, timestep, state_mux, state_demux=None, ic=None,
                 *args, **kwargs):
        self.netlist = env
        self.circuit = self.preprocess(env)
        self.timestep = timestep
        self._state_mux = state_mux
        self._state_demux = state_demux if state_demux is not None else\
                            lambda x, y: y
        self.ic = self._parse_ic() if ic is None else ic

    @property
    def env(self):
        return self.netlist


    def preprocess(self, netlist):
        """
        preprocess() is called right after Simulator is instantiated. It performs
        relevant conversions on the netlist in case an external simulation
        library is being used (in this Ahkab).
        * Converts a Netlist instance into an ahkab.Circuit instance.
        * Extracts initial conditions for the first simulator run.

        Args:
            netlist (Netlist): a Netlist instance. Must have a node named '0'.
                Required by ahkab.Circuit.
        """
        temp_file = netlist.name + '.net.temp'
        with open(temp_file, 'w') as net:
            net.write(netlist.definition)
        circuit, _, _ = ahkab.netlist_parser.parse_circuit(temp_file)
        os.remove(temp_file)
        return circuit


    def set_state(self, state):
        """
        Modifies the environment (netlist) according to the state variables.
        Changes in the modified netlist are applied to the ahkab.Circuit
        representation used by the external simulation library.

        Args:
            state (list/tuple/ndarray): A list of state variables that are used
                to change self.netlist and self.circuit.
        """
        self.netlist = self._state_mux(self.netlist, state)   # get modified netlist
        self.ic = self._parse_ic()              # get new initial conditions
        self._construct_nodes()                 # reconstruct nodes
        self._update_elements()                 # apply any element changes


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
        res_ = {l[0][0] + '(' + l[0][1:] + ')': l[1] for l in latest}
        # simulation results may contain voltages/currents for internal nodes
        # not specified by the ic. They are removed so provided and returned
        # initial conditions / results contain the same keys.
        res = {}
        for key, value in res_.items():
            if key in self.ic:
                res[key] = value
        self.ic = res
        return self._state_demux(self.netlist, res)


    def _update_elements(self):
        """
        Synchronizes any structural changes made to self.netlist:
        * Additions/removals of elements,
        * Modifications in element parameters (including reference model),
        * Does NOT handle new models/block definitions. All models/blocks to be
          used should be included from the beginning.
        """
        node_dict = self.circuit.nodes_dict
        for element in self.circuit:
            # change params for elems that still exist
            try:
                ind = self.netlist.elements.index(element.part_id)
                elem = self.netlist.elements[ind]
                # transistor node assignments are different from other elems
                if element.part_id[0] == 'm':
                    pass
                # switch elements
                elif element.part_id[0] == 's':
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.sn1 = node_dict[str(elem.passive_nodes[0])]
                    element.sn2 = node_dict[str(elem.passive_nodes[1])]
                    element.model = self.circuit.models[elem.value]
                # common case for elements w/ only 2 nodes and 1 value
                # R, C, L
                else:
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.value = float(elem.value)
            # if element does not exist anymore
            except ValueError:
                # self.circuit.remove(element)
                pass


    def _parse_ic(self):
        """
        Parses initial conditions from the netlist instance.

        Returns:
            A dict of the form {v(NODE):VOLTAGE, i(ELEMENT):CURRENT...}
        """
        icdict = {}
        if 'ic' in self.netlist.directives:
            for ic in self.netlist.directives['ic']:
                icdict.update(ic.kwargs)
        # Ahkab requires .ic directives to have a name=NAME pair, irrelevant here
        if 'name' in icdict:
            del icdict['name']
        return icdict


    def _construct_nodes(self):
        """
        Constructs the node map in ahkab.Circuit.nodes_dict using netlist.graph.
        """
        nodes = {0:'0', '0':0}
        for i, node in enumerate([n for n in self.netlist.graph if n != '0']):
            nodes.update({i+1:str(node), str(node):i+1})
        self.circuit.nodes_dict = nodes
