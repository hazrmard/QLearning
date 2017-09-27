"""
This module defines the Simulator class which conducts transient analysis
of a circuit. The simulator uses Ahkab library for calculations.

All simulators expose the following interface:

* Instantiation with the environment 'env' and class-specific parameters.
* run(state, action, **kwargs): Which simulates the environment for a given
    'state' after taking some 'action' and returns the new environment variables.
"""

import os
os.environ['LANG'] = 'en_US.UTF-8'
import ahkab
import numpy as np

# Fixed time-step too small error. Make larger if errors persist.
ahkab.options.transient_max_nr_iter = 1000


class Simulator:
    """
    The Simulator calculates behaviour of a circuit described by a netlist over
    time. It remembers its state i.e. after every run() the latest
    potentials/currents are stored as the next run()'s initial values.
    Simulator.run() accepts state/action vectors to modify the environment and
    returns a result dict, or the next state vector.

    Note: The netlist should not have inline initial conditions in element
    definitions. Instead they should be provided in special .ic directives or as
    a dict to this class. Inline initial conditions will be ignored.

    Note: For actual simulation, a third party library (ahkab) is used. So the
    Netlist representation stored in self.netlist is converted to the ahkab
    representation in self.circuit.

    Args:
        env (Netlist): a Netlist instance defining the environment.
        timestep (float): Max interval between calculations during simulation.
            Lower values are performance intensive.
        stepsize (float): The default time to run the simulation. Used when
            stepsize is not provided to run(). If not specified, defaults to
            timestep.
        state_mux (func): A function that gets a vector of state variables and
            modifies the netlist accordingly. Returns the modified netlist.
            Signature:
                modified Netlist = state_mux(state_vector, action_vector, Netlist)
        state_demux (func): A function that gets simulation results and converts
            them into a state variable vector to be returned. Signature:
                state_vector = state_demux(prev. state, prev. action, Netlist, result)
            Where 'result' is a dict of the form 'ic' (see ic in Attributes),
            'Netlist' is an instance of that class, and 'prev. state' is
            a vector describing the starting state.
        ic (dict): See 'ic' in Instance Attributes. Default None, in which case
            initial conditions are parsed from the netlist/ guessed using
            operating point calculations.

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
                 stepsize=-1, *args, **kwargs):
        self.netlist = env
        self.circuit = self.preprocess(env)
        self.timestep = timestep
        self.stepsize = timestep if stepsize <= 0 else stepsize
        self._state_mux = state_mux
        self._state_demux = state_demux if state_demux is not None else\
                            lambda w, x, y, z: z
        self.ic = self._parse_ic() if ic is None else ic

    @property
    def env(self):
        return self.netlist


    def preprocess(self, netlist):
        """
        preprocess() is called right after Simulator is instantiated. It performs
        relevant conversions on the netlist in case an external simulation
        library is being used (in this case, Ahkab).
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
        return circuit                 # apply any element changes


    def run(self, state=None, action=None, stepsize=-1, **kwargs):
        """
        Runs a simulation for the specified time. Passes simulation results to
        postprocess().

        Args:
            state (list/tuple/ndarray): A vector of state variables that are used
                to change self.netlist and self.circuit
            action (list/tuple/ndarray): A vector of action variables that are used
                to change self.netlist and self.circuit
            stepsize (float): Time over which to run simulation. Defaults to
                self.timestep.
        Returns:
            A vector of state variables describing the new state.
        """
        stepsize = self.stepsize if stepsize <= 0 else stepsize
        if state is not None or action is not None:
            self.set_state(state, action)
        # Setting initial conditions to either Operating Point or values
        # provided to the class.
        x0 = 'op' if len(self.ic) == 0 else ahkab.new_x0(self.circuit, self.ic)
        tran = ahkab.new_tran(tstart=0, tstop=stepsize, tstep=self.timestep,\
                              x0=x0, method=ahkab.transient.TRAP)
        res = ahkab.run(self.circuit, tran)['tran']
        return self.postprocess(state, action, res)


    def postprocess(self, prev_state, prev_action, result):
        """
        Runs any post-processing operations on the result of last simulation.
        Called by run(). Passes formatted results to state_demux to be converted
        to a state vector.
        * Stores initial conditions from result for the next simulator run.

        Args:
            prev_state (list/tuple/ndaray): The state vector of the last state.
            prev_action (list/tuple/ndaray): The action taken from prev_state.
            result (dict): The result of the ahkab simulation.

        Returns:
            The state vector of the new state after simulation.
        """
        # converting result dict keys in proper format
        vals = result.items()
        latest = [(v[0].lower(), v[1][-1]) for v in vals if v[0] != 'T']
        res_ = {l[0][0] + '(' + l[0][1:] + ')': l[1] for l in latest}
        # simulation results may contain voltages/currents for internal nodes
        # not specified by the ic. They are removed so provided and returned
        # initial conditions / results contain the same keys.
        res = {}
        for key, value in res_.items():
            # if key is either voltage on a node or current through element
            if key[2:-1] in self.netlist.graph or key[2:-1] in self.netlist.elements:
                res[key] = value
        self.ic = res
        return self._state_demux(prev_state, prev_action, self.netlist, res)


    def set_state(self, state, action):
        """
        Modifies the environment (netlist) according to the state/action variables.
        Changes in the modified netlist are applied to the ahkab.Circuit
        representation used by the external simulation library. Changes can be:

        * Additions/removals of circuit elements,
        * Modifications in element parameters,
        * NOT directives/models/block definitions. They should be included in the
          original netlist provided to Simulator.

        Args:
            state (list/tuple/ndarray): A list of state variables that are used
                to change self.netlist and self.circuit.
            action (list/tuple/ndarray): The action vector on the state.
        """
        self.netlist = self._state_mux(state, action, self.netlist)   # get modified netlist
        self.ic = self._parse_ic()              # get new initial conditions
        self._construct_nodes()                 # reconstruct nodes
        self._create_elements()                 # create new elements
        self._update_elements()                 # synchronize element parameters
        self._remove_elements()                 # remove redundant elements


    def _update_elements(self):
        """
        Synchronizes any structural changes made to self.netlist with ahkab.Circuit
        used by the third-party ahkab simulator:
        * Modifications in element parameters (including reference model),
        * Does NOT handle new models/block definitions/instances. All models/blocks
          to be used should be included from the beginning.
        """
        #TODO: Support block instances/definitions.
        node_dict = self.circuit.nodes_dict
        for element in self.circuit:
            # change params for elems that still exist
            try:
                ind = self.netlist.elements.index(element.part_id)
                elem = self.netlist.elements[ind]
                # transistor elements (ekv or mosq)
                if element.part_id[0] == 'm':
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.ng = node_dict[str(elem.nodes[1])]
                    element.n2 = node_dict[str(elem.nodes[2])]
                    element.nb = node_dict[str(elem.nodes[3])]
                    element.device.W = elem.param('w')
                    element.device.L = elem.param('l')
                    element.device.M = 1 if elem.param('m') is None else elem.param('m')
                    element.device.N = 1 if elem.param('n') is None else elem.param('n')
                    try:
                        element.ports = ((element.n1, element.nb), (
                            element.ng, element.n2), (element.n2, element.nb))
                        element.ekv_model = self.circuit.models[elem.value]
                        element.dc_guess = [element.ekv_model.VTO * (0.1) * element.ekv_model.NPMOS,
                                            element.ekv_model.VTO * (1.1) * element.ekv_model.NPMOS,
                                            0]
                    except AttributeError:
                        element.ports = ((element.n1, element.nb), (
                            element.ng, element.n2), (element.nb, element.n2))
                        element.mosq_model = self.circuit.models[elem.value]
                        element.dc_guess = [element.mosq_model.VTO*0.4*element.mosq_model.NPMOS,
                                            element.mosq_model.VTO*1.1*element.mosq_model.NPMOS,
                                            0]

                # diode element
                elif element.part_id[0] == 'd':
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.ports = ((element.n1, element.n2),)
                    element.model = self.circuit.models[elem.value]
                    element.off = elem.param('off') == 'true'
                    element.device.AREA = 1.0 if elem.param('area') is None else\
                        elem.param('area')
                    element.device.T = ahkab.constants.T if elem.param('t') is None\
                        else elem.param('t')

                # switch elements
                elif element.part_id[0] == 's':
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.sn1 = node_dict[str(elem.passive_nodes[0])]
                    element.sn2 = node_dict[str(elem.passive_nodes[1])]
                    element.model = self.circuit.models[elem.value]

                # independent current and voltage sources
                elif element.part_id[0] in ('v', 'i'):
                    dc = element.part_id[0] + 'dc'
                    ac = element.part_id[0] + 'ac'
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    stype = elem.param('type')
                    kwargs = {k:v for k, v in elem.kwargs.items() if k != 'type'}
                    # setting up preset time functions
                    if stype not in (dc, ac):
                        element.is_timedependent = True
                        if stype == 'sin':
                            element._time_function = ahkab.time_functions.sin(**kwargs)
                        elif stype == 'exp':
                            element._time_function = ahkab.time_functions.exp(**kwargs)
                        elif stype == 'sffm':
                            element._time_function = ahkab.time_functions.sffm(**kwargs)
                        elif stype == 'am':
                            element._time_function = ahkab.time_functions.am(**kwargs)
                        elif stype == 'pwl':
                            element._time_function = ahkab.time_functions.pwl(**kwargs)
                        elif stype == 'pulse':
                            element._time_function = ahkab.time_functions.pulse(**kwargs)
                    # setting up custom time function
                    elif stype is None:
                        element.is_timedependent = True
                        element._time_function = elem.function
                    else:   # i.e. stype is [i|v]ac/dc
                        element.is_timedependent = False
                    # setting up time invariant properties
                    element.abs_ac = np.abs(elem.param(ac)) if elem.param(ac) else None
                    element.arg_ac = np.angle(elem.param(ac)) if elem.param(ac) else None
                    element.dc_value = elem.param(dc)
                    if element.part_id[0] == 'v' and element.dc_value is not None:
                        element.dc_guess = [element.dc_value]

                # voltage controlled sources
                elif element.part_id[0] in ('e', 'g'):
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.sn1 = node_dict[str(elem.passive_nodes[0])]
                    element.sn2 = node_dict[str(elem.passive_nodes[1])]
                    element.alpha = elem.value

                # current controlled sources
                elif element.part_id[0] in ('f', 'h'):
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.alpha = elem.value[1]
                    element.source_id = elem.value[0]

                # common case for elements w/ only 2 nodes and 1 value
                # R, C, L
                else:
                    element.n1 = node_dict[str(elem.nodes[0])]
                    element.n2 = node_dict[str(elem.nodes[1])]
                    element.value = float(elem.value)
            # if element does not exist anymore
            except ValueError:
                # non-existent elements removed by self._remove_elements()
                pass


    def _create_elements(self):
        """
        Identifies new elements in self.netlist but not yet in self.circuit
        and creates them. Elements are created with basic properties. All
        attributes are checked/assigned in the _update_elements() function
        called after _create_elements().
        """
        part_ids = [e.part_id for e in self.circuit]
        new_elems = [e for e in self.netlist.elements if e.name not in part_ids]
        for elem in new_elems:
            # transistor elements (ekv or mosq)
            if elem.name[0] == 'm':
                self.circuit.add_mos(elem.name, *map(str, elem.nodes),
                                     w=elem.param('w'), l=elem.param('l'),
                                     model_label=elem.value,
                                     m=1 if elem.param('m') is None else elem.param('m'),
                                     n=1 if elem.param('n') is None else elem.param('n'))
       
            # diode element
            elif elem.name[0] == 'd':
                self.circuit.add_diode(elem.name, *map(str, elem.nodes),
                                       model_label=elem.value,
                                       Area=elem.param('area'),
                                       T=elem.param('t'),
                                       off=(elem.param('off') is True))
           
            # switch elements
            elif elem.name[0] == 's':
                self.circuit.add_switch(elem.name, *map(str, elem.nodes),
                                        *map(str, elem.passive_nodes),
                                        model_label=elem.value)

            # independent voltage/current source
            elif elem.name[0] in ('v', 'i'):
                if elem.name[0] == 'v':
                    func = self.circuit.add_vsource
                else:
                    func = self.circuit.add_isource
                # created w/ basic properties. All attributes assigned later.
                func(elem.name, *map(str, elem.nodes), 0, 0, None)

            # voltage controlled sources
            elif elem.name[0] in ('e', 'g'):
                if elem.name[0] == 'e':
                    func = self.circuit.add_vcvs
                else:
                    func = self.circuit.add_vccs
                func(elem.name, *map(str, elem.nodes), *map(str, elem.passive_nodes),
                     elem.value)

            # current controlled sources
            elif elem.name[0] in ('f', 'h'):
                if elem.name[0] == 'f':
                    func = self.circuit.add_cccs
                else:
                    func = self.circuit.add_ccvs
                func(elem.name, *map(str, elem.nodes), elem.value[0], elem.value[1])

            # resistors
            elif elem.name[0] == 'r':
                self.circuit.add_resistor(elem.name, *map(str, elem.nodes),
                                          elem.value)

            # capacitors
            elif elem.name[0] == 'c':
                self.circuit.add_capacitor(elem.name, *map(str, elem.nodes),
                                           elem.value)

            # inductors
            elif elem.name[0] == 'l':
                self.circuit.add_inductor(elem.name, *map(str, elem.nodes),
                                          elem.value)


    def _remove_elements(self):
        """
        Identifies elements that are still in self.circuit (ahkab.Circuit) but
        not in self.netlist (Netlist). Then removes them from self.circuit.
        """
        element_names = [e.name for e in self.netlist.elements]
        old_elems_ind = [i for i, e in enumerate(self.circuit)\
                         if e.part_id not in element_names]
        for index in old_elems_ind[::-1]:
            self.circuit.pop(index)


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
