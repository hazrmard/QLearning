"""
This module defines the Block class which describes a sub-circuit in a netlist.
"""

import re
import copy
try:
    import elements
    from nodes import Node
except ImportError:
    from . import elements
    from .nodes import Node


class Block:
    """
    Block defines a subcircuit in a netlist comprised of multiple elements. The
    block is opaque from the outside, only exposing its external nodes. Of the
    form:
        .subckt <NAME> <NODE1>,..., [<PARAM1=VALUE1>,...]
        ELEMENT DEFINITIONS...
        .ends <NAME>
    A block is comprised of elements and nested block definitions (Block).
    Instances of block definitions are treated as elements and represented by
    BlockInstance class which is a subclass of Element.
    Blocks (except for the top level Netlist) do not contain directives.

    Args:
        name (str): the name of the block.
        nodes (list): A list of nodes (str/Node) that are external ports.
        definition (list/tuple): List of element definitions strings in block.
            Can contain nested block definitions.
        mux (ElementMux): A multiplexer that instantiates Element subclasses
            based on element definitions. Defaults to elements.DEFAULT_MUX.

    Instance Attributes:
        name (str): Name of block.
        definition (str): Newline separated element definitions in block.
            Can contain nested block definitions. Read only.
        nodes (list): List of nodes (str/Node).
        num_nodes (int): Number of nodes exposed i.e. size of nodes.
        elements (list): List of element instances (Element).
        blocks (dict): block name: Block() dict of nested blocks.
        graph (dict): node (Node): element list dictionary of elements/blocks in
            block.

    Class Attributes:
        block_regex (str): Regex pattern to capture block information. The
            pattern must capture named groups: name, args, and defs.
        name_regex (str): Regex pattern to capture block name.
        node_regex (str): Regex pattern to capture block nodes.
        pair_regex (str): Regex pattern to capture block key=value pair params.
        begin (str): A regex pattern that defines start of block.
        end (str): A regex pattern that defines end of block.
        prefix (str): Element name prefix that defines block instance.
    """

    prefix = elements.BlockInstance.prefix
    begin = '.subckt'
    end = '.ends'
    block_regex = r'\.subckt\s+(?P<name>\w+)(?P<args>.*)' + \
                  r'\n(?P<defs>[\s\S]+?)\n' + \
                  r'\.ends\s+(?P=name)(?:$|\s)'
    node_regex = elements.Element.value_regex
    pair_regex = elements.Element.pair_regex

    def __init__(self, name, nodes, definition=(), mux=elements.DEFAULT_MUX, **kwargs):
        self.name = name.lower()
        self.mux = mux
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.elements = []
        self.blocks = {}
        self.graph = {}
        definition = '\n'.join(definition).lower()
        if len(definition):
            self._parse(definition, **kwargs)


    @property
    def definition(self):
        """
        Returns just the internal definition of the block.
        """
        return self.__str__(enclose=False)


    def __str__(self, enclose=True):
        """
        Renders the netlist of block.

        Args:
            enclose (bool): If True, wraps definition in the block begin/end
                statements. Default=True.

        Returns:
            A string representing the netlist.
        """
        result = ''
        if enclose:
            result += self.__class__.begin + ' ' + self.name + ' ' \
                    + ' '.join([str(n) for n in self.nodes]) + '\n'
        for _, block in self.blocks.items():
            result += str(block) + '\n'
        for elem in self.elements:
            result += str(elem) + '\n'
        if enclose:
            result += self.__class__.end + ' ' + self.name
            return result
        else:
            return result[:-1]  # leave off the trailing \n


    def __repr__(self):
        return self.name


    def __hash__(self):
        return hash(self.name)


    def __eq__(self, other):
        """
        Equality check by block name. Does NOT compare definition.
        """
        if isinstance(other, Block):
            return other.name == self.name
        else:
            return other.__eq__(self.name)


    def flatten(self):
        """
        Flattens the block by converting all nested blocks into top-level
        elements. Elements and their nodes are named as:
            <INSTANCE_NAME>_<NAME_IN_BLOCK>
        Where <INSTANCE_NAME> does not dontain the prefix denoting block instance.
        All block definitions/declarations are removed.
        """
        instances = [i for i in self.elements if i.prefix == self.__class__.prefix]
        for _, block in self.blocks.items():
            block.flatten()
        for instance in instances:
            # Create a name prefix to identify all flattened elements of a
            # block instance.
            # Note: name concatenation is w/o _ since ahkab does not accept _
            # as valid element names.
            name = instance.block.name + instance.name
            # a dict of nodes in prototype/block interface : instance interface
            node_map = {a:b for a, b in zip(instance.block.nodes, instance.nodes)}
            for elem in instance.block.elements:
                # make a copy of all elements in prototype to replace instance
                elemc = copy.deepcopy(elem)
                # detect prefix, or assume one from element name
                prefix = elemc.__class__.prefix if len(elemc.__class__.prefix)\
                         else elemc.name[:1]
                elemc.name = prefix + name + elemc.name
                for node in elemc.nodes:
                    # if element node connects to prototype interface, rename it
                    # to match the node connecting to the instance's interface
                    # i.e. the external node the block/prototype is connected to
                    if node in node_map:
                        node.name = str(node_map[node])
                    else:
                    # if node is internal to the prototype/block, then rename
                    # it by prepending the instance name to it.
                        node.name = name + str(node)
                self.add(elemc)
            self.remove(instance)
        self.blocks = {}


    def _parse(self, definition, sanitize=True):
        """
        Parses self.definition to populate adjacency lists for each node.

        Args:
            definition (str): Definition string.
            sanitize (bool): True to sanitize netlist. False assumes netlist
                text complies with sanitation rules.
        """
        if sanitize:
            definition = self._sanitize(definition)      # clean whitespace etc.
        no_block = self._parse_blocks(definition)        # extract nested blocks
        self._parse_elements(no_block)                   # extract elements


    def _sanitize(self, definition):
        """
        Sanitizes self.definition so it can be parsed properly.

        Args:
            definition (str): The definition string.

        Returns:
            Sanitized definition string.
        """
        # # removes empty lines
        # self.definition = re.sub(r'\s*\n\s*', r'\n', self.definition)
        # # removes spaces around = signs
        # self.definition = re.sub(r'\s*=\s*', '=', self.definition)
        # removes spaces after commas, colons, dashes etc.
        definition = definition.strip().lower()
        definition = re.sub(r'\s*(?P<sep>[,;-_=\n])\s*', r'\g<sep>', definition)
        return definition


    def _parse_blocks(self, definition):
        """
        Finds nested block definitions in current block and populates block_defs.
        Uses class attribute block_regex to find block name, args, and definition.

        Args:
            definition (str): Block definition to parse.

        Returns:
            A string with all block definitions removed. Used by _parse_elements.
        """
        block_matches = re.finditer(self.__class__.block_regex, definition)

        for match in block_matches:
            name = match.group('name')
            args = match.group('args')
            defs = match.group('defs')

            nodes = re.findall(self.__class__.node_regex, args)
            nodes = [Node(n) for n in nodes]
            pairs = re.findall(self.__class__.pair_regex, args)
            pairs = {p[0].strip(): p[1].strip() for p in
                     [pair.split('=') for pair in pairs]}
            defs = defs.split('\n')
            self.blocks[name] = Block(name, nodes, defs, mux=self.mux, **pairs)
        return re.sub(self.__class__.block_regex, '', definition)


    def _parse_elements(self, definition):
        """
        Instantiates elements/blocks in current scope. Populates self.graph.

        Args:
            definition (str): Block definition with nested blocks defs removed.
        """
        definitions = definition.split('\n')
        for elem_def in definitions:
            if self.is_element(elem_def):
                if self.is_block_instance(elem_def):
                    block_name = elem_def[elem_def.find('name=')+5:]
                    try:
                        block = self.blocks[block_name]
                    except KeyError:
                        raise KeyError('Block:' + block_name + ' is not defined.')
                    elem = elements.BlockInstance(block, definition=elem_def,\
                        num_nodes=block.num_nodes)
                else:
                    elem = self.mux.mux(elem_def)   # instantiate from mux
                self.add(elem)


    def element(self, name):
        """
        Finds the Element instance corresponding to element name in block.

        Args:
            name (str): The name of the element for e.g. r100.

        Returns:
            An Element object representing that element in current block.
            If element not found returns None.
        """
        try:
            return self.elements[self.elements.index(name.lower())]
        except ValueError:
            return None


    def elements_like(self, prefix):
        """
        Finds all Element instances with names starting with the provided
        prefix string.

        Args:
            name (str): The prefix to match element name with.

        Returns:
            A list of Element instances.
        """
        return [e for e in self.elements if e.name.startswith(prefix.lower())]


    def instance(self, instance_name, **nodes):
        """
        Creates an instance of current block as a BlockInstance object.

        Args:
            instance_name (str): Name of block. Should have an instance prefix
                e.g xBlockName.
            **nodes: Any key=value parameters of the form:
                    <NODE_IN_BLOCK>=<NODE_IN_CIRCUIT>, ...
                Which describes what nodes in the block definition will go to
                which nodes in the circuit itself.

        Returns:
            A BlockInstance object.
        """
        return elements.BlockInstance(self, instance_name, name=self.name, **nodes)


    def add(self, elem):
        """
        Adds an Element or a BlockInstance to current block.
        Elements (and BlockInstances) get appended to self.elements and the
        adjacency list in self.graph.

        Args:
            elem (Element): An Element or BlockInstance (or subclass).
        """
        if isinstance(elem, elements.Element):
            if elem in self.elements:           # check if duplicate
                raise ValueError('Duplicate element: ' + elem.name + ' already exists.')
            else:
                if self.is_block_instance(elem):
                    if elem.block.name not in self.blocks:
                        raise ValueError('Block instance of ' + elem.block.name\
                                        + ' is not defined.')
                self.elements.append(elem)      # add elem to elements list
                for node in set(elem.nodes):    # add elem to adjacency list
                    if self.graph.get(node):
                        self.graph[node].append(elem)
                    else:
                        self.graph[node] = [elem]
        else:
            raise TypeError('add() only accepts instances of Element.')


    def add_block(self, block):
        """
        Adds a block definition to current block.

        Args:
            block (Block): A Block object to be nested in the current block.
                Note that this will just add the definition and not an instance
                of that block definition to the circuit.
        """
        if isinstance(block, Block):
            if block in self.blocks:
                raise ValueError('Duplicate block:' + block.name + 'already exists.')
            else:
                self.blocks[block] = block


    def remove(self, elem):
        """
        Removes element from block. May leave hanging nodes.

        Args:
            elem (Element/str): Element instance / name to be removed.
        """
        if isinstance(elem, str):   # convert string id to Element instance
            elem = self.element(elem)
        if elem in self.elements:
            self.elements.remove(elem)    
            for node in set(elem.nodes):
                if self.graph.get(node):
                    self.graph[node].remove(elem)
                    if len(self.graph[node]) == 0:
                        del self.graph[node]


    def remove_block(self, block):
        """
        Removes block definition from current block. All instances of that
        definition are also removed.

        Args:
            block (str/Block): Block object or name of block to remove.
        """
        if block in self.blocks:
            del self.blocks[block]
            for elem in self.elements:
                if isinstance(elem, elements.BlockInstance):
                    if elem.block == block:
                        self.remove(elem)


    def short(self, node1, node2):
        """
        Short-circuit node1 and node2, removing any elements between them that
        exclusively connect to these two nodes.

        Args:
            node1 (str/Node): The node instance/ name to short. This node must
                exist in the block.
            node2 (str/Node): The node instance/ name to short with. This can
                be a new node. All occurances of node1 will be replaced with
                node2.
        """
        node2 = Node(node2) if isinstance(node2, str) else node2
        if node1 in self.graph:
            elements = self.graph[node1]
            del self.graph[node1]
            redundant = []
            for elem in elements:
                # reassign nodes on elements in shorted node
                elem.nodes[elem.nodes.index(node1)] = node2
                # if all nodes are identical, mark element as redundant
                if elem.nodes.count(node2) == len(elem.nodes):
                    redundant.append(elem)
            # After reassignment, merge/create elements into node2's adj list
            if node2 in self.graph:
                self.graph[node2] = list(set(self.graph[node2]).union(
                    set(elements)))
            else:
                self.graph[node2] = elements
            # Remove redundant nodes from merging list and block's element list
            for elem in redundant:
                self.graph[node2].remove(elem)
                self.elements.remove(elem)
        else:
            raise ValueError('Node: ' + str(node1) + ' does not exist in block: '\
                             + self.name)


    @staticmethod
    def is_element(elem):
        """
        Checks whether a line defines a component.

        Args:
            elem (str/Element): A line in the netlist or an Element instance.

        Returns:
            boolean -> True if line is component.
        """
        return str(elem)[0] != '*' and str(elem)[0] != '.'


    @staticmethod
    def is_comment(line):
        """
        Checks whether a line in the netlist is a comment.

        Args:
            line (str/Element): A netlist string or an Element instance.

        Returns:
            True of the line is a comment. False if it is an Element/element
            definition.
        """
        return str(line)[0] == '*'


    def is_block_instance(self, elem):
        """
        Checks whether a line defines a block instance.

        Args:
            elem (str/Element): A line in the netlist or an Element instance.

        Returns:
            boolean -> True if line is a block instance declaration.
        """
        if isinstance(elem, (str, bytes)):
            return str(elem)[:len(self.__class__.prefix)] == self.__class__.prefix
        elif isinstance(elem, elements.BlockInstance):
            return True
        return False
