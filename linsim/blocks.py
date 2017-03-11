"""
This module defines the Block class which describes a sub-circuit in a netlist.
"""

import re
import elements
from nodes import Node


class Block:
    """
    Block defines a subcircuit in a netlist comprised of multiple elements. The
    block is opaque from the outside, only exposing its external nodes. Of the
    form:
        .subckt <NAME> <NODE1>,..., [<PARAM1=VALUE1>,...]
        ELEMENT DEFINITIONS...
        .ends <NAME>

    Args:
        name (str): the name of the block.
        nodes (list): A list of nodes (str/Node) that are external ports.
        definition (list/tuple): List of element definitions strings in block.
            Can contain nested block definitions.
        mux (ElementMux): A multiplexer that instantiates Element subclasses
            based on element definitions. Defaults to elements.DEFAULT_MUX.

    Instance Attributes:
        name (str): Name of block.
        definition (str)): Newline separated element definitions in block.
            Can contain nested block definitions.
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
    block_regex = r'\.subckt\s+(?P<name>\w+)(?P<args>.*)' + \
                  r'\n(?P<defs>[\s\S]+?)\n' + \
                  r'\.ends\s+(?P=name)(?:$|\s)'
    node_regex = elements.Element.value_regex
    pair_regex = elements.Element.pair_regex

    def __init__(self, name, nodes, definition=(), mux=elements.DEFAULT_MUX, **kwargs):
        self.name = name
        self.mux = mux
        self.definition = '\n'.join(definition).lower()
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.elements = []
        self.blocks = {}
        self.graph = {}
        if len(self.definition):
            self._parse()


    def __str__(self):
        result = ''
        for block_name, block in self.blocks.items():
            result += block.definition + '\n'
        for elem in self.elements:
            result += str(elem) + '\n'
        return result


    def _parse(self):
        """
        Parses self.definition to populate adjacency lists for each node.
        """
        self._sanitize()                                # clean whitespace etc.
        no_block = self._parse_blocks(self.definition)  # extract nested blocks
        self._parse_elements(no_block)                  # extract elements


    def _sanitize(self):
        """
        Sanitizes self.definition so it can be parsed properly.
        """
        # # removes empty lines
        # self.definition = re.sub(r'\s*\n\s*', r'\n', self.definition)
        # # removes spaces around = signs
        # self.definition = re.sub(r'\s*=\s*', '=', self.definition)
        # removes spaces after commas, colons, dashes etc.
        self.definition = self.definition.strip().lower()
        self.definition = re.sub(r'\s*(?P<sep>[,;-_=\n])\s*', r'\g<sep>', self.definition)


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
                    block_name = elem_def[elem_def.rfind(' ')+1:]
                    try:
                        block = self.blocks[block_name]
                    except KeyError:
                        raise KeyError('Block:' + block_name + ' is not defined.')
                    elem = elements.BlockInstance(block, definition=elem_def,\
                        num_nodes=block.num_nodes)
                else:
                    elem = self.mux.mux(elem_def)   # instantiate from mux
                self.add(elem)


    def add(self, elem):
        """
        Adds element to block.

        Args:
            elem (Element): An Element instance (or subclass).
        """
        if isinstance(elem, elements.Element):
            if elem in self.elements:           # check if duplicate
                raise ValueError('Duplicate element already exists.')
            else:
                self.elements.append(elem)      # add elem to elements list
                for node in set(elem.nodes):    # add elem to adjacency list
                    if self.graph.get(node):
                        self.graph[node].append(elem)
                    else:
                        self.graph[node] = [elem]
        else:
            raise TypeError('elem must be an instance of Element.')



    def remove(self, elem):
        """
        Removes element from block. May leave hanging nodes.

        Args:
            elem (Element/str): Element instance / name to be removed.
        """
        if elem in self.elements:
            self.elements.remove(elem)
            for node in set(elem.nodes):
                if self.graph.get(node):
                    self.graph[node].remove(elem)
                    if len(self.graph[node]) == 0:
                        del self.graph[node]



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
            raise ValueError('Node1 does not exist in block: ' + self.name)


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
    def is_directive(elem):
        """
        Checks whether a line is a directive.

        Args:
            elem (str/Element): A line in the netlist or an Element instance.

        Returns:
            boolean -> True if line is directive.
        """
        return str(elem)[0] == '.'


    def is_block_instance(self, elem):
        """
        Checks whether a line defines a block instance.

        Args:
            elem (str/Element): A line in the netlist or an Element instance.

        Returns:
            boolean -> True if line is a block instance declaration.
        """
        return str(elem)[:len(self.__class__.prefix)] == self.__class__.prefix
