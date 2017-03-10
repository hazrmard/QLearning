"""
This module defines the Block class which describes a sub-circuit in a netlist.
"""

import re
import elements
from nodes import Node


class Block:
    """
    Block defines a subcircuit in a netlist comprised of multiple elements. The
    block is opaque from the outside, only exposing its external nodes.

    Args:
        name (str): the name of the block.
        nodes (list): A list of nodes (str/Node) that are external ports.
        definition (list/tuple): List of element definitions strings in block.
            Can contain nested block definitions.
        mux (ElementMux): A multiplexer that instantiates Element subclasses
            based on element definitions. Defaults to elements.DEFAULT_MUX.

    Instance Attributes:
        name (str): Name of block.
        definition (list): List of element definitions strings in block.
            Can contain nested block definitions.
        nodes (list): List of nodes (str/Node).
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

    prefix = 'x'
    block_regex = r'\.subckt\s+(?P<name>\w+)(?P<args>.*)' + \
                  r'\n(?P<defs>[\s\S]+?)\n' + \
                  r'\.ends\s+(?P=name)(?:$|\s)'
    node_regex = elements.Element.value_regex
    pair_regex = elements.Element.pair_regex

    def __init__(self, name, nodes, definition=(), mux=elements.DEFAULT_MUX, **kwargs):
        self.name = name
        self.mux = mux
        self.definition = '\n'.join(definition)
        self.nodes = nodes
        self.blocks = {}
        self.graph = {}
        if len(self.definition):
            self._parse()


    def _parse(self):
        """
        Parses self.definition to populate adjacency lists for each node.
        """
        self._sanitize()
        trimmed_def = self._parse_blocks(self.definition)
        self._parse_elements(trimmed_def)


    def _sanitize(self):
        """
        Sanitizes self.definition so it can be parsed properly.
        """
        # removes empty lines
        self.definition = re.sub(r'\n\s+', r'\n', self.definition)
        # removes spaces around = signs
        self.definition = re.sub(r'\s*=\s*', '=', self.definition)
        # removes spaces after commas, colons, dashes etc.
        self.definition = re.sub('(?P<sep>[,;-_])\\s+', '\\g<sep>', self.definition)


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
        pass


    def add(self, elem):
        """
        Adds element to block.

        Args:
            elem (Element): An Element instance (or subclass).
        """
        pass


    def remove(self, elem):
        """
        Removes element from block. May leave hanging nodes.

        Args:
            elem (Element/str): Element instance / name to be removed.
        """
        pass


    def is_element(self, line):
        """
        Checks whether a line defines a component.

        Args:
            line (str): A line in the netlist.

        Returns:
            boolean -> True if line is component.
        """
        return line[0] != '*' and line[0] != '.'


    def is_directive(self, line):
        """
        Checks whether a line is a directive.

        Args:
            line (str): A line in the netlist.

        Returns:
            boolean -> True if line is directive.
        """
        return line[0] == '.'