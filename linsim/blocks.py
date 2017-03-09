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

    Instance Attributes:
        name (str): Name of block.
        definition (list): List of element definitions strings in block.
            Can contain nested block definitions.
        nodes (list): List of nodes (str/Node).
        block_defs (dict): block name: definition dict of nested blocks.
        graph (dict): node (Node): element list dictionary of elements/blocks in
            block.

    Class Attributes:
        block_regex (str): Regex pattern that matches a block and captures
            the block definitions inside.
        begin (str): A regex pattern that defines start of block.
        end (str): A regex pattern that defines end of block.
        prefix (str): Element name prefix that defines block instance.
    """

    block_regex = ''
    begin = '.subckt'
    end = '.ends'
    prefix = 'x'

    def __init__(self, name, nodes, definition=()):
        self.name = name
        self.definition = list(definition)
        self.nodes = nodes
        self.block_defs = {}
        self.graph = {}
        if len(self.definition):
            self._parse()


    def _parse(self):
        """
        Parses self.definition to populate adjacency lists for each node.
        """
        self._parse_blocks()
        self._parse_elements()


    def _parse_blocks(self):
        """
        Finds nested block definitions in current block and populates block_defs.
        """
        pass


    def _parse_elements(self):
        """
        Instantiates elements/blocks in current scope. Populates self.graph.
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