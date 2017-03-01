"""
This module defines the Block class which describes a sub-circuit in a netlist.
"""

from elements import Element

class Block:

    begin = '.subckt'
    end = '.end'

    def __init__(self, defs):
        self.nodes = {}
        self.blocks = {}