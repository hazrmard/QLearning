"""
This module defines the Nodes class which defines the structure of a circuit.
"""



class Node:
    """
    Node represents a point of same potential/voltage in a netlist.

    Args:
        name (str): Name of node.
    """

    def __init__(self, name):
        self.name = str(name)


    def __str__(self):
        return self.name


    def __hash__(self):
        return hash(self.name)


    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == str(other.name)
        else:
            return other.__eq__(self.name)
