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
        self.name = name


    def __str__(self):
        return self.name
