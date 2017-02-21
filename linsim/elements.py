"""
This module defines the Element class and its derivatives. They are the atomic
parts of a netlist/circuit.
"""

import re
from nodes import Node


class Element:
    """
    The Element class represents a single component in a netlist.
    All arguments are case insensitive.

    Args:
        definition (str): A netlist definition of the element. Of the form:
            <PREFIX><NAME> <NODE1>... <VALUE1>... [<PARAM1>=<VALUE1>...]

    Class Attributes:
        num_nodes (int): Number of nodes element is connected to.
        prefix (str): Prefix for element definition (e.g. 'R' for resistors).
        name (str): Element type (e.g. Resistor, Capacitor)
        value_regex (str): Regular expression pattern to match all single
            values in: VALUE1 VALUE2 PARAM1=VALUE3 PARAM2=VALUE4
        param_regex: Regular expression pattern to match all PARAM=VALUE pairs.
    """
    num_nodes = 2
    prefix = ''
    name = 'Element'
    value_regex = r'(?:^|\s+)((?<!=)[\w_\.]+(?=(?:$|\s+)))'
    pair_regex = r'([\w]+=[^=]+?(?=(?:$|(?:\s+\S+=))))'

    def __init__(self, definition):
        self.definition = definition.lower()
        self._parse()


    def _parse(self):
        """
        Parses the definition and assigns attributes to instance accordingly.
        """
        split = self.definition.split()
        if len(split) < 1 + self.__class__.num_nodes + 1:
            # At least 1 name + num_nodes + 1 value
            raise AttributeError('Element definition does not have enough args.')
        if self.__class__.prefix.lower() != split[0][:len(self.__class__.prefix)]:
            raise ValueError('Incorrect element type.')

        self.name = split[0]
        self.nodes = [Node(x) for x in split[1:1+self.__class__.num_nodes]]

        # Construct the remaining value and param=value pair string
        value_str = ' '.join(split[1+self.__class__.num_nodes:])
        #   These subs are to comply with the regex pattern
        value_str = re.sub(' = ', '=', value_str)
        value_str = re.sub('(?P<sep>[,;-_])\\s+', '\\g<sep>', value_str)

        # Isolate single values
        values = re.findall(self.__class__.value_regex, value_str)
        if values:
            self.value = self._parse_values(values)

        # Isolate key=value pairs
        pairs = re.findall(self.__class__.pair_regex, value_str)
        if pairs:
            pairs_dict = {p[0].strip(): p[1].strip() for p in
                          [pair.split('=') for pair in pairs]}
            pairs_dict = self._parse_pairs(pairs_dict)
            for key, value in pairs_dict.items():
                setattr(self, key, value)


    def _parse_values(self, values):
        """
        Processes the single values string in definition. The returned value is
        assigned to self.value.
        Override this function to control what gets assigned to self.value.

        Args:
            values (list): A list of values from the definition string.

        Returns:
            The unchanged values list.
        """
        return values


    def _parse_pairs(self, pairs):
        """
        Processes the param=value pairs in definition. The returned dictionary
        is assigned as self.KEY = VALUE for all keys in the dictionary.
        Override this function to control what attributes get assigned to self.

        Args:
            pairs (dict): A dictionart of {PARAM: VALUE}

        Returns:
            The unchanged pairs dictionary.
        """
        return pairs
