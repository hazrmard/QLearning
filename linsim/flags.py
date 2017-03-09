"""
This module defines the FlagGenerator class which converts state numbers into
flag combinations for use in the simulation.
"""

import math


class FlagGenerator:
    """
    FlagGenerator encodes a combination of flags describing the state of a
    system into a number which can be used as an index. Conversely, it can
    decode a number into a unique combination of flags describing a system.

    Args:
        flags: A a sequence of int arguments with the number of states possible
            for each flag. The states are normalized between [0, 1] inclusive.
            For example: 2, 4. The first flag, will have 2 possible states: 0, 1.
            The second flag will have 4 states: 0, 0.33, 0.67, 1. Each flag
            must have at least 2 states.

    Instance Attributes:
        flags (list): Stores flag sequence provided at instantiation.
        states (int): Total number of states possible with the given flags.
    """

    def __init__(self, *flags):
        self._state = -1    # internal count of current state during for loops
        self.flags = flags
        self.states = 1
        for flag in flags:
            self.states *= flag


    def __iter__(self):
        self._state = -1
        return self


    def __next__(self):
        if self._state < self.states - 1:
            self._state += 1
            return self.decode(self._state)
        else:
            raise StopIteration


    def decode(self, state):
        """
        Decodes a state number into a list of flags. The size of the list is
        the number of flags provided at instantiation.
        """
        pass


    @staticmethod
    def convert_basis(current, to, num):
        """
        Convert a number from one basis to another.

        Args:
            current (int): Current basis of number. Must be greater than 1
            to (int): Basis to convert number to. Must be greater than 1.
            num (int/list): The number to be converted. If basis=10, then num is
                decimal number, else a list of ints with least significant part
                at the end.

        Returns:
            A list of integers with the least significant number at the end.
        """
        if current != 10:
            number = 0
            power = 0
            for i in reversed(num):
                number += i * pow(current, power)
                power += 1
        else:
            number = num

        if number == 0:
            return [0]

        result = [0] * math.ceil(math.log(number) / math.log(to))
        for i in range(-1, -len(result)-1, -1):
            result[i] = number % to
            number = (number - result[i]) / to

        return result
