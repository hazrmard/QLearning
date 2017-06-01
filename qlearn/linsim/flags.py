"""
This module defines the FlagGenerator class which converts state numbers into
flag combinations for use in the simulation (and vice versa).
"""

import numbers
import numpy as np


class FlagGenerator:
    """
    FlagGenerator encodes a combination of flags describing the state of a
    system into a number which can be used as an index. Conversely, it can
    decode a number into a unique combination of flags describing a system.

    Args:
        flags: A a sequence of arguments describing the number of states possible
            for each flag. For example:
            * 2, 4. The first flag, will have 2 possible states: 0, 1. The
            second flag will have 4 states: 0, 1, 2, 3.
            * [-1, 1], 2. The first flag will have 3 states: -1, 0, 1. The second
            flag will have 2 states: 0, 1.
            * [-5, 20, 4], 2. The first flag will have 20 states: -5, -4.5..., 4.
            The second flag will have 2 states: 0, 1.

    Instance Attributes:
        flags (list): Stores number of states for each flag.
        states (int): Total number of states possible with the given flags.
    """

    def __init__(self, *flags):
        self._state = -1    # internal count of current state during for loops

        self.bottom = np.zeros(len(flags))
        self.flags = np.zeros(len(flags), dtype=int)
        self.scale = np.ones(len(flags), dtype=float)
        for i, flag in enumerate(flags):
            if isinstance(flag, (tuple, list, np.ndarray)):
                self.bottom[i] = flag[0]
                if len(flag) == 2:
                    self.flags[i] = flag[-1] - flag[0] + 1    # upper/lower inclusive
                elif len(flag) >= 3:
                    self.flags[i] = flag[1]
                    self.scale[i] = (flag[-1] - flag[0] + 1) / flag[1]
            else:
                self.flags[i] = int(flag)

        self.states = 1
        for flag in self.flags:
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

        Args:
            state (int): The state number in basis 10.

        Returns:
            A numpy array of flag values in the same order as provided at
            instantiation.
        """
        if state >= self.states:
            raise ValueError('State number ' + str(state) + ' exceeds possible states.')
        current = 10        # basis of current state number
        state_f = int(state)# state in current basis, becomes list during loop
        flags = []          # state number translated into flags
        for flag_basis in reversed(self.flags):
            state_f = self.convert_basis(current, flag_basis, state_f)
            flags.insert(0, state_f[-1])
            state_f = state_f[:-1]
            current = flag_basis
        return np.array([flags[i] * self.scale[i] + self.bottom[i] for i in range(len(flags))])


    def encode(self, *flags):
        """
        Encodes a sequence of flags into a number representing that state.

        Args:
            flags: A sequence of int arguments representing the state of flags
                in the same order as they were provided at instantiation.
                OR a single list containing flag values.

        Returns:
            Integer state number in base 10.
        """
        if len(flags) == 1 and isinstance(flags[0], (list, tuple, np.ndarray)):
            flags = flags[0]
        flags = [np.round((flags[i] - self.bottom[i]) / self.scale[i]) for i in range(len(flags))]
        state = []
        for i in range(len(flags)-1):
            state.append(flags[i])
            state = list(self.convert_basis(self.flags[i], self.flags[i+1], state))
        state.append(flags[-1])
        state = self.convert_basis(self.flags[-1], 10, state)
        # Since state is base 10 [0-9], list elements can be concatenated
        state = ''.join([str(i) for i in state])
        return int(state)


    @staticmethod
    def convert_basis(current, to, num):
        """
        Convert a number from one basis to another.

        Args:
            current (int): Current basis of number. Must be greater than 1.
            to (int): Basis to convert number to. Must be greater than 1.
            num (int/list): The number to be converted. If int, assumes that
                current basis is 10. Else a list of ints representing the
                number in that basis with least significant part at the end.

        Returns:
            A numpy array of integers with the least significant number at the
            end.
        """
        if isinstance(num, (list, tuple, np.ndarray)):  # convert list into decimal number
            number = 0
            power = 0
            for i in reversed(num):
                number += i * pow(current, power)
                power += 1
        elif isinstance(num, numbers.Number):   # if num is int, assumes it is in base 10
            if current == 10:
                number = int(num)
            else:
                raise ValueError('Integer num only valid for current basis=10.')
        else:
            raise TypeError('"num" is either int (current=10) or a list of int.')

        if number == 0:
            return [0]

        result = np.zeros(int(np.ceil((np.log(number) + 1) / np.log(to))), dtype=int)
        for i in range(-1, -len(result)-1, -1):
            result[i] = number % to
            number = int((number - result[i]) / to)

        return result
