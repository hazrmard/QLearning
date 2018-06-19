"""
Defines the `Memory` class that is used internally by `Agent` for experience
replay.
"""

from collections import deque

import numpy as np



class Memory(deque):
    """
    Memory is a `deque` of observation tuples with a fixed size.

    Args:
    * memsize: Maximum size of experience memory.
    * batchsize: Size of each sample during replay.
    """

    def __init__(self, memsize: int=1, batchsize: int=1):
        self.memsize = memsize
        self.batchsize = batchsize
        super().__init__(maxlen=memsize)


    def sample(self):
        """
        Generate a random sample (w/o replacement) from a memory.

        Returns:
        * A list of randomly sampled tuples.
        """
        indices = np.random.randint(0, len(self), self.batchsize)
        return [self[i] for i in indices]
