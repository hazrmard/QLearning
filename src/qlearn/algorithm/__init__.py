"""
Defines various algorithms for reinforcement learning.
"""

# from .variablenstep import variablenstep
from .q import q
from .nsteptd import nsteptd
from .nstepsarsa import nstepsarsa

# TODO: compare performance of using collections.deque for storing reward, action
# histories vs using lists.

# TODO: compare performance of using multiple processes to calculate maximum
# q-values for each sample from memory batch vs. using a single thread.