"""
Tabular function approximation.
"""

from typing import Union, Tuple
import numpy as np
from .approximator import Approximator



class Tabular(Approximator):
    """
    A tabular approximation of a function. Uses gradient descent to update each
    repeated discrete observation using squared loss. Does not interpolate/
    extrapolate for unseen data points. Instead returns default value (0).
    Assumes all arguments to function are positive integers.

    Args:
    * shape: The tuple containing size of each dimension in table.
    * lrate: A 0 < float <= 1. representing the learning rate.
    """


    def __init__(self, dims: Tuple[int], lrate: float=1., default=0.):
        self.lrate = lrate
        self.table = np.zeros(dims)
        self.default = default


    def update(self, x: Union[np.ndarray, Tuple], y: float):
        key = tuple(x)
        error = self.table[key] - y
        self.table[key] -= self.lrate * error


    def predict(self, x: Union[np.ndarray, Tuple]):
        key = tuple(x)
        return self.table[key]