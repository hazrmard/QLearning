"""
Tabular function approximation.
"""

from typing import Union, Tuple, Iterable
import numpy as np
from .approximator import Approximator



class Tabular(Approximator):
    """
    A tabular approximation of a function. Uses gradient descent to update each
    repeated discrete observation using squared loss. Does not interpolate/
    extrapolate for unseen data points. Instead returns default value (0).
    Assumes all arguments to function are positive integers.

    Args:
    * dims: The tuple containing size of each dimension in table.
    * lrate: A 0 < float <= 1. representing the learning rate.
    * low: The lowest limits for each dimension. Defaults 0. Inclusive.
    * high: The highest limits for each dimension. Defaults to max. indices.
    Inclusive.
    """

    def __init__(self, dims: Tuple[int], lrate: float=1., \
        low: Tuple[int]=(), high: Tuple[int]=(), default: float=0.):
        self.lrate = lrate
        self.table = np.zeros(dims) + default
        self.shape = np.asarray(dims)
        self.low = np.asarray(low) if low else np.zeros((len(dims),))
        self.high = np.asarray(high) if high else np.asarray(dims)
        self.range = self.high - self.low


    def discretize(self, key: Iterable[Union[float, int]]) -> Tuple[int]:
        """
        Converts `x` feature tuples into valid indices that can be used to
        store/access the table.

        Args:
        * key: The feature tuple.

        Returns:
        * The corresponding index into `Tabular.table`.
        """
        key = (np.asarray(key) - self.low) * self.shape / self.range
        key = np.clip(key, 0, self.shape-1)
        return tuple(key.astype(int))


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        y = np.asarray(y)
        keys = [self.discretize(x_) for x_ in x]
        indices = list(zip(*keys))
        error = self.table[indices] - y
        self.table[indices] -= self.lrate * error


    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        keys = [self.discretize(x_) for x_ in x]
        return self.table[list(zip(*keys))]