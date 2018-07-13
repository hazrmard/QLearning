"""
Tabular function approximation.
"""

from typing import Iterable, Tuple, Union

import numpy as np
from numpy.random import RandomState

from ..helpers.parameters import Schedule
from .approximator import Approximator


class Tabular(Approximator):
    """
    A tabular approximation of a function. Uses gradient descent to update each
    repeated discrete observation using squared loss. Does not interpolate/
    extrapolate for unseen data points. Instead returns default value (0).
    Assumes all arguments to function are positive integers.

    Args:
    * dims: The tuple containing size of each dimension in table.
    * lrate: A 0 < float <= 1. representing the learning rate. Or a `Schedule`
    instance for a decaying learning rate.
    * low: The lowest limits for each dimension. Defaults 0. Inclusive.
    * high: The highest limits for each dimension. Defaults to dimension sizes.
    Inclusive.
    * random_state: Integer seed or `np.random.RandomState` instance.
    """

    def __init__(self, dims: Tuple[int], lrate: Union[float, Schedule]=1e-2,\
        low: Union[float, Tuple[int]]=0, high: Union[float, Tuple[int]]=None,\
        random_state: Union[int, RandomState]=None):
        self.lrate = Schedule(lrate) if not isinstance(lrate, Schedule) else lrate
        self.shape = np.asarray(dims)
        self.low = np.asarray(low)
        self.high = np.asarray(dims) if high is None else np.asarray(high) 
        self.range = self.high - self.low
        self.random = random_state if isinstance(random_state, RandomState)\
                      else RandomState(random_state)
        self.table = self.random.uniform(-0.5, 0.5, size=dims)


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
        self.table[indices] -= self.lrate() * error


    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        keys = [self.discretize(x_) for x_ in x]
        return self.table[list(zip(*keys))]
