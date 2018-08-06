"""
Implemants fully-connected artificial neural network function approximation.
"""

import warnings
from typing import Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.neural_network import MLPRegressor

from .polynomial import Polynomial


class Neural(Polynomial):
    """
    A fully-connected neural network to approximate a function.

    Args:
    * hidden_layer_sizes: A tuple of ints representing number of units in each
    hidden layer.
    * random_state: Integer seed or `np.random.RandomState` instance.
    * norm: Bounds to use to normalize input between [0, 1]. Must be same size
    as inputs.
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to `sklearn.neural_network.MPLRegressor`
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`.
    """


    def __init__(self, hidden_layer_sizes: Tuple[int],\
        norm: Tuple[Tuple[float, float]] = None, default = 0.,\
        random_state: Union[int, RandomState] = None, **kwargs):
        self.default = default

        kwargs['random_state'] = random_state   # to be passed to MLPRegressor
        self.model = MLPRegressor(hidden_layer_sizes, **kwargs)

        self.bounds = np.asarray(norm).T if norm is not None else None
        self.range = self.bounds[1] - self.bounds[0] if norm is not None else None


    def _project(self, x: np.ndarray) -> np.ndarray:
        if self.bounds is not None:
            return (x - self.bounds[0]) / self.range
        return x


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        # disable convergence warning for incremental learning
        with warnings.catch_warnings():
            warnings.simplefilter('once')
            return super().update(x, y)
