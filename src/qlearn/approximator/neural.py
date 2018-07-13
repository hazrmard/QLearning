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
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to `sklearn.neural_network.MPLRegressor`
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`.
    """


    def __init__(self, hidden_layer_sizes: Tuple[int], default = 0.,\
        random_state: Union[int, RandomState] = None, **kwargs):
        self.default = default
        kwargs['random_state'] = random_state   # to be passed to MLPRegressor
        self.model = MLPRegressor(hidden_layer_sizes, **kwargs)


    def _project(self, x: np.ndarray) -> np.ndarray:
        return x


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        # disable convergence warning for incremental learning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return super().update(x, y)
