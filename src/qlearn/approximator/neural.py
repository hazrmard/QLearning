"""
Implemants fully-connected artificial neural network function approximation.
"""

from typing import Tuple, Union
import warnings
import numpy as np
from .polynomial import Polynomial
from sklearn.neural_network import MLPRegressor



class Neural(Polynomial):
    """
    A fully-connected neural network to approximate a function.

    Args:
    * hidden_layer_sizes: A tuple of ints representing number of units in each
    hidden layer.
    * memory_size: The number of last observations to remember for learning.
    * batch_size: The minibatch to generate and use at each call to `update`.
    * kwargs: Any keyword arguments to be fed to `sklearn.neural_network.MPLRegressor`
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`.
    """


    def __init__(self, hidden_layer_sizes: Tuple[int], default = 0., **kwargs):
        # self.memory_size = memory_size
        # self.batch_size = batch_size
        # self.memory = []
        self.default = default
        self.model = MLPRegressor(hidden_layer_sizes, **kwargs)


    def _project(self, x: np.ndarray) -> np.ndarray:
        return x


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        # disable convergence warning for incremental learning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return super().update(x, y)
