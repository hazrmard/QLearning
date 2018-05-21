"""
Implemants fully-connected artificial neural network function approximation.
"""

from typing import Tuple, Union
import warnings
import numpy as np
from .polynomial import Polynomial
from sklearn.neural_network import MLPRegressor



class Neural(Polynomial):


    def __init__(self, hidden_layer_sizes: Tuple[int], memory_size: int,\
                batch_size: int, **kwargs):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.model = MLPRegressor(hidden_layer_sizes, warm_start=True, max_iter=1,\
                                batch_size=batch_size, **kwargs)


    def _project(self, x: np.ndarray) -> np.ndarray:
        return x


    def update(self, x: Union[np.ndarray, Tuple], y: float):
        # disable convergence warning for incremental learning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return super().update(x, y)
