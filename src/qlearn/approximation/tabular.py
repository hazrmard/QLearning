"""
Tabular function approximation.
"""

from typing import Union, Tuple
import numpy as np
from .approximator import Approximator



class Tabular(Approximator):


    def __init__(self, lrate: float=1.):
        self.lrate = lrate
        self.table = {}


    def update(self, x: Union[np.ndarray, Tuple], y: float):
        key = tuple(x)
        try:
            error = self.table[key] - y
        except KeyError:
            self.table[key] = 0
            error = - y
        self.table[key] -= self.lrate * error


    def predict(self, x: Union[np.ndarray, Tuple]):
        key = tuple(x)
        try:
            return self.table[key]
        except KeyError:
            return 0.