"""
Implements the base Approximator class.
"""
from typing import Tuple, Union
import numpy as np


class Approximator:


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        """
        Update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.
        * y (Tuple, ndarray): A *1D* array of values to be learned at that point.
        """
        pass
    

    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Predict value from the learned function given the input x.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.

        Returns:
        * A *1D* array of predictions for each feature in `x`.
        """
        pass
    

    def __getitem__(self, x):
        return self.predict(x)
    

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)