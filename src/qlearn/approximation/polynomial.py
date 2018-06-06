"""
Implements a polynomial function approximation.
"""

from typing import Tuple, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor

from .approximator import Approximator


class Polynomial(Approximator):
    """
    Incremental approximation of a function using polynomial basis.

    Args:
    * order (int): The order of the polynomial (>=1).
    * memory_size: The number of last observations to remember for learning.
    * batch_size: The minibatch to generate and use at each call to `update`.
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to sklearn.linear_model.SGDRegressor
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`,
    and `fit_intercept`.
    """

    def __init__(self, order: int, memory_size: int, batch_size: int, \
                default=0., tol: float=1e-3, **kwargs):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.default = default
        self.memory = []
        self.order = order
        self.powers: np.ndarray = np.arange(1, self.order + 1)
        self.model = SGDRegressor(fit_intercept=True, tol=tol, **kwargs)


    def _project(self, x: np.ndarray) -> np.ndarray:
        """
        Converts an input instance into higher dimensions consistemt with the
        order of the polynomial being approximated.

        Args:
        * x (np.ndarray): A *2D* array representing a single instance.
        """
        # use broadcasting to calculate powers of x in a 2D array, then reshape
        # it into a single array.
        # See: https://stackoverflow.com/q/50428668/4591810
        # order='f' : [x1^1, x2^1, x1^2, x2^2...],
        # order='c' : [x1^1, x1^2, ..., x2^1, x2^2]
        return (x[:, None] ** self.powers).reshape(-1, order='f')  # 1D
        # return (x[:, None] ** self.powers[:, None]).reshape(x.shape[0], -1, order='c')


    def _minibatch_from_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly samples instances from the replay memory.
        """
        indices = np.random.randint(0, len(self.memory), self.batch_size)
        chosen = [self.memory[i] for i in indices]
        xdata, ydata = list(zip(*chosen))
        return (np.array(xdata), np.array(ydata))


    def update(self, x: Union[np.ndarray, Tuple], y: float):
        """
        Incrementally update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.
        * y (float): The value to be learned at that point.
        """
        # TODO: update using multiple instances added to memory at once
        x = np.asarray(x)
        self.memory.append((self._project(x), y))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        xdata, ydata = self._minibatch_from_memory()
        self.model.partial_fit(xdata, ydata)


    def predict(self, x: Union[np.ndarray, Tuple]) -> float:
        """
        Predict value from the learned function given the input x.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.

        Returns:
        * A float representing the approximate function value.
        """
        x = np.asarray(x)
        try:
            return self.model.predict(self._project(x).reshape(1, -1))[0]
        except NotFittedError:
            return self.default
