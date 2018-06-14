"""
Implements a polynomial function approximation.
"""

from typing import Tuple, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

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
    # TODO: Move memory/ experience replay to algorithms.

    def __init__(self, order: int, memory_size: int, batch_size: int, \
                default=0., tol: float=1e-3, **kwargs):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.default = default
        self.memory = []
        self.order = order
        self.transformer = PolynomialFeatures(degree=order, include_bias=False)
        # self.powers: np.ndarray = np.arange(1, self.order + 1)
        self.powers = np.arange(1, self.order + 1)[:, None]
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
        # return (x[:, None] ** self.powers).reshape(-1, order='f')  # 2D
        # return (x[:, None] ** self.powers).reshape(x.shape[0], -1, order='c')

        # TODO: A faster polynomial feature generator.
        return self.transformer.fit_transform(x)


    def _minibatch_from_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly samples instances from the replay memory.
        """
        indices = np.random.randint(0, len(self.memory), self.batch_size)
        chosen = [self.memory[i] for i in indices]
        xdata, ydata = list(zip(*chosen))
        return (np.array(xdata), np.array(ydata))


    def update(self, x: Union[np.ndarray, Tuple], y: np.ndarray):
        """
        Incrementally update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.
        * y (ndarray): A *1D* array of values to be learned at that point.
        """
        x = np.asarray(x)
        # Maintain memory by truncaitng excess if size specified
        if self.memory_size > 0:
            self.memory.extend(zip(self._project(x), y))
            if self.memory_size < len(self.memory):
                self.memory = self.memory[len(self.memory)-self.memory_size:]
            xdata, ydata = self._minibatch_from_memory()
        else:
            xdata, ydata = np.asarray(x), np.asarray(y)
        self.model.partial_fit(xdata, ydata)


    def predict(self, x: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Predict value from the learned function given the input x.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.

        Returns:
        * A *1D* array of predictions for each feature in `x`.
        """
        x = np.asarray(x)
        try:
            # return self.model.predict(self._project(x).reshape(1, -1))[0]
            return self.model.predict(self._project(x)).ravel()
        except NotFittedError:
            return np.asarray(self.default).ravel()
