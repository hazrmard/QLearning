"""
Implements a polynomial function approximation.
"""

from typing import Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

from .approximator import Approximator


class Polynomial(Approximator):
    """
    Incremental approximation of a function using polynomial basis.

    Args:
    * order (int): The order of the polynomial (>=1).
    * random_state: Integer seed or `np.random.RandomState` instance.
    * default (float): The default value to return if predict called before fit.
    * kwargs: Any keyword arguments to be fed to sklearn.linear_model.SGDRegressor
    which fits to the function. Hard-coded arguments are `warm_start`, `max_iter`,
    and `fit_intercept`.
    """

    def __init__(self, order: int, default=0., tol: float=1e-3,\
        random_state: Union[int, RandomState] = None, **kwargs):
        self.default = default
        self.order = order
        self.transformer = PolynomialFeatures(degree=order, include_bias=False)
        self.powers = np.arange(1, self.order + 1)[:, None]
        kwargs['random_state'] = random_state   # to be passed to SGDRegressor
        self.model = SGDRegressor(fit_intercept=True, tol=tol, **kwargs)


    def _project(self, x: np.ndarray) -> np.ndarray:
        """
        Converts an input instance into higher dimensions consistent with the
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


    def update(self, x: Union[np.ndarray, Tuple], y: Union[np.ndarray, Tuple]):
        """
        Incrementally update function approximation using stochastic gradient
        descent.

        Args:
        * x (Tuple/np.ndarray): A *2D* array representing a single instance.
        * y (ndarray): A *1D* array of values to be learned at that point.
        """
        x, y = np.asarray(x), np.asarray(y)
        self.model.partial_fit(self._project(x), y)


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
