"""
Defines the `Approximator` class and sub-classes. `Approximator` uses various
models to learn input -> output mappings. Available methods are:

* `Polynomial` function approximation,
* Artificial `Neural` network approximation,
* `Tabular` approximation.

All `Approximator`s have the following API:

* Methods:
  * update(x, y): Takes a 2D array of training features in each row and a 1D
  array of target values.
  * predict(x): Takes a 2D array of features in each row and returns a 1D array
  of predicted values.
"""

from .approximator import Approximator
from .polynomial import Polynomial
from .neural import Neural
from .tabular import Tabular
