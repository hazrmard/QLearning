"""
Defines the `Approximator` class and sub-classes. `Approximator` uses various
models to learn input -> output mappings. Available methods are:

* `Polynomial` function approximation,
* Artificial `Neural` network approximation,
* `Tabular` approximation.
"""

from .approximator import Approximator
from .polynomial import Polynomial
from .neural import Neural
from .tabular import Tabular