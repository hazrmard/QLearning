"""
Defines functions to take weighed integrals over discrete and continuous spaces.
"""

from typing import Callable, Tuple, Union, Iterable

import numpy as np
from scipy.integrate import quad
from gym.core import Space



def integrate_discrete(func: Callable[[Tuple], np.ndarray], over: Iterable[Tuple],\
    state: Tuple[Union[int, float]]):
    pass
