from typing import List, Tuple, Union, Iterable, Callable
import itertools
import numpy as np
from gym.spaces import MultiBinary, MultiDiscrete
from gym.spaces import Tuple as TupleSpace


def enumerate_discrete_space(space: Union[MultiBinary, MultiDiscrete, TupleSpace])\
    -> List[Tuple[int]]:
    """
    Generates a list of soace coordinates corresponding the the discrete space.
    i.e a (2,2) MultiDiscrete space becomes [(0,0), (0,1), (1,0), (1,1)].

    Args:
    * space: A discrete space instance from `gym.spaces`. Can be MultiBinary or
    MultiDiscrete.

    Returns:
    * A list of tuples of coordinates for each point in the space. For invalid
    arguments, returns empty list.
    """
    linspaces = []
    if isinstance(space, MultiBinary):
        linspaces = [(0, 1)] * space.n
    elif isinstance(space, MultiDiscrete):
        linspaces = [np.linspace(0, n-1, n, dtype=space.dtype) for n in space.nvec]
    elif isinstance(space, TupleSpace):
        subspaces_enum = []
        for subspace in space.spaces:
            subspaces_enum.append(enumerate_discrete_space(subspace))
        return list(itertools.product(*subspaces_enum))
    grids = np.meshgrid(*linspaces)
    return list(zip(*[g.ravel() for g in grids]))



def max_discrete(func: Callable[[Tuple], float], over: [Iterable[Tuple]]):
    """
    Calculates the maximum value of a function over a discrete space.

    Args:
    * func: The function that accepts a tuple of arguments and returns a float
    to maximize.
    * over: An iterable of tuples to maximize over.

    Returns a tuple of:
    * The maximum value,
    * The corresponding argument tuple.
    """
    vals = [func(args) for args in over]
    maximum = max(vals)
    return (maximum, over[vals.index(maximum)])



def max_continuous(func: Callable[[Tuple], float], over: [Iterable[Tuple]]):
    """
    Calculates the maximum value of a function over a continuous range.

    Args:
    * func: The function that accepts a tuple of arguments and returns a float
    to maximize.
    * over: An iterable of tuples describing the "box" to maximize over.

    Returns a tuple of:
    * The maximum value,
    * The corresponding argument tuple.
    """
    pass