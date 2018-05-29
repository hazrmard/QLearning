from typing import List, Tuple, Union, Iterable, Callable
import itertools
from collections import OrderedDict
import numpy as np
from gym.core import Space
from gym.spaces import MultiBinary, MultiDiscrete, Box, Discrete, Dict
from gym.spaces import Tuple as TupleSpace


def enumerate_discrete_space(space: Space) -> List[Tuple[int]]:
    """
    Generates a list of soace coordinates corresponding the the discrete space.
    i.e a (2,2) MultiDiscrete space becomes [(0,0), (0,1), (1,0), (1,1)].

    Args:
    * space: A discrete space instance from `gym.spaces`.

    Returns:
    * A list of tuples of coordinates for each point in the space. For invalid
    arguments, returns empty list.
    """
    linspaces = []
    if isinstance(space, MultiBinary):
        linspaces = [(0, 1)] * space.n
    elif isinstance(space, Discrete):
        linspaces = [np.linspace(0, space.n-1, space.n, dtype=space.dtype)]
    elif isinstance(space, MultiDiscrete):
        linspaces = [np.linspace(0, n-1, n, dtype=space.dtype) for n in space.nvec]
    elif isinstance(space, Box) and np.issubdtype(space.dtype, np.integer):
        # TODO: enumerate integer box spaces and preserve shape
        pass
    elif isinstance(space, TupleSpace):
        subspaces_enum = []
        for subspace in space.spaces:
            subspaces_enum.append(enumerate_discrete_space(subspace))
        return list(itertools.product(*subspaces_enum))
    elif isinstance(space, Dict):
        subspaces_enum = []
        keys = space.spaces.keys()
        for _, subspace in space.spaces.items():
            subspaces_enum.append(enumerate_discrete_space(subspace))
        enumerated = itertools.product(*subspaces_enum)
        dictform = [OrderedDict(zip(keys, value)) for value in enumerated]
        return dictform
    grids = np.meshgrid(*linspaces)
    return list(zip(*[g.ravel() for g in grids]))



def size_discrete_space(space: Space) -> List[Tuple[int]]:
    """
    Returns the number of possible states in a discrete space.
    i.e a (2,2) MultiDiscrete space returns 4.

    Args:
    * space: A discrete space instance from `gym.spaces`.

    Returns:
    * The number of possible states. Infinity if space is `gym.spaces.Box` and
    dtype is not integer.
    """
    n = 1
    if isinstance(space, MultiBinary):
        n = 2 ** space.n
    elif isinstance(space, Discrete):
        n = space.n
    elif isinstance(space, MultiDiscrete):
        n = np.prod(space.nvec)
    elif isinstance(space, Box) and np.issubdtype(space.dtype, np.integer):
        n = np.prod(space.high - space.low)
    elif isinstance(space, Box):
        n = np.inf
    elif isinstance(space, TupleSpace):
        for subspace in space.spaces:
            n *= size_discrete_space(subspace)
    elif isinstance(space, Dict):
        for _, subspace in space.spaces.items():
            n *= size_discrete_space(subspace)
    return n



def len_space_tuple(space: Space) -> int:
    """
    Calculates the length of the flattened tuple generated from a sample from
    space. For e.g. TupleSpace((MultiDiscrete([3,4]), MultiBinary(2))) -> 4

    Args:
    * space: A discrete space instance from `gym.spaces`.

    Returns:
    * The length of the tuple when a space sample is flattened.
    """
    n = 0
    if isinstance(space, MultiBinary):
        n = space.n
    elif isinstance(space, Discrete):
        n = 1
    elif isinstance(space, MultiDiscrete):
        n = len(space.nvec)
    elif isinstance(space, Box):
        n = np.prod(space.low.shape)
    elif isinstance(space, TupleSpace):
        for subspace in space.spaces:
            n += len_space_tuple(subspace)
    elif isinstance(space, Dict):
        for _, subspace in space.spaces.items():
            n += len_space_tuple(subspace)
    return n



def to_tuple(space: Space, sample: Union[Tuple, np.ndarray, int, OrderedDict])\
    -> Tuple:
    """
    Converts a sample from one of `gym.core.Space` instances into a flat tuple
    of values. I.e. a Dict sample {'a':1, 'b':2} becomes (1, 2).

    Args:
    * space (Space): Space instance describing the sample.

    Returns:
    * A flat tuple of state variables.
    """
    if isinstance(space, (Discrete, MultiBinary, MultiDiscrete)):
        return tuple(sample)
    elif isinstance(space, Box):
        return tuple(sample.ravel())
    elif isinstance(space, TupleSpace):
        spaces = zip(space.spaces, sample)
    elif isinstance(space, Dict):
        spaces = zip([subspace for _, subspace in space.spaces.items()],\
                    [subsample for _, subsample in sample.items()])
    flattened = []
    for subspace, subsample in spaces:
        flattened.extend(to_tuple(subspace, subsample))
    return tuple(flattened)



def to_space(space: Space, sample: Tuple) -> Union[tuple, np.ndarray, int,\
    OrderedDict]:
    if isinstance(space, Discrete):
        return sample[0]
    elif isinstance(space, (MultiBinary, MultiDiscrete)):
        return tuple(sample)
    elif isinstance(space, Box):
        return np.asarray(sample).reshape(space.shape).astype(space.dtype)
    elif isinstance(space, TupleSpace):
        aggregate = []
        i = 0
        for subspace in space.spaces:
            if isinstance(subspace, Discrete):
                aggregate.append(to_space(subspace, sample[i:i+1]))
                i += 1
            elif isinstance(subspace, MultiBinary):
                aggregate.append(to_space(subspace, sample[i:i+subspace.n]))
                i += subspace.n
            elif isinstance(subspace, MultiDiscrete):
                aggregate.append(to_space(subspace, sample[i:i+len(subspace.nvec)]))
                i += len(subspace.nvec)
            elif isinstance(subspace, Box):
                aggregate.append(to_space(subspace, sample[i:i+np.prod(subspace.shape)]))
                i += np.prod(subspace.shape)
            elif isinstance(subspace, TupleSpace):
                sub_len = len_space_tuple(subspace)
                aggregate.append(to_space(subspace, sample[i:i+sub_len]))
        return tuple(aggregate)
    elif isinstance(space, Dict):
        aggregate = OrderedDict()
        i = 0
        for name, subspace in space.spaces.items():
            if isinstance(subspace, Discrete):
                aggregate[name] = to_space(subspace, sample[i:i+1])
                i += 1
            elif isinstance(subspace, MultiBinary):
                aggregate[name] = to_space(subspace, sample[i:i+subspace.n])
                i += subspace.n
            elif isinstance(subspace, MultiDiscrete):
                aggregate[name] = to_space(subspace, sample[i:i+len(subspace.nvec)])
                i += len(subspace.nvec)
            elif isinstance(subspace, Box):
                aggregate[name] = to_space(subspace, sample[i:i+np.prod(subspace.shape)])
                i += np.prod(subspace.shape)
            elif isinstance(subspace, TupleSpace):
                sub_len = len_space_tuple(subspace)
                aggregate[name] = to_space(subspace, sample[i:i+sub_len])
        return aggregate
    



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