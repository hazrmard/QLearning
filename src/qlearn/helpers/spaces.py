"""
Defines functions that convert space variables to and from tuples of floats.
Defines functions to analyze spaces like size of tuple, size of space,
continuous or discrete etc.
"""

from collections import OrderedDict
from itertools import product, zip_longest
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
from gym.core import Space
from gym.spaces import Tuple as TupleSpace
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete



def enumerate_discrete_space(space: Space, prod: bool = True) -> List[Tuple[int]]:
    """
    Generates a list of space coordinate tuples corresponding the the discrete space.
    i.e a (2,2) `MultiDiscrete` space becomes [(0,0), (0,1), (1,0), (1,1)].

    Args:
    * space: A discrete space instance from `gym.spaces`.
    * prod: Whether to return a product or individual enumerations of variables.
    For e.g for Multibinary(2) with product: (0,0), (0,1), (1,0), (1,1) and
    without product: [(0,1), (0,1)].

    Returns:
    * Either iterator of tuples of coordinates for each point in the space. For
    invalid arguments, or for Tuple or Dict spaces with Box
    spaces that are continuous, returned tuples contain 'None' in place.
    Or a list of tuples enumerating each variable in a state individually.
    """
    # TODO: Add resolution parameter to enumerate continuous spaces.
    linspaces = [(None,)]
    if isinstance(space, MultiBinary):
        linspaces = [(0, 1)] * space.n
    elif isinstance(space, Discrete):
        linspaces = [tuple(np.linspace(0, space.n-1, space.n, dtype=space.dtype))]
    elif isinstance(space, MultiDiscrete):
        linspaces = [tuple(np.linspace(0, n-1, n, dtype=space.dtype)) for \
                    n in space.nvec]
    elif isinstance(space, Box):
        if np.issubdtype(space.dtype, np.integer):
            linspaces = [tuple(np.linspace(l, h, h-l+1, dtype=space.dtype)) for \
                        l, h in zip(space.low.ravel(), space.high.ravel())]
        else:
            linspaces = [(None,)] * np.prod(space.shape)
    elif isinstance(space, TupleSpace):
        linspaces = []
        for subspace in space.spaces:
            linspaces.extend(enumerate_discrete_space(subspace, False))
    elif isinstance(space, Dict):
        linspaces = []
        for _, subspace in space.spaces.items():
            linspaces.extend(enumerate_discrete_space(subspace, False))
    return product(*linspaces) if prod else linspaces



def size_space(space: Space) -> List[Tuple[int]]:
    """
    Calculates the size of a space. For continuous spaces, size is simply the
    range of values (# of states is infinite, however). I.e a (2,2) MultiDiscrete
    space returns 4. A Box(low=[0,0], high=[3, 2], dtype=float) returns 6.

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
    elif isinstance(space, Box):
        if np.issubdtype(space.dtype, np.integer):
            n = np.prod(space.high - space.low + 1)  # +1 since high is inclusive
        else:
            n = np.prod(space.high - space.low)
    elif isinstance(space, TupleSpace):
        for subspace in space.spaces:
            n *= size_space(subspace)
    elif isinstance(space, Dict):
        for _, subspace in space.spaces.items():
            n *= size_space(subspace)
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



def bounds(space: Space) -> Tuple:
    """
    Computes the inclusive bounds for each variable in a tuple representing the
    space. So a TupleSpace(MultiDiscrete(2), MultiBinary([3, 5])) will have
    bounds of ((0,1), (0,1), (0,1), (0,2), (0, 4)). Infinite limits are returned
    as None in bounds.

    Args:
    * space (Space): Space instance describing the sample.

    Returns:
    * A flat tuple of inclusive (low, high) bounds for each variable in state.
    """
    if isinstance(space, Discrete):
        return ((0, space.n-1),)
    elif isinstance(space, MultiDiscrete):
        return tuple(zip(np.zeros_like(space.nvec), space.nvec-1))
    elif isinstance(space, MultiBinary):
        return tuple([(0, 1)] * space.n)
    elif isinstance(space, Box):
        bds = zip(space.low.ravel(), space.high.ravel())
        bds = [(None if l==-np.inf else l, None if h==np.inf else h) for \
                    l, h in bds]
        return tuple(bds)
    elif isinstance(space, TupleSpace):
        spaces = space.spaces
    elif isinstance(space, Dict):
        spaces = [subspace for _, subspace in space.spaces.items()]
    flattened = []
    for subspace in spaces:
        flattened.extend(bounds(subspace))
    return tuple(flattened)



def is_bounded(space: Space) -> Tuple[bool]:
    """
    For each variable in a tuple representing the space, returns a boolean
    indicating if its a bounded variable. So a 
    TupleSpace(MultiDiscrete(2), Box([-1, -2], [1, inf])) will return a tuple
    (True, True, True, False).

    Args:
    * space (Space): Space instance describing the sample.

    Returns:
    * A flat tuple of booleans indicating if the corresponding variable is finite.
    """
    if isinstance(space, Discrete):
        return (True,)
    elif isinstance(space, MultiDiscrete):
        return tuple([True] * len(space.nvec))
    elif isinstance(space, MultiBinary):
        return tuple([True] * space.n)
    elif isinstance(space, Box):
        low_inf = space.low == -np.inf
        high_inf = space.high == np.inf
        return tuple((low_inf | high_inf).ravel())
    elif isinstance(space, TupleSpace):
        spaces = space.spaces
    elif isinstance(space, Dict):
        spaces = [subspace for _, subspace in space.spaces.items()]
    flattened = []
    for subspace in spaces:
        flattened.extend(is_bounded(subspace))
    return tuple(flattened)



def is_continuous(space: Space) -> Tuple[bool]:
    """
    Checks whether each variable in space is continuous of discrete. Only True
    for Box space with float dtype.

    Args:
    * space: The `gym.core.Space` instance.

    Returns:
    * A tuple of length equal to variables in space which is True when the
    corresponding variable is continuous.
    """
    continuous = []
    if isinstance(space, Discrete):
        return (False,)
    elif isinstance(space, MultiDiscrete):
        return tuple([False] * len(space.nvec))
    elif isinstance(space, MultiBinary):
        return tuple([False] * space.n)
    elif isinstance(space, Box):
        if np.issubdtype(space.dtype, np.integer):
            res = False
        else:
            res = True
        return tuple([res] * np.prod(space.shape))
    elif isinstance(space, Dict):
        spaces = spaces = [subspace for _, subspace in space.spaces.items()]
    elif isinstance(space, TupleSpace):
        spaces = space.spaces
    for s in spaces:
        continuous.extend(is_continuous(s))
    return tuple(continuous)



def to_tuple(space: Space, sample: Union[Tuple, np.ndarray, int, OrderedDict])\
    -> Tuple:
    """
    Converts a sample from one of `gym.spaces` instances into a flat tuple
    of values. I.e. a Dict sample {'a':1, 'b':2} becomes (1, 2).

    Args:
    * space (Space): Space instance describing the sample.
    * sample: The sample (`space.sample()` result type) taken from the space to
    be flattened.

    Returns:
    * A flat tuple of state variables.
    """
    if isinstance(space, Discrete):
        return tuple((sample,))
    elif isinstance(space, (MultiBinary, MultiDiscrete)):
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
    """
    Reconstructs a `gym.spaces` sample from a flat tuple. Reverse of
    `to_tuple`. I.e. (1, 2) becomes a Dict sample {'a':1, 'b':2}.

    Args:
    * space (Space): Space instance describing the sample.
    * sample: The flat tuple to be reconstructed into a sample (`space.sample()`
    result type).

    Returns:
    * Any one of int, tuple, np.array, OrderedDict depending on space.
    """
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
