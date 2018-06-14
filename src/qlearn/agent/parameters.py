"""
Defines the `Schedules` class which can vary parameters over training. All
schedules return a sequence of values beginning from `init` at  `step=0` and
ending at `final` at step=`steps`.
"""

from typing import Callable

import numpy as np



class Schedule:
    """
    A constant schedule. Value remains the same at each call.
    """

    def __init__(self, init: float, *args, **kwargs):
        self.init = init
        self.final = init
        self.steps = np.inf


    def __call__(self, at):
        return self.init



class LinearSchedule(Schedule):
    """
    A linearly increasing/decreasing schedule. Value changes from `init` to
    `final` over `steps` calls. The slope of the line is determined by initial
    arguments.
    """

    def __init__(self, init: float, final: float, steps: int):
        self.low, self.high = min(init, final), max(init, final)
        self.init = init
        self.final = final
        self.steps = steps
        if final > init:
            self.slope = (final - init + 1) / steps
        else:
            self.slope = (final - init - 1) / steps


    def __call__(self, at):
        return np.clip(self.init + self.slope * at, self.low, self.high)



class LogarithmicSchedule(Schedule):
    """
    A logarithmically increasing/decreasing schedule. Value changes from `init`
    to `final` over `steps` calls. The base of logarithm is determined by
    initial arguments.
    """

    def __init__(self, init: float, final: float, steps: int):
        self.low, self.high = min(init, final), max(init, final)
        self.init = init
        self.final = final
        self.steps = steps
        self.logbase = np.log(steps) / (abs(final - init))


    def __call__(self, at):
        if self.final > self.init:
            return np.clip(self.init + np.log(at + 1) / self.logbase, self.low,\
                        self.high)
        else:
            return np.clip(self.final + np.log(self.steps - at) / self.logbase,\
                        self.low, self.high)



class ExponentialSchedule(Schedule):

    def __init__(self, init: float, final: float, steps: int):
        a, b = min(init, final), max(init, final)
        self.init = init
        self.final = final
        self.steps = steps


    def __call__(self, at):
        return



def evaluate_schedule_kwargs(at: int, **kwargs):
    """
    Takes the episode/epoch and evaluates any schedule keyword arguments.

    Args:
    * at (int): The episode to evaluate at.
    * kwargs: Any number of keyword arguments with or without a `Schedule` value.

    Returns:
    * A dictionary of the same keys as `kwargs` but with `Schedule` values
    replaced by evaluations.
    """
    return {k: v(at) if isinstance(v, Schedule) else v for k, v in kwargs.items()}
