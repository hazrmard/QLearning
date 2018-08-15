import numpy as np
from gym.spaces import Discrete, Box

from ..environment import Environment



class Dummy1DContinuous(Environment):
    """
    A 1D line with limits `+/- MAX_X`, `+/- MAX_V`, and `+/- MAX_A` for
    displacement, velocity, and acceleration respectively. There are two
    actions, left (0) and right (1) which cause acceleration. The goal is to
    remain in the middle with low velocity defined by tolerances `GOAL_X_TOL`
    and `GOAL_V_TOL` as fractions of `MAX_X/V` values.

      |---------- 0 ----------|
    -MAX_X                  +MAX_X

    Class attributes:

    * `MAX_V`, `MAX_A`, `MAX_X`: maximum values of displacement, velocity,
    acceleration.
    * `GOAL_X`, `GOAL_V`: The goal displacement and velocity as a fraction
    of maximum values.
    * `GOAL_X_TOL`, `GOAL_V_TOL`: Relative tolerances of goal state values as a
    fraction of maximum values.
    """

    MAX_V = 0.05
    MAX_A = 0.01
    MAX_X = 1.

    GOAL_X = 0.         # As a fraction of MAX_X
    GOAL_V = 0.         # As a fraction of MAX_V

    GOAL_X_TOL = 0.1    # As a fraction of MAX_X
    GOAL_V_TOL = 1.0    # As a fraction of MAX_V


    @classmethod
    def goal_func(cls, state: np.ndarray) -> bool:
        x_rel, v_rel = state
        centre_x = abs(x_rel - cls.GOAL_X)
        centre_v = abs(v_rel - cls.GOAL_V)
        return (centre_v <= cls.GOAL_V_TOL) and (centre_x <= cls.GOAL_X_TOL)


    @classmethod
    def transition_func(cls, state: np.ndarray, action: int) -> np.ndarray:
        x, v = state[0] * cls.MAX_X, state[1] * cls.MAX_V
        dv = cls.MAX_A * (action * 2 - 1)
        v += dv
        v = np.clip((v,), -cls.MAX_V, cls.MAX_V)[0]
        x += v
        x = np.clip((x,), -cls.MAX_X, cls.MAX_X)[0]
        return np.asarray((x / cls.MAX_X, v / cls.MAX_V))


    @classmethod
    def reward_func(cls, state: np.ndarray, action: int, nstate: np.ndarray) -> float:
        if cls.goal_func(nstate):
            return 10
        return -1
        x, v = nstate[0] * cls.MAX_X, nstate[1] * cls.MAX_V
        centre_x = abs(x - cls.GOAL_X)
        centre_v = abs(v - cls.GOAL_V)
        return -centre_x - centre_v


    def __init__(self, random_state=None):
        action_space = Discrete(2)
        observation_space = Box(low=np.asarray((-1, -1)),\
                                high=np.asarray((1, 1)), dtype=float)
        
        accel_steps = self.MAX_V / self.MAX_A
        accel_x =  (accel_steps + 1) * self.MAX_V / 2
        min_goal_steps = (self.MAX_X - abs(self.GOAL_X) - accel_x) / self.MAX_V + accel_steps
        super().__init__(reward=self.reward_func,
                         transition=self.transition_func,
                         goal=self.goal_func,
                         observation_space=observation_space,
                         action_space=action_space,
                         maxsteps=int(2*min_goal_steps),
                         random_state=random_state)
