"""
Contains off-policy temporal difference agents:

* Q-Learning
* TD(lambda)
* N-step Tree Backup
"""

import numpy as np

from .agent import Agent, GREEDY
from .parameters import Schedule
from ..algorithm import q, tdlambda



class QAgent(Agent):
    """
    Implements Q-Learning: Off-policy temporal difference learning which
    only considers immediate rewards.
    """

    def learn(self, episodes: int=100, policy: str=GREEDY,\
        epsilon: Schedule=Schedule(0,), **kwargs) -> np.ndarray:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * episolon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * kwargs: Any learning parameters required by the learning function.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * An array of rewards for each episode.
        """
        return super().learn(algorithm=q, episodes=episodes, policy=policy,
            epsilon=epsilon, **kwargs)




class TDLambdaAgent(Agent):
    """
    Implements `td(lambda)`: Off-policy temporal difference learning with
    delayed rewards up to a horizon of `lambda` steps into the future.
    """

    def learn(self, episodes: int=100, policy: str=GREEDY,\
        epsilon: Schedule=Schedule(0,), **kwargs) -> np.ndarray:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * episolon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * kwargs: Any learning parameters required by the learning function.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * An array of rewards for each episode.
        """
        return super().learn(algorithm=tdlambda, episodes=episodes, policy=policy,
            epsilon=epsilon, **kwargs)
