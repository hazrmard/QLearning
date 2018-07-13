"""
Contains off-policy temporal difference agents:

* Q-Learning
* TD(lambda)
* N-step Tree Backup
"""

import numpy as np

from ...helpers.parameters import Schedule
from ...algorithm import q, nsteptd
from .agent import Agent, GREEDY



class QAgent(Agent):
    """
    Implements Q-Learning: Off-policy temporal difference learning which
    only considers immediate rewards.
    """

    def learn(self, episodes: int=100, policy: str=GREEDY,\
        discount: Schedule=Schedule(1.,), epsilon: Schedule=Schedule(0,),\
        **kwargs) -> np.ndarray:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used during learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * discount: The discount level for future rewards. Between 0 and 1.
        * maxsteps: Number of steps at most to take if episode continues.
        * epsilon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * memsize: Size of experience memory. Default 1 most recent observation.
        * batchsize: Number of past experiences to replay. Default 1.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * An array of rewards for each episode.
        """
        kwargs['discount'] = discount
        return super().learn(algorithm=q, episodes=episodes, policy=policy,
            epsilon=epsilon, **kwargs)




class NStepTDAgent(Agent):
    """
    Implements `n-step TD`: Off-policy temporal difference learning with
    delayed rewards up to a horizon of `n` steps into the future.
    """

    def learn(self, episodes: int=100, policy: str=GREEDY, steps: int=5,
        discount: Schedule=Schedule(1.,), epsilon: Schedule=Schedule(0,),\
        **kwargs) -> np.ndarray:
        """
        Calls the learning algorithm `episodes` times.

        Args:
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * steps: The number of steps to accumulate reward. Default=5.
        * epsilon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * discount: The discount level for future rewards. Between 0 and 1.
        * maxsteps: Number of steps at most to take if episode continues.
        * memsize: Size of experience memory. Default 1 most recent observation.
        * batchsize: Number of past experiences to replay. Default 1.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * An array of rewards for each episode.
        """
        kwargs['discount'] = discount
        return super().learn(algorithm=nsteptd, episodes=episodes, policy=policy,
            epsilon=epsilon, steps=steps, **kwargs)
