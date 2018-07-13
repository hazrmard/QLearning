"""
Contains on-policy temporal difference agents:

* N-step SARSA

Note: On-policy agents are better for value function approximation since
they use the action selection policy to estimate future returns (as opposed
to off-policy algorithms like q-learning which uses the *most* valuable
state/action pair as a value backup). Therefore the expected future returns
will reflect the true value of states (and not a biased estimate).
"""

import numpy as np

from ...helpers.parameters import Schedule
from ...algorithm import nstepsarsa
from .agent import Agent, GREEDY



class NStepSarsaAgent(Agent):
    """
    Implements n-step SARSA: On-policy temporal difference learning which
    considers discounted delayed rewards.
    """

    def learn(self, episodes: int=100, policy: str=GREEDY,\
        epsilon: Schedule=Schedule(0,), **kwargs) -> np.ndarray:
        """
        Calls the n-step SARSA `episodes` times.

        Args:
        * episodes: Number of eposides to learn over.
        * policy (str): The action selection policy. Used durung learning/
        exploration to randomly select actions from a state. One of
        `agent.[UNIFORM | GREEDY | SOFTMAX]`. Default UNIFORM.
        * episolon: A `Schedule` instance describing how the exploration rate
        changes for each episode (for GREEDY policy).
        * discount: The discount level for future rewards. Between 0 and 1.
        * steps: The number of steps to accumulate reward.
        * maxsteps: Number of steps at most to take if episode continues.
        * memsize: Size of experience memory. Default 1 most recent observation.
        * batchsize: Number of past experiences to replay. Default 1.
        If a parameter is a `Schedule`, it is evaluated for each episode and
        passed as a number.

        Returns:
        * An array of rewards for each episode.
        """
        return super().learn(algorithm=nstepsarsa, episodes=episodes, policy=policy,
            epsilon=epsilon, **kwargs)