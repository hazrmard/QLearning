"""
Implements the standard q-learning algorithm, known as TD(0) for a single
episode. Q-learning is an off-policy temporal difference learning algorithm.
"""
from typing import List

from ..agent.spaces import to_space, to_tuple


def q(agent: 'Agent', discount: float) -> List[float]:
    """
    Q-learning: Off-policy Temporal difference learning with no look-ahead.
    Uses value iteration to learn policy. Value function is incrementaly learned.
    New estimate of value is:

        `V'(s, a) = reward + discount * max_{a'}V(s', a')`

    Note: Temporal difference methods with off-policy and non-tabular value
    function approximations may not converge [4.2 Ch. 11.3 - Deadly Triad].

    Args:
    * agent: The agent calling the learning function.
    * discount: The discount level for future rewards. Between 0 and 1.
    """
    states = [to_tuple(agent.env.observation_space, agent.env.reset())]
    actions = []
    rewards = []
    done = False
    while not done:
        state = states[-1]
        # select exploratory action
        action = agent.next_action(state)
        # observe next state and rewards
        nstate, reward, done, _ = agent.env.step(to_space(agent.env.action_space, action))
        nstate = to_tuple(agent.env.observation_space, nstate)
        # calculate new estimate of return
        nvalue, _ = agent.maximum(nstate)
        ret = reward + discount * nvalue
        # update value function with new estimate
        agent.value.update(((*state, *action),), (ret,))
        states.append(nstate)
        actions.append(action)
        rewards.append(reward)
        # print(state, action, ret)
    return reward