"""
Implements the standard q-learning algorithm, known as TD(0).
"""
from gym.core import Env

from ..agent.helpers import to_space, to_tuple


def td0(agent: 'Agent', discount: float):
    """
    Temporal difference learning with no look-ahead. Uses value iteration to
    learn policy. Value function is incrementaly learned. New estimate of value
    is: `V'(s, a) = reward + discount * V(s', a)`.

    Args:
    * agent: The agent calling the learning function.
    * discount: The discount level for future rewards. Between 0 and 1.
    """
    # TODO: move updates to end of each episode as a batch call to update()
    states = [to_tuple(agent.env.observation_space, agent.env.reset())]
    actions = []
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
        agent.value.update((*state, *action), ret)
        states.append(nstate)
        actions.append(action)
        # print(state, action, ret)
    return states, actions