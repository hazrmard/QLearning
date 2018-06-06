"""
Implements the multi-step q-learning algorithm, known as TD(lambda).
"""
import numpy as np
from gym.core import Env

from ..agent import Agent
from ..agent.helpers import to_space, to_tuple


def tdlambda(agent: Agent, discount: float, steps: int=0):
    """
    Temporal difference learning with no look-ahead. Uses value iteration to
    learn policy. Value function is incrementaly learned. New estimate of value
    is: `V'(s, a) = discounted future reward + discounted V(s', a)`.

    Args:
    * agent: The agent calling the learning function.
    * discount: The discount level for future rewards. Between 0 and 1.
    * steps: The number of steps to accumulate reward.
    """
    # TODO: move updates to end of each episode as a batch call to update()
    states = [to_tuple(agent.env.observation_space, agent.env.reset())]
    actions = []
    rewards = []
    done = False
    T = np.inf      # termination time (i.e. terminal state)
    tau = 0         # time of state being updated
    t = 0           # time from beginning of episode

    # update states until penultimate state
    while tau < T - 1:
        # observe new states until episode ends
        if t < T:
            state = states[-1]
            action = agent.next_action(state)    # select exploratory action
            actions.append(action)               # store history of actions
            # observe next state and rewards
            nstate, reward, done, _ = agent.env.step(to_space(agent.env.action_space, action))
            nstate = to_tuple(agent.env.observation_space, nstate)
            states.append(nstate)    # store history of states
            rewards.append(reward)   # accumulate rewards for aggregation
            if done:
                T = t + 1
                if (T - steps) < 0:
                    steps = T
        
        # if the minimum number of rewards have been observed, start calculating
        # estimates of return
        tau = t - steps
        if tau >= 0:
            ret = 0
            for k in range(tau, min(tau + steps + 1, T - 1)):
                ret += discount**k * rewards[k]
            nvalue, _ = agent.maximum(states[k+1])
            ret += discount**(k+1) * nvalue
            # update value function with new estimate
            # print('t:{}, tau:{}, T:{}, s:{}, ret:{}'.format(t, tau, T, steps, ret))
            agent.value.update((*states[tau], *actions[tau]), ret)
        t += 1
    return states, actions