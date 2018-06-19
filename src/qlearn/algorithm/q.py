"""
Implements the standard q-learning algorithm, known as TD(0) for a single
episode. Q-learning is an off-policy temporal difference learning algorithm.
"""
import numpy as np

from ..agent.spaces import to_space, to_tuple, len_space_tuple


def q(agent: 'Agent', memory: 'Memory', discount: float, maxsteps: int=np.inf,\
    **kwargs) -> float:
    """
    Q-learning: Off-policy Temporal difference learning with no look-ahead.
    Uses value iteration to learn policy. Value function is incrementaly learned.
    New estimate of value is:

        `Q'(s, a) = reward + discount * max_{a'}Q(s', a')`

    Note: Temporal difference methods with off-policy and non-tabular value
    function approximations may not converge [4.2 Ch. 11.3 - Deadly Triad].

    Args:
    * agent: The agent calling the learning function.
    * memory: A Memory instance that can store and sample past observations.
    * discount: The discount level for future rewards. Between 0 and 1.
    * maxsteps: Number of steps at most to take if episode continues.
    * kwargs: All other keyword arguments discarded silently.
    """
    state = to_tuple(agent.env.observation_space, agent.env.reset())
    rewards = []        # history of rewards
    done = False
    t = 0   # keeping track of steps

    # preallocate arrays for states (X) -> value targets (Y) for approximator
    batchX = np.zeros((memory.batchsize, len(state) + \
                        len_space_tuple(agent.env.action_space)))
    batchY = np.zeros(memory.batchsize)

    while (not done) and (t < maxsteps):
        t += 1
        # select exploratory action
        action = agent.next_action(state)
        # observe next state and rewards
        nstate, reward, done, _ = agent.env.step(to_space(agent.env.action_space, action))
        nstate = to_tuple(agent.env.observation_space, nstate)
        
        # memorize experience
        memory.append((state, action, reward, nstate))
        # replay experience from memory
        samples = memory.sample()
        for i, (s, a, r, ns) in enumerate(samples):
            # calculate new estimate of return
            nvalue, _ = agent.maximum(ns)
            ret = r + discount * nvalue
            # fill batch with state/actions -> values
            batchX[i] = [*s, *a]
            batchY[i] = ret
        # update value function with new estimate
        agent.value.update(batchX, batchY)

        state = nstate
        rewards.append(reward)
    return np.sum(rewards)
