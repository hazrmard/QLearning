"""
Implements the multi-step q-learning algorithm, known as n-step TD for a single
episode.
"""

import numpy as np

from ..helpers.spaces import to_space, to_tuple, len_space_tuple


def nsteptd(agent: 'Agent', memory: 'Memory', discount: float, steps: int=0,\
    maxsteps: int=np.inf, **kwargs) -> float:
    """
    Off-policy temporal difference learning with delayed rewards. Extension
    of qlearning `q` algorithm with muiti-step loopahead. Uses value
    iteration to learn policy. Value function is incrementaly learned. New
    estimate of value (`V'`) is (`d`=discount, `r`=reward):

        `Q'(s, a) = r + d * max_{a'}Q(s', a') + d^2 * max_{a''}Q(s'', a'') + ...`

    Note: Temporal difference methods with off-policy and non-tabular value
    function approximations may not converge [4.2 Ch. 11.3 - Deadly Triad].

    Args:
    * agent: The agent calling the learning function.
    * memory: A Memory instance that can store and sample past observations.
    * discount: The discount level for future rewards. Between 0 and 1. If -1,
      then return is average of rewards instead of a discounted sum.
    * steps: The number of steps to accumulate reward.
    * maxsteps: Number of steps at most to take if episode continues.
    * kwargs: All other keyword arguments discarded silently.
    """
    # Illustration of various TD methods at timestep 't'. tau is the state whose
    # value is updated. t is the current timestep of the episode.
    # s0  s1  s2  s3  s4  s5  s6  s7
    # a0  a1  a2  a3  a4  a5  a6        a0 is action from s0 -> s1
    # r0  r1  r2  r3  r4  r5  r6        r0 is reward from a0
    # 0   1   2   3   4   5   t  t+1
    #        tau  -   -   -   -         TD(4) - current + delayed rewards
    #                tau  -   -         TD(2) - current + delayed rewards
    #                        tau        TD(0) - only current reward
    states = [to_tuple(agent.env.observation_space, agent.env.reset())]
    actions = []    # history of actions in episode
    rewards = []    # history of rewards for actions
    done = False
    T = maxsteps    # termination time (i.e. terminal state)
    tau = 0         # time of state being updated
    t = 0           # time from beginning of episode i.e current timestep

    # preallocate arrays for states (X) -> value targets (Y) for approximator
    batchX = np.zeros((memory.batchsize, len(states[-1]) + \
                        len_space_tuple(agent.env.action_space)))
    batchY = np.zeros(memory.batchsize)

    # Choice between using discounted total returns, or average returns.
    if discount == -1:
        average_returns = True
        discount = 1
    else:
        average_returns = False

    # update states until penultimate state. No action is taken from the final
    # state, therefore there is no reward to consider.
    while tau < T - 1:
        # Observe new states until episode ends.
        if t < T:
            state = states[-1]
            action = agent.next_action(state)    # select exploratory action
            actions.append(action)               # store history of actions
            # Observe next state and rewards
            nstate, reward, done, _ = agent.env.step(to_space(agent.env.action_space, action))
            nstate = to_tuple(agent.env.observation_space, nstate)
            states.append(nstate)    # store history of states
            rewards.append(reward)   # accumulate rewards for aggregation
            if done:
                T = t + 1
                # If episode ends before sufficient delayed rewards, salvage
                # whatever rewards have accumulated so far.
                if T < steps:
                    steps = t

        # If the episode has proceeded far enough (t >= steps), start estimating
        # returns for states observed earlier.
        tau = t - steps
        if tau >= 0:
            partial_ret = 0.    # partial return (without adding final value)
            horizon = min(tau + steps + 1, T)   # final timestep relative to episode
            lookahead = horizon - tau           # ...relative to tau i.e. interval
            k = 0               # index from current state being valued
            for k in range(lookahead):
                partial_ret += discount**k * rewards[tau + k]

            # memorize experience
            memory.append((states[tau], actions[tau], partial_ret, states[tau+k+1], k))

            # replay experience from memory
            samples = memory.sample()
            for i, (s, a, partial_r, ns, k) in enumerate(samples):
                # calculate new estimate of return
                nvalue, _ = agent.maximum(ns)
                ret = partial_r + discount**(k+1) * nvalue
                # If discount=-1 initially, then calculate mean return instead
                # of discounted total return. k+2 is the number of terms in
                # the return calculation (k+1 rewards + 1 prior value term)
                if average_returns:
                    ret /= k + 2
                # fill batch with state/actions -> values
                batchX[i] = [*s, *a]
                batchY[i] = ret
            # update value function with new estimate
            agent.value.update(batchX, batchY)
        t += 1
    return np.sum(rewards)
