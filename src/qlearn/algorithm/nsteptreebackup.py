"""
An implementation of the variable n-step tree backup algorithm. Where the current
value of a state/action pair is updated by the discounted expected reward of
choosing n-actions from the current state. If n=1, then it is QLearning.

The size of a step can be a function of the current state.

All learning algorithms should be able to learn from a single state/action pair.
All learning algorithms return a tuple of lists:
    [list of states traversed after the initial state, including final state],
    [list of actions taken to traverse states, starting with the first action]
"""


import numpy as np

from ..helpers.spaces import to_space, to_tuple, len_space_tuple


def nsteptreebackup(agent: 'Agent', memory: 'Memory', discount: float, steps: int=0,\
    maxsteps: int=np.inf, **kwargs) -> float:
    """
    Begins learning procedure over all (state, action) pairs. Calculates errors
    between last and current estimation of q-value and calls agent.update to
    modify policy.
    Implements the n-step Tree Backup algorithm which is a variation of
    QLearning. The algorithm is modified to allow for variable step sizes.
    Compatible with integer and vector representation of states and actions.
    See Reinforcement Learning - an Introduction by Sutton/Barto (Ch. 7)

    Args:
    * agent: The agent calling the learning function.
    * memory: A Memory instance that can store and sample past observations.
    * discount: The discount level for future rewards. Between 0 and 1.
    * steps: The number of steps to accumulate reward.
    * maxsteps: Number of steps at most to take if episode continues.
    * kwargs: All other keyword arguments discarded silently.
    """

    T = np.inf      # termination time (i.e. terminal state)
    tau = 0         # time of state being updated
    t = 0           # time from beginning of episode
    delta = []      # error in current and next value estimate at time t
    Q = []          # history of Q-values of taken actions
    A = []          # history of actions taken for n-step lookahead
    S = [to_tuple(agent.env.observation_space, agent.env.reset())]     # history of states taken
    pi = [1]        # history of action probabilities for each state

    A.append(agent.next_action(state) if action is None else action)
    Q.append(agent.qvalue(state, A[-1]))

    # Loop from start of episode until the state before terminal state
    while tau <= T-1 and t < agent.depth:
        # The algorithm looks n-steps ahead of the state in the episode
        # being updated. If a terminal state comes before n-steps, it
        # stops looking ahead.
        if t < T:
            action = A[-1]                          # current action
            state = S[-1]                           # current state
            naction = agent.next_action(state)       # next action
            step = agent.stepsize(state)             # size of lookahead
            nstate = agent.next_state(state, action, stepsize=step) # next state
            cqvalue = agent.qvalue(state, action)    # current Q-value
            nqvalue = agent.qvalue(nstate, naction)  # next Q-value

            A.append(naction)
            S.append(nstate)
            Q.append(nqvalue)

            reward = agent.reward(state, action, nstate, stepsize=step)
            aprobs = agent.a_probs(nstate)
            # For QLearner subclasses with vector representation,
            # naction cannot be used as an index
            if isinstance(naction, (int, np.integer)):
                pi.append(aprobs[naction])
            else:
                pi.append(aprobs[agent.actionconverter.encode(naction)])

            if agent.goal(nstate):   # Episode stops look-ahead by
                T = t + 1           # updating T from infinity to t+1
                delta.append(reward - cqvalue)
            else:
                delta.append(
                    reward \
                    + agent.discount * np.dot(aprobs, agent.qvalue(nstate))\
                    - cqvalue
                    )
        # In the second step, the algorithm updates a state's value
        # using the errors/rewards computed from the look-ahead.
        tau = t - agent.steps + 1 # tau trails look-ahead (t) by n-steps
        if tau >= 0:
            E = 1
            G = Q[tau]  # G is the expected return using n-step lookahead
            # Iterating from current state to n-steps ahead or terminal
            # state (whichever's closer), computes the n-step error
            # and updates q-matrix accordingly.
            for k in range(tau, min(tau + agent.steps, T)):
                G += E * delta[k]
                E = agent.discount * E * pi[k+1]
            agent.update(S[tau], A[tau], agent.qvalue(S[tau], A[tau]) - G)
        t += 1
    return S[1:], A
