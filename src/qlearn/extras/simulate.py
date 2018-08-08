from typing import Union

import gym
from gym.core import Env

from ..agent import Agent
from ..helpers import spaces


def simulate(agent: Agent, envId: Union[str, Env], episodes: int=3,
             render: bool=True, verbose: bool=True, random: bool=False):
    """
    Simulates an agent in an environment. Able to visualize behaviour or return
    statistics silently.

    Args:

    * agent: An `Agent` instance.
    * envId: A string ID for a `gym` registered environment or an `Environment`
    instance itself.
    * episodes: Number of episodes to simulate over.
    * render: Whether to visualize agent (if `render` method available).
    * verbose: Whether to print out total rewards after each episode.
    * random: Whether to use random agent behaviour (as a benchmark)

    Returns:
    * A list of total rewards for each episode.
    """
    rewards = []
    if isinstance(envId, str):
        env = gym.make(envId)
    else:
        env = envId
    state = env.reset()
    if render: env.render()
    reward = 0.
    i = 0
    while True:
        state = spaces.to_tuple(env.observation_space, state)
        if random:
            action = spaces.to_tuple(env.action_space, env.action_space.sample())
        else:
            action = agent.recommend(state)
        nstate, r, done, _ = env.step(spaces.to_space(env.action_space, action))
        reward += r
        state = nstate
        if render: env.render()
        if done:
            i += 1
            state = env.reset()
            if verbose: print('Reward: {}'.format(reward))
            rewards.append(reward)
            reward = 0.
            if i >= episodes: break
    env.close()
    return rewards
