import unittest
from collections import namedtuple

import gym
import numpy as np

from .memory import Memory
from .gpi import Agent, UNIFORM, GREEDY, SOFTMAX



class TestAgent(unittest.TestCase):


    def setUp(self):

        # dummy environment
        state_space = gym.spaces.Box(low=-10, high=9, shape=(1,), dtype=int)
        action_space = gym.spaces.Discrete(2)
        state = [state_space.sample()]
        # simple number line centered at 0. Action=0 is left, action=1 is right.
        # states range from -10 to 9 inclusive.
        tfunc = lambda s, a: abs(s) + (2*a - 1)
        # rewarded for moving closer to 0
        rfunc = lambda s, a, ns: abs(s) - abs(ns)
        def step(a):
            ns = tfunc(state[-1], a)
            r = rfunc(state[-1], a, ns)
            d = ns >= 10
            state[-1] = ns
            return ns, r, d, None
        def reset():
            return state_space.sample()
        Env = namedtuple('Env', ('step', 'reset', 'render', 'action_space',\
                                'observation_space'))
        self.env = Env(step, reset, lambda x: None, action_space, state_space)

        # dummy approximator
        class Approx:
            def __init__(self):
                self.values = np.zeros((20, 2))
            def predict(self, x):
                sind = int(10 + x[:, 0])
                aind = int(x[0, 1])
                return np.array((self.values[sind, aind],))
            def update(self, x, y):
                sind = int(10 + x[0, 0])
                aind = int(x[0, 1])
                self.values[sind, aind] = y[0]
            def __getitem__(self, *args):
                self.predict(*args)
            def __call__(self, *args, **kwargs):
                self.update(*args, **kwargs)
        self.approx = Approx()


    def test_agent_recommend(self):
        agent = Agent(self.env, self.approx)
        self.approx.values[0:10, 0] = 1
        self.approx.values[10:20, 1] = 1
        action = agent.recommend((-5,))
        self.assertEqual(action, (0,))


    def test_uniform_policy(self):
        agent = Agent(self.env, self.approx)
        agent.set_action_selection_policy(UNIFORM)
        actions = []
        for i, ep in enumerate(agent.episodes()):
            if i >= 1000:
                break
            actions.append(agent.next_action(ep))
        mean = np.asarray(actions).mean()
        self.assertLess(mean, 1)
        self.assertGreater(mean, 0)


    def test_greedy_policy(self):
        agent = Agent(self.env, self.approx)
        agent.set_action_selection_policy(GREEDY)
        agent.eps_curr = 0.0 # no exploration, only exploitation
        self.approx.values = np.zeros((20, 2))
        self.approx.values[0:10, 1] = 1
        self.approx.values[10:20, 0] = 1
        actions = []
        states = []
        for i, ep in enumerate(agent.episodes()):
            if i >= 100:
                break
            actions.append(agent.next_action(ep))
            states.append(ep)
        states = np.asarray(states)
        actions = np.asarray(actions)
        self.assertTrue(np.all(actions[states < 0] == 1) and \
                        np.all(actions[states >= 0] == 0))


    def test_softmax_policy(self):
        pass



class TestMemory(unittest.TestCase):

    def test_sample(self):
        for x in range(1, 5):
            for y in range(1, 5):
                m = Memory(x, y)
                for i in range(10):
                    m.append(i)
                    s = m.sample()
                    self.assertEqual(len(s), y)
                    self.assertTrue(all([z in m for z in s]))



if __name__ == '__main__':
    unittest.main(verbosity=0)
