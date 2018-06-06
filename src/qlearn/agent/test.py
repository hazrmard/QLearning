import unittest
from collections import OrderedDict, namedtuple

import gym
import numpy as np

from . import helpers
from .agent import Agent, UNIFORM, GREEDY, SOFTMAX


class TestHelpers(unittest.TestCase):


    def setUp(self):
        self.boxspace = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=int)
        self.boxcont = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=float)
        self.discspace = gym.spaces.Discrete(3)
        self.binspace = gym.spaces.MultiBinary(2)
        self.multispace = gym.spaces.MultiDiscrete([3, 2])
        
        self.tuplespace = gym.spaces.Tuple((self.multispace, self.binspace, self.discspace))
        self.tuplecont = gym.spaces.Tuple((self.boxcont, self.binspace, self.discspace))
        self.dictspace = gym.spaces.Dict({'b': self.binspace, 'm': self.multispace, 'd': self.discspace})


    def test_enumerate_discrete_space_atomic(self):
        # multibinary
        b = list(helpers.enumerate_discrete_space(self.binspace))
        self.assertEqual(len(b), 4)
        self.assertEqual(len(b[0]), 2)
        # multidiscrete
        m = list(helpers.enumerate_discrete_space(self.multispace))
        self.assertEqual(len(m), 6)
        self.assertEqual(len(m[0]), 2)
        self.assertTrue(np.issubdtype(type(m[0][0]), np.integer))
        # integer box
        b = list(helpers.enumerate_discrete_space(self.boxspace))
        self.assertEqual(len(b), 5**2)
        self.assertEqual(len(b[0]), 2)
        self.assertTrue(np.issubdtype(type(b[0][0]), np.integer))


    def test_enumerate_discrete_space_composite(self):
        # tuple space
        t = list(helpers.enumerate_discrete_space(self.tuplespace))
        self.assertEqual(len(t), 72)
        self.assertEqual(len(t[0]), 5)
        doubletuplespace = gym.spaces.Tuple((self.tuplespace, self.tuplespace))
        t2 = list(helpers.enumerate_discrete_space(doubletuplespace))
        self.assertEqual(len(t2), 72*72)
        self.assertEqual(len(t2[0]), 10)
        # dict space
        d = list(helpers.enumerate_discrete_space(self.dictspace))
        self.assertEqual(len(d), 72)
        self.assertEqual(len(d[0]), 5)


    def test_size_space(self):
        self.assertEqual(helpers.size_space(self.tuplespace), 6*4*3)
        self.assertEqual(helpers.size_space(self.boxspace), 5**2)
        self.assertEqual(helpers.size_space(self.tuplecont), 16*4*3)


    def test_len_space_tuple(self):
        self.assertEqual(helpers.len_space_tuple(self.tuplespace), 5)
        self.assertEqual(helpers.len_space_tuple(self.boxspace), 2)
        self.assertEqual(helpers.len_space_tuple(self.dictspace), 5)
        self.assertEqual(helpers.len_space_tuple(self.discspace), 1)


    def test_bounds(self):
        b = helpers.bounds(self.tuplespace)
        self.assertEqual(b, ((0, 2), (0, 1), (0, 1), (0, 1), (0, 2)))


    def test_is_bounded(self):
        pass


    def test_is_continuous(self):
        pass


    def test_to_tuple(self):
        sample = self.tuplespace.sample()
        t = helpers.to_tuple(self.tuplespace, sample)
        self.assertEqual(len(t), 5)
        self.assertIsInstance(t, tuple)
        sample = self.dictspace.sample()
        t = helpers.to_tuple(self.dictspace, sample)
        self.assertEqual(len(t), 5)
        self.assertIsInstance(t, tuple)


    def test_to_space(self):
        sample = (1, 1, 1, 1, 1)
        space = helpers.to_space(self.tuplespace, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space[0], tuple)
        self.assertIsInstance(space[1], tuple)
        self.assertTrue(np.issubdtype(type(space[2]), np.integer))
        space = helpers.to_space(self.dictspace, sample)
        self.assertIsInstance(space, OrderedDict)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space['b'], tuple)
        self.assertIsInstance(space['m'], tuple)
        self.assertTrue(np.issubdtype(type(space['d']), np.integer))
        sample = (1, 1, 1, 1, 1, 1, 1, 1)
        space = helpers.to_space(self.tuplecont, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 3)
        self.assertTrue(np.issubdtype(type(space[2]), np.integer))

    
    def test_max_discrete(self):
        over = [(1,), (2,), (3,), (4,)]
        func = lambda x: -x[0]
        state = tuple()
        maximum, arg = helpers.max_discrete(func, over, state)
        self.assertEqual(arg, (1,))
        self.assertEqual(maximum, -1)


    def test_max_continuous(self):
        func = lambda x: x[0]**2
        over = ((-2, 2),)
        state = tuple()
        maximum, arg = helpers.max_continuous(func, over, state)
        self.assertAlmostEqual(maximum, 4, places=4)


    def test_max_hybrid(self):
        def func(arg):
            x, y = arg
            return 10 - x**2 - y**2
        
        discrete = gym.spaces.Tuple((gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=int), gym.spaces.Discrete(5)))
        halfcontinuous = gym.spaces.Tuple((gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float), gym.spaces.Discrete(5)))
        continuous = gym.spaces.Box(low=np.array([-5, 0]), high=np.array([5, 5]), dtype=float)
        
        contd = helpers.is_continuous(discrete)
        conthc = helpers.is_continuous(halfcontinuous)
        contc = helpers.is_continuous(continuous)
        bounds = helpers.bounds(discrete)
        state = tuple()

        m1, a1 = helpers.max_hybrid(func, bounds, state, contd, helpers.enumerate_discrete_space(discrete))
        m2, a2 = helpers.max_hybrid(func, bounds, state, conthc, helpers.enumerate_discrete_space(halfcontinuous))
        m3, a3 = helpers.max_hybrid(func, bounds, state, contc, helpers.enumerate_discrete_space(continuous))

        self.assertAlmostEqual(m1, 10, places=4)
        self.assertAlmostEqual(m2, 10, places=4)
        self.assertAlmostEqual(m3, 10, places=4)
        self.assertTrue(np.allclose(a1, (0, 0), atol=1e-3))
        self.assertTrue(np.allclose(a2, (0, 0), atol=1e-3))
        self.assertTrue(np.allclose(a3, (0, 0), atol=1e-3))



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
                sind = int(10 + x[0])
                aind = int(x[1])
                return self.values[sind, aind]
            def update(self, x, y):
                sind = int(10 + x[0])
                aind = int(x[1])
                self.values[sind, aind] = y
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
        agent = Agent(self.env, self.approx, policy=UNIFORM)
        actions = []
        for i, ep in enumerate(agent.episodes()):
            if i >= 1000:
                break
            actions.append(agent.next_action(ep))
        mean = np.asarray(actions).mean()
        self.assertLess(mean, 1)
        self.assertGreater(mean, 0)


    def test_greedy_policy(self):
        agent = Agent(self.env, self.approx, policy=GREEDY, greedy_prob=1.0)
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


unittest.main(verbosity=2)
