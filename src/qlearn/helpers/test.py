import unittest
from collections import OrderedDict

import gym
import numpy as np

from . import spaces, maximum
from .parameters import *



class TestSpaces(unittest.TestCase):


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
        b = list(spaces.enumerate_discrete_space(self.binspace))
        self.assertEqual(len(b), 4)
        self.assertEqual(len(b[0]), 2)
        # multidiscrete
        m = list(spaces.enumerate_discrete_space(self.multispace))
        self.assertEqual(len(m), 6)
        self.assertEqual(len(m[0]), 2)
        self.assertTrue(np.issubdtype(type(m[0][0]), np.integer))
        # integer box
        b = list(spaces.enumerate_discrete_space(self.boxspace))
        self.assertEqual(len(b), 5**2)
        self.assertEqual(len(b[0]), 2)
        self.assertTrue(np.issubdtype(type(b[0][0]), np.integer))


    def test_enumerate_discrete_space_composite(self):
        # tuple space
        t = list(spaces.enumerate_discrete_space(self.tuplespace))
        self.assertEqual(len(t), 72)
        self.assertEqual(len(t[0]), 5)
        doubletuplespace = gym.spaces.Tuple((self.tuplespace, self.tuplespace))
        t2 = list(spaces.enumerate_discrete_space(doubletuplespace))
        self.assertEqual(len(t2), 72*72)
        self.assertEqual(len(t2[0]), 10)
        # dict space
        d = list(spaces.enumerate_discrete_space(self.dictspace))
        self.assertEqual(len(d), 72)
        self.assertEqual(len(d[0]), 5)


    def test_size_space(self):
        self.assertEqual(spaces.size_space(self.tuplespace), 6*4*3)
        self.assertEqual(spaces.size_space(self.boxspace), 5**2)
        self.assertEqual(spaces.size_space(self.tuplecont), 16*4*3)


    def test_len_space_tuple(self):
        self.assertEqual(spaces.len_space_tuple(self.tuplespace), 5)
        self.assertEqual(spaces.len_space_tuple(self.boxspace), 2)
        self.assertEqual(spaces.len_space_tuple(self.dictspace), 5)
        self.assertEqual(spaces.len_space_tuple(self.discspace), 1)


    def test_bounds(self):
        b = spaces.bounds(self.tuplespace)
        self.assertEqual(b, ((0, 2), (0, 1), (0, 1), (0, 1), (0, 2)))


    def test_is_bounded(self):
        pass


    def test_is_continuous(self):
        pass


    def test_to_tuple(self):
        sample = self.tuplespace.sample()
        t = spaces.to_tuple(self.tuplespace, sample)
        self.assertEqual(len(t), 5)
        self.assertIsInstance(t, tuple)
        sample = self.dictspace.sample()
        t = spaces.to_tuple(self.dictspace, sample)
        self.assertEqual(len(t), 5)
        self.assertIsInstance(t, tuple)


    def test_to_space(self):
        sample = (1, 1, 1, 1, 1)
        space = spaces.to_space(self.tuplespace, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space[0], tuple)
        self.assertIsInstance(space[1], tuple)
        self.assertTrue(np.issubdtype(type(space[2]), np.integer))
        space = spaces.to_space(self.dictspace, sample)
        self.assertIsInstance(space, OrderedDict)
        self.assertEqual(len(space), 3)
        self.assertIsInstance(space['b'], tuple)
        self.assertIsInstance(space['m'], tuple)
        self.assertTrue(np.issubdtype(type(space['d']), np.integer))
        sample = (1, 1, 1, 1, 1, 1, 1, 1)
        space = spaces.to_space(self.tuplecont, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 3)
        self.assertTrue(np.issubdtype(type(space[2]), np.integer))



class TestMaximum(unittest.TestCase):


    def test_max_discrete(self):
        over = [(1,), (2,), (3,), (4,)]
        func = lambda x: -np.asarray(x).ravel()       # returns an array of 1 elem
        state = tuple()
        m, arg = maximum.max_discrete(func, over, state)
        self.assertEqual(arg, (1,))
        self.assertEqual(m, -1)


    def test_max_continuous(self):
        func = lambda x: np.asarray(x).ravel()**2       # returns an array of 1 elem
        over = ((-2, 2),)
        state = tuple()
        m, arg = maximum.max_continuous(func, over, state)
        self.assertAlmostEqual(m, 4, places=4)


    def test_max_hybrid(self):
        def func(arg):
            x, y = arg[0]
            return np.asarray((10 - x**2 - y**2,))   # returns an array of 1 elem
        
        discrete = gym.spaces.Tuple((gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=int), gym.spaces.Discrete(5)))
        halfcontinuous = gym.spaces.Tuple((gym.spaces.Box(low=-5, high=5, shape=(1,), dtype=float), gym.spaces.Discrete(5)))
        continuous = gym.spaces.Box(low=np.array([-5, 0]), high=np.array([5, 5]), dtype=float)
        
        contd = spaces.is_continuous(discrete)
        conthc = spaces.is_continuous(halfcontinuous)
        contc = spaces.is_continuous(continuous)
        bounds = spaces.bounds(discrete)
        state = tuple()

        m1, a1 = maximum.max_hybrid(func, bounds, state, contd, spaces.enumerate_discrete_space(discrete))
        m2, a2 = maximum.max_hybrid(func, bounds, state, conthc, spaces.enumerate_discrete_space(halfcontinuous))
        m3, a3 = maximum.max_hybrid(func, bounds, state, contc, spaces.enumerate_discrete_space(continuous))

        self.assertAlmostEqual(m1, 10, places=4)
        self.assertAlmostEqual(m2, 10, places=4)
        self.assertAlmostEqual(m3, 10, places=4)
        self.assertTrue(np.allclose(a1, (0, 0), atol=1e-3))
        self.assertTrue(np.allclose(a2, (0, 0), atol=1e-3))
        self.assertTrue(np.allclose(a3, (0, 0), atol=1e-3))
        self.assertIsInstance(a1[0], int)
        self.assertIsInstance(a2[0], float)
        self.assertIsInstance(a2[1], int)
        self.assertIsInstance(a3[0], float)



class TestParameters(unittest.TestCase):


    def setUp(self):
        pass

    def schedule_tester(self, schedule: Schedule):
        # test ascending
        s = schedule(1, 4, 5)
        vals = [s(i) for i in range(-1, 6)]
        self.assertEqual(1, vals[0])
        self.assertEqual(1, vals[1])
        self.assertEqual(4, vals[-1])
        self.assertEqual(4, vals[-2])
        # test ascending with implicit `at` parameter
        s = schedule(1, 4, 5)
        vals = [s() for _ in range(6)]
        self.assertEqual(1, vals[0])
        self.assertEqual(4, vals[-1])
        self.assertEqual(4, vals[-2])
        # test descending
        s = schedule(4, 1, 5)
        vals = [s(i) for i in range(-1, 6)]
        self.assertEqual(1, vals[-1])
        self.assertEqual(1, vals[-2])
        self.assertEqual(4, vals[0])
        self.assertEqual(4, vals[1])
        # test descending with implicit `at` parameter
        s = schedule(4, 1, 5)
        vals = [s() for _ in range(6)]
        self.assertEqual(1, vals[-1])
        self.assertEqual(1, vals[-2])
        self.assertEqual(4, vals[0])


    def test_linear_schedule(self):
        self.schedule_tester(LinearSchedule)


    def test_logarithmic_schedule(self):
        self.schedule_tester(LogarithmicSchedule)


    def test_exponential_schedule(self):
        self.schedule_tester(ExponentialSchedule)


    def test_evaluate_schedule_kwargs(self):
        kwargs = {'log': LogarithmicSchedule(0, 4, 5),
                'lin': LinearSchedule(0, 4, 5),
                'extra': 5,
                'blah': 'blah'}
        for i in range(10):
            res = evaluate_schedule_kwargs(i, **kwargs)
            self.assertEqual(len(res), len(kwargs))
            self.assertEqual(res['lin'], kwargs['lin'](i))
            self.assertEqual(res['log'], kwargs['log'](i))



if __name__ == '__main__':
    unittest.main(verbosity=0)
