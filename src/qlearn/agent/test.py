import unittest
from collections import OrderedDict
import numpy as np
import gym
from . import helpers


class TestHelpers(unittest.TestCase):


    def setUp(self):
        self.boxspace = gym.spaces.Box(low=0, high=5, shape=(5,), dtype=int)
        self.boxcont = gym.spaces.Box(low=0, high=5, shape=(5,), dtype=float)
        self.discspace = gym.spaces.Discrete(3)
        self.binspace = gym.spaces.MultiBinary(2)
        self.multispace = gym.spaces.MultiDiscrete([3, 2])
        
        self.tuplespace = gym.spaces.Tuple((self.multispace, self.binspace))
        self.dictspace = gym.spaces.Dict({'b': self.binspace, 'm': self.multispace})
        self.tuplecont = gym.spaces.Tuple((self.boxcont, self.binspace, self.discspace))


    def test_multibinary_enumeration(self):
        b = helpers.enumerate_discrete_space(self.binspace)
        self.assertEqual(len(b), 4)
        self.assertEqual(len(b[0]), 2)


    def test_multidiscrete_enumeration(self):
        m = helpers.enumerate_discrete_space(self.multispace)
        self.assertEqual(len(m), 6)
        self.assertEqual(len(m[0]), 2)
        self.assertTrue(np.issubdtype(type(m[0][0]), np.integer))


    def test_tuplespace_enumeration(self):
        t = helpers.enumerate_discrete_space(self.tuplespace)
        self.assertEqual(len(t), 24)
        self.assertEqual(len(t[0]), 2)
        doubletuplespace = gym.spaces.Tuple((self.tuplespace, self.tuplespace))
        t2 = helpers.enumerate_discrete_space(doubletuplespace)
        self.assertEqual(len(t2), 24*24)
        self.assertEqual(len(t2[0]), 2)


    def test_dictspace_enumeration(self):
        d = helpers.enumerate_discrete_space(self.dictspace)
        self.assertEqual(len(d), 24)
        self.assertIsInstance(d[0], OrderedDict)
        self.assertEqual(len(d[0]['b']), 2)


    def test_size_discrete_space(self):
        self.assertEqual(helpers.size_discrete_space(self.tuplespace), 24)
        self.assertEqual(helpers.size_discrete_space(self.boxspace), 5**5)
        self.assertEqual(helpers.size_discrete_space(self.tuplecont), np.inf)


    def test_len_space_tuple(self):
        self.assertEqual(helpers.len_space_tuple(self.tuplespace), 4)
        self.assertEqual(helpers.len_space_tuple(self.boxspace), 5)
        self.assertEqual(helpers.len_space_tuple(self.dictspace), 4)
        self.assertEqual(helpers.len_space_tuple(self.discspace), 1)


    def test_to_tuple(self):
        sample = self.tuplespace.sample()
        t = helpers.to_tuple(self.tuplespace, sample)
        self.assertEqual(len(t), 4)
        self.assertIsInstance(t, tuple)
        sample = self.dictspace.sample()
        t = helpers.to_tuple(self.dictspace, sample)
        self.assertEqual(len(t), 4)
        self.assertIsInstance(t, tuple)


    def test_to_space(self):
        sample = (1, 1, 1, 1)
        space = helpers.to_space(self.tuplespace, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 2)
        self.assertIsInstance(space[0], tuple)
        self.assertIsInstance(space[1], tuple)
        space = helpers.to_space(self.dictspace, sample)
        self.assertIsInstance(space, OrderedDict)
        self.assertEqual(len(space), 2)
        self.assertIsInstance(space['b'], tuple)
        self.assertIsInstance(space['m'], tuple)
        sample = (1, 1, 1, 1, 1, 1, 1, 1)
        space = helpers.to_space(self.tuplecont, sample)
        self.assertIsInstance(space, tuple)
        self.assertEqual(len(space), 3)
        self.assertTrue(np.issubdtype(type(space[2]), np.integer))

    
    def test_max_discrete(self):
        over = [(1,), (2,), (3,), (4,)]
        func = lambda x: -x[0]
        maximum, arg = helpers.max_discrete(func, over)
        self.assertEqual(arg, (1,))
        self.assertEqual(maximum, -1)



class TestAgent(unittest.TestCase):


    def setUp(self):
        # dummy env
        pass



unittest.main(verbosity=2)