import unittest
import numpy as np
import gym
from . import helpers


class TestHelpers(unittest.TestCase):


    def test_multibinary_enumeration(self):
        binspace = gym.spaces.MultiBinary(3)
        b = helpers.enumerate_discrete_space(binspace)
        self.assertTrue(len(b)==8)
        self.assertTrue(len(b[0])==3)


    def test_multidiscrete_enumeration(self):
        multispace = gym.spaces.MultiDiscrete([3, 4])
        m = helpers.enumerate_discrete_space(multispace)
        self.assertTrue(len(m)==12)
        self.assertTrue(len(m[0])==2)
        self.assertTrue(np.issubdtype(m[0][0], np.integer))


    def test_tuplespace_enumeration(self):
        binspace = gym.spaces.MultiBinary(2)
        multispace = gym.spaces.MultiDiscrete([3, 2])
        tuplespace = gym.spaces.Tuple((multispace, binspace))
        t = helpers.enumerate_discrete_space(tuplespace)
        self.assertTrue(len(t)==24)
        self.assertTrue(len(t[0])==2)
        doubletuplespace = gym.spaces.Tuple((tuplespace, tuplespace))
        t2 = helpers.enumerate_discrete_space(doubletuplespace)
        self.assertTrue(len(t2)==24*24)
        self.assertTrue(len(t2[0])==2)

    

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