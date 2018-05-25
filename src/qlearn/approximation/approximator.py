"""
Implements the base Approximator class.
"""
from typing import Tuple
import numpy as np


class Approximator:


    def update(self, *args, **kwargs):
        pass
    

    def predict(self, *args):
        pass


    def max(self, over=()):
        pass
    

    def __getitem__(self, *args):
        self.predict(*args)
    

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)