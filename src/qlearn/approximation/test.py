import unittest
import numpy as np
from . import Tabular
from . import Polynomial
from . import Neural



def gen_data(order, num, indim, outdim=1):
    # generating sample points and sorting them in increasing order
    x = np.random.randint(0, 10, (num, indim))
    # projecting points into higher dimension
    x_ = (x[:, None] ** np.arange(1, order + 1)[:, None]).reshape(x.shape[0], -1, order='c')
    # generating sample weights
    w = np.random.rand(order * indim, outdim) - 0.5
    # calculating target variable
    y = x_ @ w + np.random.rand(outdim) - 0.5  # add biases
    return x, x_, y, w



class TestModels(unittest.TestCase):


    def setUp(self):
        dimsize = 10
        ndims = 3
        x, _, y, _ = gen_data(2, 50, ndims, 1)
        self.shape = tuple([dimsize] * ndims)
        self.x = x
        self.y = y.ravel()
        self.epochs = 50


    def model_tester(self, model):
        errs = []
        for i in range(self.epochs):
            for x, y in zip(self.x, self.y):
                model.update(x, y)
            err = 0
            for x, y in zip(self.x, self.y):
                ypred = model.predict(x)
                err += abs(ypred - y)
            errs.append(err)
        self.assertTrue(errs[-1] < errs[0])


    def test_polynomial(self):
        model = Polynomial(2, 20, 10)
        self.model_tester(model)


    def test_neural(self):
        model = Neural((2, 3), 20, 10)
        self.model_tester(model)


    def test_tabular(self):
        model = Tabular(self.shape, lrate=0.3)
        self.model_tester(model)




unittest.main(verbosity=2)