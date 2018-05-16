"""
Implements the base Approximator class.
"""



class Approximator:

    def update(self, *args, **kwargs):
        pass
    

    def predict(self, *args):
        pass
    

    def __getitem__(self, *args):
        self.predict(*args)
    

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)