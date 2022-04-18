"""
References:

"""

from skPlumber import SkPlumberBase


class LocalNB(SkPlumberBase):

    def __init__(self, param1=10):
        self.param1 = param1

    def transform(self, X):
        Y = self.param1*X
        return Y
