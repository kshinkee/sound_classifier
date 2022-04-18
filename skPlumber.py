"""
Reference:
https://qiita.com/kazetof/items/fcfabfc3d737a8ff8668
"""

from sklearn.base import BaseEstimator, TransformerMixin


class SkPlumberBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self
