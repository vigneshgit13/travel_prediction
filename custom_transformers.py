# custom_transformers.py

from sklearn.base import BaseEstimator, TransformerMixin

class TotalVisitingAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Total_visiting'] = X['NumberOfPersonVisiting'] + X['NumberOfChildrenVisiting']
        return X
