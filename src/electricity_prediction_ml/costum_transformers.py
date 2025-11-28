from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class HourExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.to_datetime(X['timestamp']).dt.hour.to_frame()

    def get_feature_names_out(self, input_features=None):
        return ['hour']


class LogTransformer(BaseEstimator, TransformerMixin):
   
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        return [f"{col}_log" for col in input_features]
