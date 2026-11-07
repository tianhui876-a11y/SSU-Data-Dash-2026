"""Improved, leakage-aware models for the two SSU Data Dash tracks.
Drop-in replacement for official models.py; compatible with run.py.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import features


def make_model(track):
    if track == 'off_market':
        return PriceModel(features.off_market_features, list_aware=False)
    if track == 'list_aware':
        return PriceModel(features.list_aware_features, list_aware=True)
    raise ValueError(f'Unknown track: {track}')


class PriceModel:
    def __init__(self, feature_function, list_aware=False):
        self.feature_function = feature_function
        self.list_aware = list_aware

    def fit(self, train):
        X = self.feature_function(train)
        categorical = [c for c in features.CATEGORICAL if c in X.columns]
        numeric = [c for c in X.columns if c not in categorical]
        self.preprocessor = ColumnTransformer([
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                   encoded_missing_value=np.nan), categorical),
            ('num', SimpleImputer(strategy='median', add_indicator=False), numeric),
        ], sparse_threshold=0)
        transformed = self.preprocessor.fit_transform(X)
        self.model = HistGradientBoostingRegressor(
            loss='squared_error', learning_rate=0.06, max_iter=280,
            max_leaf_nodes=31, min_samples_leaf=35, l2_regularization=2.0,
            categorical_features=None, random_state=42
        )
        target = np.log(train['sold_price'].to_numpy(dtype=float))
        self.model.fit(transformed, target)
        return self

    def predict(self, homes):
        X = self.feature_function(homes)
        transformed = self.preprocessor.transform(X)
        pred = np.exp(self.model.predict(transformed))
        return np.clip(pred, 10000, 100000000)
