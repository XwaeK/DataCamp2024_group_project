from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin

set_config(transform_output="pandas")


def preprocess_data(X, n_columns=10):
    categorical_columns = X.select_dtypes(include=[object]).columns
    X_clean = X.drop(columns=categorical_columns)
    X_clean = X_clean.iloc[:, :n_columns]
    X_clean.fillna(0, inplace=True)
    return X_clean


class MultiOutputRegressorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = MultiOutputRegressor(estimator)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator.predict(X)

    def predict(self, X):
        return self.estimator.predict(X)


def get_estimator():
    return make_pipeline(
        FunctionTransformer(preprocess_data),
        MultiOutputRegressorTransformer(RandomForestRegressor(n_estimators=50)),
    )
