from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

set_config(transform_output="pandas")


def preprocess_data(X, n_columns=10):
    categorical_columns = X.select_dtypes(include=[object]).columns
    X_clean = X.drop(columns=categorical_columns)
    X_clean = X_clean.iloc[:, :n_columns]
    X_clean.fillna(0, inplace=True)
    return X_clean


class MultiOutputRegressorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        if self.estimator is None:
            raise ValueError("Estimator cannot be None")
        self.estimator_ = MultiOutputRegressor(self.estimator)
        self.estimator_.fit(X, y)
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X)

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X)


def get_estimator():
    return make_pipeline(
        FunctionTransformer(preprocess_data),
        MultiOutputRegressorTransformer(
            RandomForestRegressor(n_estimators=50)
            ),
    )
