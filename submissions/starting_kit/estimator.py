from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

set_config(transform_output="pandas")
_label_names = [
    "nb_ope_SUAP",
    "nb_ope_INCN",
    "nb_ope_INCU",
    "nb_ope_ACCI",
    "nb_ope_AUTR",
]


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


class MyEstimator:
    def __init__(self):
        self.pipe = make_pipeline(
            FunctionTransformer(preprocess_data),
            MultiOutputRegressorTransformer(
                RandomForestRegressor(n_estimators=50)
            ),
        )
        self.column_names = _label_names

    def fit(self, X, y):
        self.pipe.fit(X, y)
        # print(f"Debug: fit on {X.shape}/{y.shape}")
        return self

    def predict(self, X):
        pred = self.pipe.predict(X)
        # pred_df = pd.DataFrame(pred, columns=self.column_names)

        # print(f"Debug: has predicted {pred_df.shape} {pred_df.columns}")
        # return pred_df
        return pred


def get_estimator():
    model = MyEstimator()
    return model
