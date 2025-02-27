from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config

set_config(transform_output="pandas")

def preprocess_data(X, n_columns=10):
    categorical_columns = X.select_dtypes(include=[object]).columns
    X_clean = X.drop(columns=categorical_columns)
    X_clean = X_clean.iloc[:, :n_columns]
    X_clean.fillna(0, inplace=True)
    return X_clean

def get_estimator():
    return make_pipeline(
        FunctionTransformer(preprocess_data),
        MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50)
        )
    )