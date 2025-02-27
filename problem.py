import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import rampwf as rw
import warnings

warnings.filterwarnings("ignore")


problem_title = "sdis91-estimation"

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()


class WMAE(rw.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="wmae", precision=4, weights=None):
        self.name = name
        self.precision = precision
        # Définir les poids par catégorie
        if weights is None:
            self.weights = {
                "nb_ope_SUAP": 0.55,
                "nb_ope_INCN": 1.00,
                "nb_ope_INCU": 0.94,
                "nb_ope_ACCI": 0.95,
                "nb_ope_AUTR": 0.95,
            }
        else:
            self.weights = weights

    def __call__(self, y_true, y_pred):
        # Initialisation de la somme des erreurs pondérées
        total_weighted_error = 0

        # Calculer le WMAE pour chaque catégorie et sommer
        for category, weight in self.weights.items():
            if category in y_true.columns:
                # Appliquer le masque pour ignorer les valeurs manquantes
                mask = y_true[category] != -1

                # Vérifier qu'il reste des valeurs après application du masque
                if mask.sum() > 0:
                    # Calculer l'erreur absolue pour cette catégorie
                    category_errors = np.abs(
                        y_true[category][mask] - y_pred[category][mask]
                    )

                    # Calculer le MAE pondéré pour cette catégorie
                    weighted_mae = weight * category_errors.mean()

                    # Ajouter à la somme totale
                    total_weighted_error += weighted_mae

        return total_weighted_error


score_types = [
    WMAE(name="weighted_mean_absolute_error", precision=5),
]


def get_cv(X, y):
    # Make sure the index is a range index so it is compatible with sklearn API
    X = X.reset_index(drop=True)

    chunks = X["chunk"].fillna("t")

    def split():
        train_idx = chunks[chunks != "val"].index
        val_idx = chunks[chunks == "val"].index
        yield train_idx, val_idx
        # yield X.query("chunk != 'val'").index,
        # X.query("chunk == 'val'").index

    return split()


# READ DATA
def _get_data(path=".", split="train"):
    """
    Get the data for the given split (train or test).
    This function reads the h5 files

    Parameters:
    -----------
    path : str, optional
        The base path to the data directory. Default is current directory (".").
    split : str, optional
        The data split to retrieve, either "train" or "test". Default is "train".

    Returns:
    --------
    X : DataFrame (, 135)

    y : DataFrame (, 6)

    """
    X_file = "X_" + split + ".h5"
    y_file = "y_" + split + ".h5"
    data_path = Path(path) / "data"
    X = pd.read_hdf(data_path / X_file)
    y = pd.read_hdf(data_path / y_file)

    if os.environ.get("RAMP_TEST_MODE", False):
        # Launched with --quick-test option; only a small subset of the data
        # Extract simulation numbers from file paths
        quick_test_indices = [np.arange(100)]

        # Select subset of data
        X = X[quick_test_indices]
        y = y[quick_test_indices]

    return X, y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")
