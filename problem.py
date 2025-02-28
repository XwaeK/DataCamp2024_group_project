import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import rampwf as rw
import warnings
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


problem_title = "sdis91-estimation"

# A type (class) which will be used to create wrapper objects for y_pred
_label_names = [
    "nb_ope_SUAP",
    "nb_ope_INCN",
    "nb_ope_INCU",
    "nb_ope_ACCI",
    "nb_ope_AUTR",
]
Predictions = rw.prediction_types.make_regression(label_names=_label_names)

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
        # Ensure y_true and y_pred are dataframes to use columns names
        y_true = pd.DataFrame(y_true, columns=_label_names)
        y_pred = pd.DataFrame(y_pred, columns=_label_names)
        # Initialisation de la somme des erreurs pondérées
        total_weighted_error = 0
        # print(f"Debug: {y_true}")
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
    """Get the cross validation scheme, preserving the commune using
    GroupShuffleSplit. Group on "code_insee" """

    # Convert the insee code to numpy array
    groups = np.array(X["code_insee"])
    print(f"Split according to insee {groups.shape[0]} rows")
    gss = GroupShuffleSplit(n_splits=5, train_size=0.7, random_state=1)

    # for train_idx, test_idx in gss.split(X, y, groups=groups):
    #     print(f"Train indices: {train_idx[:10]}... (total {len(train_idx)})")
    #     print(f"Test indices: {test_idx[:10]}... (total {len(test_idx)})")
    #     yield train_idx, test_idx
    return gss.split(X, y, groups=groups)


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
    X : DataFrame (, 172)

    y : DataFrame (, 5)

    """
    file = split + ".h5"
    data_path = Path(path) / "data" / "public"  # TODO Check the ramp process
    X = pd.read_hdf(data_path / file, key="data", mode="r").reset_index(
        drop=True
    )
    y = pd.read_hdf(data_path / file, key="target", mode="r").reset_index(
        drop=True
    )
    y = y.to_numpy()

    if os.environ.get("RAMP_TEST_MODE", False):
        # Launched with --quick-test option; only a small subset of the data
        # Extract simulation numbers from file paths
        quick_test_indices = [np.arange(100)]

        # Select subset of data
        X = X[quick_test_indices]
        y = y[quick_test_indices]

    print(f"Load {split} data, {X.shape=}, {y.shape=}")
    return X, y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")
