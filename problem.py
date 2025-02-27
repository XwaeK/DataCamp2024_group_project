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


def _load_data(file, start=None, stop=None, load_waveform=True):
    X_df = pd.read_hdf(file, key="data", start=start, stop=stop)

    y = X_df["map"]
    X_df = X_df.drop(columns=["map", "sbp", "dbp"], errors="ignore")

    if load_waveform:
        with h5py.File(file, "r") as f:
            X_df["ecg"] = list(f["ecg"][start:stop])
            X_df["ppg"] = list(f["ppg"][start:stop])

    # Replace None value in y by `-1
    y = y.fillna(-1).values

    return X_df, y


# READ DATA
def get_train_data(path=".", start=None, stop=None, load_waveform=True):

    # Avoid loading the data if it is already loaded
    # We use a global variable in rw as the problem.py module is created
    # dynamically and the global variables are not always reused.
    hash_train = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TRAIN", -1) == hash_train:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TRAIN = hash_train

    train_file = Path(path) / "data" / "train.h5"
    val_file = Path(path) / "data" / "validation.h5"
    if os.environ.get("RAMP_TEST_MODE", False):
        start_s, stop_s = 0, 1000
        start_t, stop_t = -1001, -1
        start_val, stop_val = 0, 100
    else:
        start_s, stop_s = 0, int(1.5e5)
        start_t, stop_t = -int(1.5e5 + 1), -1
        start_val, stop_val = None, None
    X_s, y_s = _load_data(train_file, start_s, stop_s, load_waveform)
    X_t, y_t = _load_data(train_file, start_t, stop_t, load_waveform)
    X_val, y_val = _load_data(val_file, start_val, stop_val, load_waveform)
    X_val["chunk"] = "val"
    X_train = pd.concat([X_s, X_t, X_val], axis=0, ignore_index=True)
    y_train = np.concatenate([y_s, y_t, y_val], axis=0)

    rw.X_TRAIN, rw.Y_TRAIN = X_train, y_train
    return X_train, y_train


def get_test_data(path=".", start=None, stop=None, load_waveform=True):

    hash_test = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TEST", -1) == hash_test:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TEST = hash_test

    file = "test.h5"
    file = Path(path) / "data" / file
    if os.environ.get("RAMP_TEST_MODE", False):
        start, stop = 0, 100
    rw.X_TEST, rw.Y_TEST = _load_data(file, start, stop, load_waveform)
    return rw.X_TEST, rw.Y_TEST


def preprocess():
    # Charger les fichiers CSV
    print("Chargement des fichiers CSV...")
    interventions_features = pd.read_csv(
        "data/interventions-hebdo-2010-2017.csv", sep=";"
    )
    interventions_features_test = pd.read_csv(
        "data/interventions-sdis91.csv", encoding="ISO-8859-1", sep=";"
    )
    communes_features = pd.read_csv("data/91-communes-features.csv")
    variability_features = pd.read_csv("data/91-variability-features.csv")
    print("   -> Fichiers CSV chargés avec succès.")

    print("Traitement des données...")
    print("   - Interventions_features")
    # Suppression des lignes qui n'ont que des valeurs NaN dans
    # interventions_features
    interventions_features = interventions_features.dropna(how="all")

    # Fonction de mise en forme de la date
    def mise_en_forme_date(df):
        df["date"] = df["ope_annee"] * 100 + df["ope_semaine"]
        df = df.drop(columns=["ope_annee", "ope_semaine"])
        df = df[["date"] + [col for col in df.columns if col != "date"]]
        return df

    interventions_features = mise_en_forme_date(interventions_features)
    interventions_features_test = mise_en_forme_date(
        interventions_features_test
    )

    # Remplacer les valeurs NAN par 'AUTR' dans interventions_features_test
    interventions_features_test["ope_categorie"] = interventions_features_test[
        "ope_categorie"
    ].fillna("AUTR")

    # Création d'un DataFrame avec les communes et leur code INSEE
    communes = pd.DataFrame(columns=["code_insee", "nom_commune"])
    communes_list = []

    for code_insee in interventions_features_test["ope_code_insee"].unique():
        nom_commune = interventions_features_test[
            interventions_features_test["ope_code_insee"] == code_insee
        ]["ope_nom_commune"].unique()[0]
        communes_list.append(
            {"code_insee": code_insee, "nom_commune": nom_commune}
        )

    communes = pd.concat(
        [communes, pd.DataFrame(communes_list)], ignore_index=True
    )
    communes["code_insee"] = communes["code_insee"].astype(int)

    # Elaboration du training set et du test set
    X_train = pd.DataFrame(columns=["semaine", "code_insee"])
    X_test = pd.DataFrame(columns=["semaine", "code_insee"])

    # X_train avec l'ensemble des valeurs (2010-2017) d'interventions_features
    for date in interventions_features["date"].unique():
        for code_insee in interventions_features_test[
            "ope_code_insee"
        ].unique():
            new_row = pd.DataFrame(
                {"semaine": [date], "code_insee": [code_insee]}
            )
            X_train = pd.concat([X_train, new_row], ignore_index=True)

    # X_test avec les valeurs de 2018 d'interventions_features_test
    for date in interventions_features_test["date"].unique():
        for code_insee in interventions_features_test[
            "ope_code_insee"
        ].unique():
            if date >= 201801 and date <= 201852:
                new_row = pd.DataFrame(
                    {"semaine": [date], "code_insee": [code_insee]}
                )
                X_test = pd.concat([X_test, new_row], ignore_index=True)

    X_train["semaine"] = X_train["semaine"].astype(int)
    X_train["code_insee"] = X_train["code_insee"].astype(int)
    X_test["semaine"] = X_test["semaine"].astype(int)
    X_test["code_insee"] = X_test["code_insee"].astype(int)

    # Fusionner X_train/X_test et communes pour ajouter le nom de la commune
    X_train = pd.merge(X_train, communes, on="code_insee", how="left")
    X_test = pd.merge(X_test, communes, on="code_insee", how="left")

    # Création des "target" y_train et y_test
    type_ope = ["SUAP", "INCN", "INCU", "ACCI", "AUTR"]

    y_train = X_train[["semaine", "code_insee"]]
    y_test = X_test[["semaine", "code_insee"]]

    def implement_target(df, y):
        for t in type_ope:
            y["nb_ope_" + t] = 0

        for date in df["date"].unique():
            for code_insee in df["ope_code_insee"].unique():
                for t in type_ope:
                    df_ = df[
                        (df["date"] == date)
                        & (df["ope_code_insee"] == code_insee)
                    ]

                    if not df_.empty:
                        matching_rows = df_[df_["ope_categorie"] == t]
                        if not matching_rows.empty:
                            y.loc[
                                (y["semaine"] == date)
                                & (y["code_insee"] == code_insee),
                                "nb_ope_" + t,
                            ] = matching_rows["nb_ope"].values[0]
                        else:
                            y.loc[
                                (y["semaine"] == date)
                                & (y["code_insee"] == code_insee),
                                "nb_ope_" + t,
                            ] = 0
                    else:
                        y.loc[
                            (y["semaine"] == date)
                            & (y["code_insee"] == code_insee),
                            "nb_ope_" + t,
                        ] = 0
        return y

    y_train = implement_target(interventions_features, y_train)
    y_test = implement_target(interventions_features_test, y_test)

    # Suppression des colonnes devenues inutiles
    y_train = y_train.drop(columns=["semaine", "code_insee"])
    y_test = y_test.drop(columns=["semaine", "code_insee"])
    print("      -> interventions_features processed successfully.")

    print("   - Merging communes_features...")
    # Vérification et intégration des données de communes_features
    if not communes_features["code_insee"].isin(X_train["code_insee"]).all():
        print("communes_features a des communes non présentes dans X_train")

    # Création des colonnes dans X_train et X_test pour les communes_features
    for col in communes_features.columns:
        X_train["com_" + col] = np.nan
        X_test["com_" + col] = np.nan

    # Remplir les colonnes avec les informations de communes_features
    for date in communes_features["annee"].unique():
        for code_insee in communes_features["code_insee"].unique():
            df = communes_features[
                (communes_features["annee"] == date)
                & (communes_features["code_insee"] == code_insee)
            ]

            if not df.empty:
                for col in communes_features.columns:
                    if communes_features[col].dtype == "object":
                        X_train.loc[
                            (X_train["semaine"] // 100 == date)
                            & (X_train["code_insee"] == code_insee),
                            "com_" + col,
                        ] = str(df[col].values[0])
                        X_test.loc[
                            (X_test["semaine"] // 100 == date)
                            & (X_test["code_insee"] == code_insee),
                            "com_" + col,
                        ] = str(df[col].values[0])
                    else:
                        X_train.loc[
                            (X_train["semaine"] // 100 == date)
                            & (X_train["code_insee"] == code_insee),
                            "com_" + col,
                        ] = df[col].values[0]
                        X_test.loc[
                            (X_test["semaine"] // 100 == date)
                            & (X_test["code_insee"] == code_insee),
                            "com_" + col,
                        ] = df[col].values[0]

    print("      -> communes_features merged successfully.")
    print("   - Merging variability_features...")
    # Traitement des données de variability_features
    variability_features = variability_features.dropna(
        subset=["pollution_code_insee"]
    )

    if (
        not variability_features["pollution_code_insee"]
        .isin(X_train["code_insee"])
        .all()
    ):
        print("variability_features a des communes non présentes dans X_train")

    variability_features["date"] = (
        variability_features["cal_annee"] * 100
        + variability_features["cal_semaine"]
    )
    variability_features = variability_features.rename(
        columns={"pollution_code_insee": "code_insee"}
    )

    for col in variability_features.columns:
        if variability_features[col].dtype == "float64":
            if variability_features[col].apply(lambda x: x.is_integer()).all():
                variability_features[col] = variability_features[col].astype(
                    int
                )

    for col in variability_features.columns:
        X_train["var_" + col] = np.nan
        X_test["var_" + col] = np.nan

    # Remplir les colonnes avec les informations de variability_features
    for date in variability_features["date"].unique():
        for code_insee in variability_features["code_insee"].unique():
            df = variability_features[
                (variability_features["date"] == date)
                & (variability_features["code_insee"] == code_insee)
            ]

            if not df.empty:
                for col in variability_features.columns:
                    if variability_features[col].dtype == "object":
                        X_train.loc[
                            (X_train["semaine"] == date)
                            & (X_train["code_insee"] == code_insee),
                            "var_" + col,
                        ] = str(df[col].values[0])
                        X_test.loc[
                            (X_test["semaine"] == date)
                            & (X_test["code_insee"] == code_insee),
                            "var_" + col,
                        ] = str(df[col].values[0])
                    else:
                        X_train.loc[
                            (X_train["semaine"] == date)
                            & (X_train["code_insee"] == code_insee),
                            "var_" + col,
                        ] = df[col].values[0]
                        X_test.loc[
                            (X_test["semaine"] == date)
                            & (X_test["code_insee"] == code_insee),
                            "var_" + col,
                        ] = df[col].values[0]
    print("      -> variability_features merged successfully.")

    # Sauvegarder les datasets traités en H5
    X_train.to_hdf("data/X_train.h5", key="data", mode="w")
    X_test.to_hdf("data/X_test.h5", key="data", mode="w")
    y_train.to_hdf("data/y_train.h5", key="target", mode="w")
    y_test.to_hdf("data/y_test.h5", key="target", mode="w")

    print("Preprocessing and data merging completed successfully !")
    print(
        "-> Datasets saved in 'data' folder as "
        "X_train.h5, X_test.h5, y_train.h5 and y_test.h5"
    )
