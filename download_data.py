import pandas as pd
import numpy as np
import requests
import os

# Liens vers les trois datasets
datasets = [
    "https://www.data.gouv.fr/fr/datasets/r/d7e5740b-9c28-4caa-9801-"
    "8354390a9bcb",
    "https://www.data.gouv.fr/fr/datasets/r/5d8dc837-ff82-420f-9130-"
    "5ea456293288",
    "https://www.data.gouv.fr/fr/datasets/r/cc19ff95-8ff8-43b7-a180-"
    "c84258a5c0c3",
    "https://www.data.gouv.fr/fr/datasets/r/90d589c8-0849-4392-852c-"
    "78bfcd820785",
]

# Dossier de destination pour les fichiers téléchargés
download_folder = "data"

file_names = [
    "interventions-hebdo-2010-2017.csv",
    "91-variability-features.csv",
    "91-communes-features.csv",
    "interventions-sdis91.csv",
]

# Créer le dossier si nécessaire
os.makedirs(download_folder, exist_ok=True)


def download_file(url, file_name, dest_folder):
    # Extraire le nom du fichier depuis l'URL
    filename = file_name
    filepath = os.path.join(dest_folder, filename)

    # Télécharger le fichier
    print(f"Téléchargement de {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Vérifier si la requête a réussi

    # Sauvegarder le fichier localement
    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"Fichier téléchargé et sauvegardé sous {filepath}")


# Télécharger tous les datasets
for url, file_name in zip(datasets, file_names):
    download_file(url, file_name, download_folder)

print("Tous les fichiers ont été téléchargés avec succès.")


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
        for code_insee in interventions_features[
            "ope_code_insee"
        ].unique():
            new_row = pd.DataFrame(
                {"semaine": [date], "code_insee": [code_insee]}
            )
            X_train = pd.concat([X_train, new_row], ignore_index=True)

    # X_test avec les valeurs de 2018 d'interventions_features_test
    for date in interventions_features_test["date"].unique():
        for code_insee in interventions_features[
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
    X_train.to_hdf("data/train.h5", key="data", mode="w")
    X_test.to_hdf("data/test.h5", key="data", mode="w")
    y_train.to_hdf("data/train.h5", key="target", mode="a")
    y_test.to_hdf("data/test.h5", key="target", mode="a")

    print("Preprocessing and data merging completed successfully !")
    print(
        "-> Datasets saved in 'data' folder as "
        "train.h5 & test.h5"
    )
