import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path


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


def create_idx(
    df: pd.DataFrame, col_year: str, col_week: str, col_insee: str
) -> pd.DataFrame:
    """Create the 'idx' column that serve as a unique indenfifier for SDIS91
    project in the form of the integer YYYYWWWNNNNN where YYYY is the year,
    WW the number of the week and NNNNN is the code insee of the commune.

    args:
    - df: the orignal dataframe that will be modified
    - col_year: name of the column of df that contains the year as int
    - col_week: name of the column of df that contains the week number as int
    - col_insee: name of the column of df that contains the insee code as int
    """
    df["idx"] = (
        df[col_year] * 10_000_000 + df[col_week] * 100_000 + df[col_insee]
    ).astype("int64")
    return df


def process_data():
    """Load the csv files and make the X/y files."""

    # -----------------------
    # Step 1 : communes features
    yearly_communes = pd.read_csv(Path("data", "91-communes-features.csv"))
    yearly_communes = yearly_communes.drop(
        columns=["code_postal", "dept", "region"]
    )
    # Create a year insee index
    yearly_communes["YYYYNNNNN"] = (
        yearly_communes["annee"] * 100_000 + yearly_communes["code_insee"]
    )

    # -----------------------
    # Step 2 : variables features
    variable_communes = pd.read_csv(Path("data", "91-variability-features.csv"))
    # get all the insee code for future reference
    all_insee = (
        variable_communes["pollution_code_insee"].dropna().unique().astype(int)
    )
    # Split calendar data
    cal_data = variable_communes[
        [
            col_name
            for col_name in variable_communes.columns
            if col_name.startswith("cal_")
        ]
    ]
    cal_data = cal_data.drop_duplicates()
    # Create a year week index
    cal_data["YYYYWW"] = cal_data["cal_annee"] * 100 + cal_data["cal_semaine"]

    # Split the variable data
    var_data = variable_communes.drop(
        columns=[
            col_name
            for col_name in variable_communes.columns
            if col_name.startswith("cal_semaine_")
        ]
    )
    # Drop emtpy insee
    var_data = var_data[~var_data["pollution_code_insee"].isna()]
    # creation of idx column
    var_data = create_idx(
        var_data, "cal_annee", "cal_semaine", "pollution_code_insee"
    )

    # ------------------------
    # Step 3 : construct the final dataset

    # Step 3.1 : create the index
    # Expected data rows
    # Add empty rows for weeks of communes without interventions
    # Expect {(3 * 53 + 6 * 52) * 196} rows,
    # Year 2010, 2015 and 2016 have 53 weeks

    # Create expected index
    all_years = np.arange(2010, 2019)
    all_weeks = np.array(
        [
            f"{year}{week:02d}"
            for year in all_years
            for week in range(1, 54 if year in [2010, 2015, 2016] else 53)
        ]
    ).astype(int)
    # final year week insee index
    all_weeks_communes = (all_weeks[:, None] * 100_000 + all_insee).flatten()

    # ------------------------
    # Step 3.2 merge with calendar data
    df_index = pd.DataFrame(all_weeks_communes, columns=["idx"])
    df_index["YYYYWW"] = df_index["idx"] // 100_000
    # Get the calendar data
    X_df = pd.merge(df_index, cal_data, how="left", on="YYYYWW").drop(
        columns=["YYYYWW"]
    )

    # ------------------------
    # Step 3.3 merge with other variable features
    X_df = pd.merge(
        X_df,
        var_data.drop(columns=["cal_annee", "cal_semaine"]),
        on="idx",
        how="left",
    )

    # ------------------------
    # Step 3.4 merge with the yearly features
    # Create the year insee index
    X_df["YYYYNNNNN"] = X_df["cal_annee"] * 100_000 + X_df["cal_semaine"]
    X_df = pd.merge(X_df, yearly_communes, how="left", on="YYYYNNNNN").drop(
        columns=["YYYYNNNNN"]
    )
    # Ensure consistent type
    X_df["pollution_commune"] = X_df["pollution_commune"].astype(str)
    X_df["commune_nom"] = X_df["commune_nom"].astype(str)
    print("X sucessfully created.")

    # ------------------------
    # Step 4 Construct the target
    inter_train = pd.read_csv(
        Path("data", "interventions-hebdo-2010-2017.csv"), sep=";"
    )
    # drop the na code insee
    inter_train = inter_train[~inter_train["ope_code_insee"].isna()]

    inter_test = pd.read_csv(
        Path("data", "interventions-sdis91.csv"), encoding="ISO-8859-1", sep=";"
    )
    # Keep only 2018 interventions
    inter_test = inter_test[inter_test["ope_annee"] == 2018]

    # Drop the commune 91310, not in Essone
    inter_test = inter_test[inter_test["ope_code_insee"] != 91310]

    # Merge the interventions for processing
    interventions = pd.concat(
        [inter_train.drop(columns=["ope_code_postal"]), inter_test], axis=0
    ).astype(
        {
            col: "int64"
            for col in inter_test.columns
            if col not in ["ope_categorie", "ope_nom_commune"]
        }
    )
    # Fill undetermined interventions as "AUTR"
    interventions["ope_categorie"] = interventions["ope_categorie"].fillna(
        "AUTR"
    )

    interventions = create_idx(
        interventions, "ope_annee", "ope_semaine", "ope_code_insee"
    )
    y = interventions[["idx", "ope_categorie", "nb_ope"]].fillna(0)
    # Convert columns to int where possible
    y = y.astype({"nb_ope": "int64", "idx": "int64"})

    # pivot the target to get 5 columns, 1 per ope categeorie
    y = y.groupby(by=["idx", "ope_categorie"]).sum().reset_index()
    y = (
        y.pivot(index="idx", columns="ope_categorie", values="nb_ope")
        .fillna(0)
        .astype("int64")
    )
    # Create the missing rows
    y_df = (
        pd.merge(df_index, y, how="left", left_on="idx", right_on="idx")
        .fillna(0)
        .drop(columns=["YYYYWW"])
    )
    print("y sucessfully created")

    # ------------------------
    # Step 5 Split the dataset
    TRAIN_END_DATE = 2018_01_00000
    PUBLIC_TEST_END_DATE = 2018_04_00000

    X_train = X_df[X_df["idx"] < TRAIN_END_DATE]
    X_test = X_df[
        (X_df["idx"] > TRAIN_END_DATE) & (X_df["idx"] < PUBLIC_TEST_END_DATE)
    ]
    X_test_private = X_df[X_df["idx"] > PUBLIC_TEST_END_DATE]
    print(f"{X_train.shape=}, {X_test.shape=}, {X_test_private.shape=}")

    y_train = y_df[y_df["idx"] < TRAIN_END_DATE]
    y_test = y_df[
        (y_df["idx"] > TRAIN_END_DATE) & (X_df["idx"] < PUBLIC_TEST_END_DATE)
    ]
    y_test_private = y_df[X_df["idx"] > PUBLIC_TEST_END_DATE]
    print(f"{y_train.shape=}, {y_test.shape=}, {y_test_private.shape=}")

    # Save dataset in h5
    # create the data/public dir if not exists
    os.makedirs("data/public", exist_ok=True)
    # public data
    X_train.to_hdf("data/public/train.h5", key="data", mode="w")
    X_test.to_hdf("data/public/test.h5", key="data", mode="w")
    y_train.to_hdf("data/public/train.h5", key="target", mode="a")
    y_test.to_hdf("data/public/test.h5", key="target", mode="a")
    # private test data
    X_test_private.to_hdf("data/test.h5", key="data", mode="w")
    y_test_private.to_hdf("data/test.h5", key="target", mode="w")


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
        for code_insee in interventions_features["ope_code_insee"].unique():
            new_row = pd.DataFrame(
                {"semaine": [date], "code_insee": [code_insee]}
            )
            X_train = pd.concat([X_train, new_row], ignore_index=True)

    # X_test avec les valeurs de 2018 d'interventions_features_test
    for date in interventions_features_test["date"].unique():
        for code_insee in interventions_features["ope_code_insee"].unique():
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
    print("-> Datasets saved in 'data' folder as train.h5 & test.h5")


if __name__ == "__main__":
    """Download the data for the first time"""
    # Liens vers les trois datasets
    datasets = {
        "interventions-hebdo-2010-2017.csv": "https://www.data.gouv.fr/fr/datasets/r/d7e5740b-9c28-4caa-9801-8354390a9bcb",
        "91-variability-features.csv": "https://www.data.gouv.fr/fr/datasets/r/5d8dc837-ff82-420f-9130-5ea456293288",
        "91-communes-features.csv": "https://www.data.gouv.fr/fr/datasets/r/cc19ff95-8ff8-43b7-a180-c84258a5c0c3",
        "interventions-sdis91.csv": "https://www.data.gouv.fr/fr/datasets/r/90d589c8-0849-4392-852c-78bfcd820785",
    }

    # Dossier de destination pour les fichiers téléchargés
    download_folder = "data"
    # Créer le dossier si nécessaire
    os.makedirs(download_folder, exist_ok=True)
    # Télécharger tous les datasets
    for file_name, url in datasets.items():
        download_file(url, file_name, download_folder)

    print(f"{len(datasets)} files sucessfully downloaded.")

    # Process the files and save as h5
    process_data()
