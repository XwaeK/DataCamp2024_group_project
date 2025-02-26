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
