# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:53:09 2025

@author: Pol
"""

import pandas as pd
import numpy as np

def process_yearly_data(file_path):
    """
    Traite un fichier CSV pour extraire les années valides et compter les observations par année,
    en ne conservant que les suites de 4 chiffres comprises entre 1000 et 2030.

    Arguments :
    - file_path : Chemin du fichier CSV.

    Retourne :
    - Deux listes : une liste des années triées et une liste des nombres d'observations correspondants.
    """
    # Charger le fichier CSV en ignorant les deux premières lignes
    data = pd.read_csv(file_path, header=None, skiprows=2)

    # Garder uniquement la première colonne et extraire la première donnée avant l'espace
    data['year'] = data[0].str.split().str[0]

    # Filtrer pour garder uniquement les suites composées de 4 chiffres et exclure les valeurs hors plage [1000, 2030]
    data = data[data['year'].str.fullmatch(r'\d{4}', na=False)]  # Correspondance exacte à 4 chiffres
    data = data[data['year'].astype(int).between(1000, 2030)]  # Filtrer les années dans la plage

    # Convertir en entier
    data['year'] = data['year'].astype(int)

    # Compter les occurrences de chaque année
    year_counts = data['year'].value_counts()

    # Obtenir les années de la plus ancienne à la plus récente
    years_sorted = np.arange(year_counts.index.min(), year_counts.index.max() + 1)
    observations = [year_counts.get(year, 0) for year in years_sorted]

    return years_sorted, observations


def invert_time_data(observations):
    """
    Traite un vecteur d'observations correspondant à des années dans le sens forward
    pour renvoyer un vecteur d'observaations dans le sens backward, avec l'année la plus
    récente au temps 0
    
    Arguments :
    - observations : Vecteur d'observations forward
    
    Retourne :
    - count_data : Vecteur d'observations backward
    """
    count_data = np.flip(observations)
    return count_data