# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:09:37 2025

@author: Pol
"""

import numpy as np
from simulate_abundance_data import simulate_abundance_data
from infer_abundance import infer_abundance

def test_abundance_model(N0_values, N1_values, param_values, param_name, binary_param, binary_name, fixed_params):
    """
    Teste le modèle d'abondance en variant 2 paramètres quantitatifs (N0, N1),
    1 paramètre quantitatif additionnel et 1 paramètre binaire.

    Arguments :
    - N0_values, N1_values : Listes des valeurs possibles pour N0 et N1.
    - param_values : Liste des valeurs possibles pour le troisième paramètre quantitatif.
    - param_name : Nom du troisième paramètre quantitatif.
    - binary_param : Liste contenant les deux options binaires (par exemple [False, True]).
    - binary_name : Nom du paramètre binaire.
    - fixed_params : Dictionnaire contenant les autres paramètres fixés.

    Retourne :
    - results_matrix : Matrice 4D contenant les résultats d'inférence.
        Dimensions : [binary_param, param_values, N0_values, N1_values]
    - meta_info : Dictionnaire contenant les informations sur les dimensions et les variations.
    """
    # Initialiser la matrice des résultats : [binary_param, param_values, N0, N1, 3 (tau_est, N0_est, N1_est)]
    results_matrix = np.empty((len(binary_param), len(param_values), len(N0_values), len(N1_values), 3))

    # Parcourir toutes les combinaisons des paramètres
    for b_idx, b_val in enumerate(binary_param):
        for p_idx, param_val in enumerate(param_values):
            # Mettre à jour le paramètre variable et binaire
            current_params = fixed_params.copy()
            current_params[param_name] = param_val
            current_params[binary_name] = b_val

            for n0_idx, N0_real in enumerate(N0_values):
                for n1_idx, N1_real in enumerate(N1_values):
                    # Simulation des données
                    y = simulate_abundance_data(
                        N0_real=N0_real,
                        N1_real=N1_real,
                        T=current_params["T"],
                        tau_real=current_params["tau_real"],
                        theta=current_params["theta"],
                        missing_data=current_params["missing_data"],
                        p=current_params["p"],
                        Gamma=current_params["Gamma"],
                        beta_gamma=current_params["beta_gamma"]
                    )

                    # Inférence
                    tau_est, alpha_post_N0, beta_post_N0, alpha_post_N1, beta_post_N1, nan_percentage = infer_abundance(
                        y=y,
                        T=current_params["T"],
                        epsilon=current_params["epsilon"],
                        seuil_correction=current_params["seuil_correction"],
                        alpha_prior=current_params["alpha_prior"],
                        beta_prior=current_params["beta_prior"]
                    )

                    # Calcul des maxima des distributions a posteriori
                    N0_est = max((alpha_post_N0 - 1) / beta_post_N0, 0) if alpha_post_N0 > 1 else 0
                    N1_est = max((alpha_post_N1 - 1) / beta_post_N1, 0) if alpha_post_N1 > 1 else 0

                    # Stocker les résultats
                    results_matrix[b_idx, p_idx, n0_idx, n1_idx] = [tau_est, N0_est, N1_est]

    # Mettre en place les métadonnées pour faciliter l'affichage
    meta_info = {
        "dimensions": ["binary_param", "param_values", "N0", "N1"],
        "variations": {
            "binary_param": binary_param,
            "param_values": param_values,
            "N0": N0_values,
            "N1": N1_values
        },
        "results": ["tau_est", "N0_est", "N1_est"]
    }


    return results_matrix, meta_info
